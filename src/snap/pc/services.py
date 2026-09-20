"""Snap's pipecat services — the two stages that are ours, not the framework's.

SnapWhisperSTT   faster-whisper today (small default / large-v3-turbo via --quality);
                 the NPU build swaps the internals for the AI Hub QNN context binary
                 via ONNX Runtime + QNN EP. Signature stays.
SnapLlamaLLM     llama.cpp GGUF today; GenieX/QAIRT on NPU later (`geniex infer
                 ai-hub-models/Qwen3-1.7B`). Implements the strict JSON tool protocol:
                 plain replies stream sentence-by-sentence; tool-call JSON is never
                 spoken raw — the router executes locally and the templated result is
                 emitted with no second LLM pass.
"""

import asyncio
import threading
import time
from collections.abc import AsyncGenerator, Callable
from datetime import datetime, timezone

import numpy as np

from snap import config, tools
from snap.pc.sentence_stream import SentenceSegmenter
from snap.timing import TIMINGS

from pipecat.frames.frames import (
    Frame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    TextFrame,
    TranscriptionFrame,
)
from pipecat.services.llm_service import LLMService
from pipecat.services.stt_service import STTService
from pipecat.services.tts_service import TTSService

# pipecat 1.x renamed OpenAILLMContext -> LLMContext; keep a single import surface.
try:  # newer pipecat
    from pipecat.processors.aggregators.llm_context import LLMContext  # type: ignore

    Context = LLMContext
except ImportError:  # older pipecat
    from pipecat.processors.aggregators.openai_llm_context import (  # type: ignore
        OpenAILLMContext as Context,
    )

try:  # pipecat ships a piper service; falls back if the extra isn't installed
    from pipecat.services.piper.tts import PiperTTSService  # type: ignore
except ImportError:  # pragma: no cover
    PiperTTSService = None

_DONE = object()  # sentinel: LLM token stream ended


def _now_iso() -> str:
    try:
        from pipecat.utils.time import time_now_iso8601

        return time_now_iso8601()
    except ImportError:
        return datetime.now(timezone.utc).isoformat()


class SnapWhisperSTT(STTService):
    """faster-whisper behind pipecat's STTService. Measures its own stage time."""

    def __init__(self, model_size: str | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        from faster_whisper import WhisperModel  # heavy import stays lazy

        self._model = WhisperModel(
            model_size or config.WHISPER_MODEL, device="cpu", compute_type="int8"
        )
        self._language = config.WHISPER_LANGUAGE

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame, None]:
        pcm = np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0
        with TIMINGS.stage("stt"):
            segments, _info = await asyncio.to_thread(
                lambda: self._model.transcribe(pcm, language=self._language, beam_size=1)
            )
            text = " ".join(s.text.strip() for s in segments).strip()
        if text:
            yield TranscriptionFrame(text, "user", _now_iso())

    async def warm_up(self) -> None:
        """One silent pass so mel filterbanks and decoder state are hot before turn 1."""
        await asyncio.to_thread(self._warm)

    def _warm(self) -> None:
        silence = np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 s
        for _seg in self._model.transcribe(silence, language="en", beam_size=1)[0]:
            break


class SnapLlamaLLM(LLMService):
    """Qwen3 via llama-cpp-python with Snap's tool protocol, behind pipecat's LLMService.

    `emit` receives sentence strings. In the pipecat pipeline it pushes TextFrames;
    in text mode (no pipeline) it is a plain console callback. One code path for both.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        from llama_cpp import Llama  # heavy import stays lazy

        self._llm = Llama(
            model_path=str(config.LLM_GGUF),
            n_ctx=config.LLM_N_CTX,
            n_threads=config.LLM_N_THREADS,
            verbose=False,
        )
        self._cancel = asyncio.Event()
        self._history: list[dict] = []

    # -- pipecat plumbing ----------------------------------------------------

    async def process_frame(self, frame, direction):  # noqa: ANN001
        if isinstance(frame, InterruptionFrame):
            self._cancel.set()  # barge-in kills generation at the next token
        await super().process_frame(frame, direction)
        # pipecat 1.x: the base LLMService does NOT route LLMContextFrame to a
        # subclass hook — the concrete service does (mirrors BaseOpenAILLMService).
        if isinstance(frame, LLMContextFrame):
            try:
                await self.push_frame(LLMFullResponseStartFrame())
                await self.start_processing_metrics()
                await self._process_context(frame.context)
            except Exception as e:  # noqa: BLE001
                await self.push_error(error_msg=f"Error during completion: {e}", exception=e)
            finally:
                await self.stop_processing_metrics()
                await self.push_frame(LLMFullResponseEndFrame())

    async def _process_context(self, context) -> None:  # noqa: ANN001
        user_text = ""
        for msg in reversed(list(context.get_messages())):
            if msg.get("role") == "user":
                user_text = msg.get("content", "")
                break
        await self.respond(user_text, emit=self.push_text_frame)

    async def push_text_frame(self, sentence: str) -> None:
        await self.push_frame(LLMTextFrame(sentence))

    # -- shared generation path (pipeline + text mode) ------------------------

    def _messages_for(self, user_text: str) -> list[dict]:
        return (
            [{"role": "system", "content": tools.SYSTEM_PROMPT}]
            + self._history
            + [{"role": "user", "content": user_text}]
        )

    def _remember(self, user_text: str, reply: str) -> None:
        self._history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        # Prompt hygiene, KV-cache aware: append-only by default (llama.cpp reuses
        # the cached prefix); trim rarely and in chunks so invalidation is rare.
        if len(self._history) > 24:
            del self._history[:8]

    async def warm_up(self) -> None:
        """One 1-token completion so chat-template + compute-graph caches are hot."""
        self._cancel.clear()
        for _part in self._llm.create_chat_completion(
            messages=[{"role": "user", "content": "hi"}], max_tokens=1
        ):
            break

    def _generate(self, messages: list[dict]):
        stream = self._llm.create_chat_completion(
            messages=messages, max_tokens=config.LLM_MAX_TOKENS, stream=True
        )
        for part in stream:
            if self._cancel.is_set():
                break
            yield part["choices"][0]["delta"].get("content") or ""

    async def respond(self, user_text: str, emit: Callable[[str], object]) -> str:
        """One full turn: stream tokens, emit sentences, route tool JSON. Returns reply.

        `emit` may be sync (text mode) or async (pipeline); normalized here.
        Thinking blocks (`<think>…`) are stripped before speech by the segmenter.
        """
        if not asyncio.iscoroutinefunction(emit):
            sync_emit = emit

            async def emit(sentence: str) -> None:  # noqa: F811 — normalized wrapper
                sync_emit(sentence)

        self._cancel.clear()
        messages = self._messages_for(user_text)

        queue: asyncio.Queue = asyncio.Queue()
        loop = asyncio.get_running_loop()

        def worker() -> None:
            try:
                for token in self._generate(messages):
                    loop.call_soon_threadsafe(queue.put_nowait, token)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _DONE)

        threading.Thread(target=worker, daemon=True).start()

        t0 = time.perf_counter()
        collected: list[str] = []
        segmenter = SentenceSegmenter()
        tool_mode: bool | None = None  # decided by the first non-empty token
        first_token_done = False

        while True:
            token = await queue.get()
            if token is _DONE:
                break
            if not first_token_done:
                TIMINGS.record("llm_first_token", (time.perf_counter() - t0) * 1000)
                first_token_done = True
            collected.append(token)
            if tool_mode is None and token.strip():
                # A tool call is one JSON object — mute sentence streaming; the
                # router speaks the templated result below.
                tool_mode = token.lstrip().startswith("{")
            if not tool_mode:
                for sentence in segmenter.feed(token):
                    await emit(sentence)
        if tool_mode is None:
            tool_mode = False
        TIMINGS.record("llm_total", (time.perf_counter() - t0) * 1000)

        raw = "".join(collected).strip()
        if not tool_mode:
            tail = segmenter.flush()
            if tail:
                await emit(tail)
        reply = tools.route(raw) if raw else "…"
        if reply and reply != raw:  # tool path: templated result, no second LLM pass
            await emit(reply)

        self._remember(user_text, reply)
        if config.PRINT_TURN_REPORT:
            import json

            print(f"[latency] {json.dumps(TIMINGS.latency_report())}")
        return reply


class ConsoleTTSService(TTSService):
    """Pipeline-safe zero-audio TTS: prints instead of speaking. Spike fallback."""

    async def run_tts(self, text: str, context_id: str = "") -> AsyncGenerator[Frame, None]:
        with TIMINGS.stage("tts_console"):
            print(f"\n[Snap] {text}\n> ", end="", flush=True)
        return
        yield  # noqa: B901 — marks this an async generator with no audio output


def build_tts() -> TTSService:
    if config.PIPER_VOICE and PiperTTSService is not None:
        return PiperTTSService(voice_path=str(config.PIPER_VOICE))
    return ConsoleTTSService()


__all__ = [
    "ConsoleTTSService",
    "Context",
    "PiperTTSService",
    "SnapLlamaLLM",
    "SnapWhisperSTT",
    "build_tts",
]
