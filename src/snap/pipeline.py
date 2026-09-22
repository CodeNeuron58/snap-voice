"""The Snap legacy pipeline (custom loop, no pipecat) — the ARM64 fallback path.

Mirrors snap/pc's behavior without framework dependencies beyond numpy:
  1. Sentence-level streaming with thinking-block stripping (SentenceSegmenter).
  2. JSON guard — tool-call JSON is never spoken raw; the router executes it
     and the templated result is spoken with no second LLM pass.
  3. Mic mode: Silero VAD (ONNX, torch-free) capture + SmartTurn prosodic
     end-of-turn; transcription and turns run off the audio thread.
Barge-in: `stop_tts` is checked between TTS chunks (use a headset — speaker
echo can self-trigger; there is no AEC in this loop).
"""

import logging
import queue
import threading

import numpy as np

from snap import config, tools
from snap.pc.sentence_stream import THINK_BLOCK, SentenceSegmenter
from snap.timing import TIMINGS

log = logging.getLogger(__name__)


class Snap:
    def __init__(self, stt, llm, tts) -> None:
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.history: list[dict] = [{"role": "system", "content": tools.SYSTEM_PROMPT}]

    def warm_up(self) -> None:
        """Boot warm-up: silent Whisper pass + 1-token completion (Deepgram keeps models
        hot; we do too, on-device). Never raises — warm-up must not block boot."""
        if not config.WARM_UP_ON_BOOT:
            return
        import time as _time

        t0 = _time.perf_counter()
        for service in (self.stt, self.llm):
            if service is None or not hasattr(service, "warm_up"):
                continue
            try:
                service.warm_up()
            except Exception as exc:  # noqa: BLE001
                log.warning("warm-up skipped (%s)", exc)
        log.info("warm-up: models hot in %.2fs", _time.perf_counter() - t0)

    # --- one full conversational turn ---------------------------------------

    def turn(self, user_text: str) -> str:
        t0 = TIMINGS.now_ms()
        # Deterministic answers never touch the LLM: instant, hallucination-free.
        pre = tools.pre_route(user_text)
        if pre is not None:
            TIMINGS.record("tool_answer_no_llm", TIMINGS.now_ms() - t0)
            self.tts.speak(pre, threading.Event())
            self.history += [
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": pre},
            ]
            print(f"[Snap] {pre}\n")
            return pre

        messages = self.history + [{"role": "user", "content": user_text}]

        stop_tts = threading.Event()
        sentences: queue.Queue[str] = queue.Queue()
        tts_done = threading.Event()

        def tts_worker() -> None:
            while True:
                try:
                    sentence = sentences.get(timeout=0.1)
                except queue.Empty:
                    if tts_done.is_set():
                        return
                    continue
                self.tts.speak(sentence, stop_tts)
                if stop_tts.is_set():  # barge-in: drain and bail
                    while not sentences.empty():
                        sentences.get_nowait()
                    return

        speaker = threading.Thread(target=tts_worker, daemon=True)
        speaker.start()

        collected: list[str] = []
        segmenter = SentenceSegmenter()

        def speakable(sentence: str) -> bool:
            return not sentence.lstrip().startswith("{")  # raw JSON never reaches speech

        def on_token(token: str) -> None:
            collected.append(token)
            for sentence in segmenter.feed(token):
                if speakable(sentence):
                    sentences.put(sentence)

        # First-token timing is owned by stream_reply's stage (prefill-inclusive);
        # recording it here too double-counts under the same key.
        try:
            raw = self.llm.stream_reply(messages, on_token=on_token).strip()
            TIMINGS.record("llm_total", TIMINGS.now_ms() - t0)
            tail = segmenter.flush()
            if tail and speakable(tail):
                sentences.put(tail)
        finally:
            # Must fire even if the LLM raised mid-stream — otherwise tts_worker
            # spins on its 0.1s poll forever and the thread leaks.
            tts_done.set()
        speaker.join(timeout=30)

        # Think text must not reach routing or history (unterminated <think> from
        # token-budget exhaustion would otherwise become the "reply").
        raw = THINK_BLOCK.sub("", raw, count=1).strip()
        reply = tools.route(raw) if raw else "…"
        if reply and reply != raw:  # tool path: speak the templated result now
            self.tts.speak(reply, stop_tts)

        self.history += [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": reply},
        ]
        # KV-cache aware trim: append-only by default, rare chunked invalidation.
        if len(self.history) > 13:
            del self.history[:4]
        print(f"[Snap] {reply}\n")
        return reply

    # --- text mode (Day-1 latency spike) -------------------------------------

    def run_text(self) -> None:
        self.warm_up()
        print("Snap (text mode). 'exit' to quit.")
        while True:
            try:
                text = input("> ").strip()
            except EOFError:
                return
            if not text or text.lower() in {"exit", "quit"}:
                return
            with TIMINGS.stage("stt_input"):  # placeholder keeps table rows aligned
                pass
            try:
                self.turn(text)
            except Exception as exc:  # noqa: BLE001 — one bad turn must not kill the session
                log.error("turn failed: %s", exc)
                print("[Snap] Sorry, something went wrong on my side.\n")

    # --- mic mode: Silero VAD + SmartTurn end-of-turn (adapted from Yumi) -----

    def run_mic(self) -> None:  # pragma: no cover — hardware loop, exercised manually
        """Capture with a real VAD; SmartTurn (prosody) decides end-of-turn, not just
        silence. Transcription and turns run off the audio thread so no frames drop."""
        self.warm_up()
        import sounddevice as sd

        from snap.turn import get_smart_turn
        from snap.vad import SileroVAD

        try:
            vad = SileroVAD()
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        turn_detector = get_smart_turn()
        if turn_detector is None:
            log.warning("smart-turn unavailable — fixed-silence end-of-turn fallback")

        sr = config_sample_rate()
        block = 512  # Silero v5 wants 512-sample frames @16k (32 ms)
        block_ms = 1000.0 * block / sr
        speech_q: queue.Queue = queue.Queue()
        st = {"capturing": False, "buf": [], "sil_ms": 0.0, "last_check": -1e9}

        def finalize() -> None:
            audio = np.concatenate(st["buf"])
            st.update(capturing=False, buf=[], sil_ms=0.0, last_check=-1e9)
            vad.reset_states()
            speech_q.put(audio)

        def callback(indata, _frames, _time, status) -> None:  # noqa: ANN001
            if status:
                return
            pcm = np.frombuffer(indata, dtype=np.int16)
            prob = vad(pcm.astype(np.float32) / 32768.0, sr)
            if prob >= config.VAD_SPEECH_THRESHOLD:
                st["capturing"] = True
                st["sil_ms"] = 0.0
                st["buf"].append(pcm.copy())
                return
            if not st["capturing"]:
                return
            st["sil_ms"] += block_ms
            st["buf"].append(pcm.copy())
            samples_cap = int(sr * config.MAX_UTTERANCE_MS / 1000)
            if sum(len(x) for x in st["buf"]) >= samples_cap:
                finalize()  # hard cap — a run-on ramble still becomes a turn
                return
            if turn_detector is None:
                if st["sil_ms"] >= config.SILENCE_END_OF_TURN_MS:
                    finalize()
                return
            # SmartTurn: first check after CHECK_SILENCE_MS, then every RECHECK_MS.
            if (
                st["sil_ms"] >= config.SMART_TURN_CHECK_SILENCE_MS
                and st["sil_ms"] - st["last_check"] >= config.SMART_TURN_RECHECK_MS
            ):
                st["last_check"] = st["sil_ms"]
                audio = np.concatenate(st["buf"]).astype(np.float32) / 32768.0
                try:
                    complete, _prob = turn_detector.is_complete(audio)
                except Exception:  # noqa: BLE001 — degrade to fixed-silence timing
                    complete = st["sil_ms"] >= config.SILENCE_END_OF_TURN_MS
                if complete:
                    finalize()

        def worker() -> None:
            while True:
                audio = speech_q.get()
                if audio is None:
                    return
                try:
                    text = self.stt.transcribe(audio)
                    if text:
                        print(f"\nYou: {text}")
                        self.turn(text)
                except Exception as exc:  # noqa: BLE001 — one bad turn must not kill the mic loop
                    log.error("turn failed: %s", exc)
                    print("[Snap] Sorry, something went wrong on my side.\n")
                print("> ", end="", flush=True)

        threading.Thread(target=worker, daemon=True).start()
        with sd.InputStream(
            samplerate=sr, channels=1, dtype="int16", blocksize=block, callback=callback
        ):
            print("Snap listening (Silero VAD + SmartTurn). Ctrl+C to quit.\n> ", end="")
            try:
                threading.Event().wait()
            except KeyboardInterrupt:
                speech_q.put(None)
                return


def config_sample_rate() -> int:
    return config.SAMPLE_RATE
