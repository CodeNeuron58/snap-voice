"""Snap on pipecat — two entry points.

run_text(): typed Day-1 spike. No pipeline, no audio: the LLM service streams
            sentences straight to the console. Validates models, tools, latency.
run_mic():  the real product loop on a full pipecat Pipeline — LocalAudioTransport
            (mic + speaker), Silero VAD, STT, LLM, TTS, interruptions on.
"""

import asyncio

from snap import config, tools


def run_text(quality: bool = False) -> None:  # noqa: ARG001 — STT not exercised in text mode
    """Typed spike: same LLM service the pipeline uses, minus the voice plumbing."""
    asyncio.run(_text_loop())


async def _text_loop() -> None:
    from snap.pc.services import SnapLlamaLLM
    from snap.pc.warmup import warm_up

    llm = SnapLlamaLLM()
    await warm_up(llm=llm)

    def emit(sentence: str) -> None:
        print(f"\n[Snap] {sentence}", end="", flush=True)

    print("Snap (text mode). 'exit' to quit.")
    while True:
        try:
            text = await asyncio.to_thread(input, "> ")
        except (EOFError, KeyboardInterrupt):
            return
        if not text or text.strip().lower() in {"exit", "quit"}:
            return
        await llm.respond(text.strip(), emit=emit)
        print()


def run_mic(quality: bool = False) -> None:
    """The product: full pipecat voice pipeline with interruptions and TTFA probes."""
    asyncio.run(_mic_pipeline(quality))


async def _mic_pipeline(quality: bool) -> None:
    from pipecat.audio.vad.silero import SileroVADAnalyzer
    from pipecat.audio.vad.vad_analyzer import VADParams
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.transports.local.audio import (
        LocalAudioTransport,
        LocalAudioTransportParams,
    )

    from snap.pc.latency import FirstAudioMeasure, TranscriptionMark
    from snap.pc.services import Context, SnapLlamaLLM, SnapWhisperSTT, build_tts
    from snap.pc.warmup import warm_up

    transport = LocalAudioTransport(
        LocalAudioTransportParams(
            audio_in_enabled=True,
            audio_out_enabled=True,
            vad_analyzer=SileroVADAnalyzer(params=VADParams(stop_secs=config.VAD_STOP_SECS)),
        )
    )
    stt = SnapWhisperSTT(model_size=config.WHISPER_MODEL_QUALITY if quality else None)
    llm = SnapLlamaLLM()
    tts = build_tts()
    await warm_up(stt=stt, llm=llm)

    context = Context([{"role": "system", "content": tools.SYSTEM_PROMPT}])
    pair = _context_aggregators(context)

    pipeline = Pipeline(
        [
            transport.input(),        # mic in, VAD-gated
            stt,                      # Whisper (ours)
            TranscriptionMark(),      # TTFA stage 0 + transcript echo
            pair.user(),              # context aggregation
            llm,                      # Qwen3 + tool router (ours)
            tts,                      # Piper / console (CPU by design)
            FirstAudioMeasure(),      # TTFA recorded on first audio chunk
            transport.output(),       # speaker out
            pair.assistant(),
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))
    await PipelineRunner().run(task)


def _context_aggregators(context):
    """One pair of user/assistant aggregators across pipecat versions."""
    try:
        from pipecat.processors.aggregators.llm_response_universal import (
            LLMContextAggregatorPair,
        )

        return LLMContextAggregatorPair(context)  # pipecat >= 1.x
    except ImportError:
        return context.aggregator()  # legacy API: OpenAILLMContext.aggregator()
