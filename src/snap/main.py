"""Snap CLI.

    snap chat --text            # Day-1 latency spike (pipecat LLM service, typed input)
    snap chat --mic             # full pipecat voice pipeline: VAD, STT, LLM, TTS, barge-in
    snap chat --mic --quality   # Whisper-Large-V3-Turbo STT (better accuracy, TTFA ~1.4s)
    snap chat --legacy --mic    # custom loop (no pipecat) — the ARM64 fallback path
    snap bench                  # fixed prompt set, per-stage medians -> benchmark table
    snap bench --quality        # same, with Turbo STT (write-up: compare both)
"""

import argparse
import json
import sys
from pathlib import Path

from snap import config
from snap.timing import TIMINGS

BENCH_PROMPTS = [
    "Explain photosynthesis in one sentence.",
    "What is 15% of 2400?",                                    # tool: calculate
    "Convert 5 km to miles.",                                  # tool: convert
    "From my notes, what did I write about the water cycle?",  # tool: search_notes
    "नमस्ते — बस इतना बताओ कि तुम ऑफ़लाइन काम करते हो या नहीं।",
    "Tell me a two-sentence bedtime story about a robot.",
    "What is the square root of 1764?",                        # tool: calculate
    "Summarize your own capabilities in three lines.",
    "12 kg in pounds?",                                        # tool: convert
    "Why is on-device AI better for privacy? Two lines.",
]


def build_assistant(with_audio: bool, quality_stt: bool = False):
    """--legacy path: the custom loop (no pipecat). Kept as the ARM64 fallback."""
    from snap.pipeline import Snap
    from snap.stages.llamacpp_llm import LlamaCppLLM
    from snap.stages.tts import ConsoleTTS, PiperTTS

    stt = None
    if with_audio:
        from snap.stages.whisper_stt import FasterWhisperSTT

        model_size = config.WHISPER_MODEL_QUALITY if quality_stt else config.WHISPER_MODEL
        stt = FasterWhisperSTT(model_size=model_size)
    tts = (
        PiperTTS(str(config.PIPER_VOICE))
        if config.PIPER_VOICE and Path(config.PIPER_VOICE).exists()
        else ConsoleTTS()
    )
    return Snap(stt=stt, llm=LlamaCppLLM(), tts=tts)


def cmd_chat(args: argparse.Namespace) -> None:
    if args.legacy:
        assistant = build_assistant(with_audio=args.mic, quality_stt=args.quality)
        if args.mic:
            assistant.run_mic()
        else:
            assistant.run_text()
        return
    from snap.pc import app

    if args.mic:
        app.run_mic(quality=args.quality)
    else:
        app.run_text(quality=args.quality)


def cmd_bench(args: argparse.Namespace) -> None:
    assistant = build_assistant(quality_stt=args.quality)
    print("Warming up (model load, allocations)...")
    assistant.turn("Hello.")
    TIMINGS.clear()

    for run in range(args.runs):
        for prompt in BENCH_PROMPTS:
            print(f"[bench] run {run + 1}/{args.runs}: {prompt[:48]}")
            assistant.turn(prompt)

    print("\n## Benchmark (medians)\n")
    print(TIMINGS.markdown())
    out = Path(f"bench_results{'_quality' if args.quality else ''}.json")
    out.write_text(json.dumps(TIMINGS.summary(), indent=2), encoding="utf-8")
    print(f"\nSaved -> {out.resolve()}")
    print(
        "Next: paste the table into docs/deck-and-video-script.md; "
        "fill the NPU column after AI Hub profiling (scripts/profile_on_aihub.py)."
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="snap")
    sub = parser.add_subparsers(dest="cmd", required=True)

    chat = sub.add_parser("chat", help="talk to Snap")
    chat.add_argument("--text", action="store_true", help="typed input, no audio (Day-1 spike)")
    chat.add_argument("--mic", action="store_true", help="full pipecat voice pipeline")
    chat.add_argument("--quality", action="store_true", help="Whisper-Large-V3-Turbo (better STT, TTFA ~1.4s)")
    chat.add_argument("--legacy", action="store_true", help="custom loop, no pipecat (ARM64 fallback)")

    bench = sub.add_parser("bench", help="run the fixed benchmark prompt set")
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--quality", action="store_true", help="bench with Whisper-Large-V3-Turbo STT")

    args = parser.parse_args()
    try:
        if args.cmd == "chat":
            cmd_chat(args)
        elif args.cmd == "bench":
            cmd_bench(args)
    except KeyboardInterrupt:
        print("\nbye.")
        sys.exit(130)


if __name__ == "__main__":
    main()
