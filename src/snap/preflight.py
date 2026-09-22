"""Boot-time preflight — fail fast with the fix, never with a traceback.

Every fresh clone starts without models (``data/`` is gitignored) and demo
machines sometimes have no microphone. These checks run before any heavy
dependency loads, so a missing GGUF or absent mic exits with instructions
instead of a stack trace from deep inside llama.cpp / PortAudio.
"""

import sys

from snap import config


def run(mic: bool) -> None:
    """Check everything the requested mode needs; exits(1) with guidance on failure."""
    _check_llm()
    if mic:
        _check_mic()


def _check_llm() -> None:
    gguf = config.LLM_GGUF
    if gguf.exists():
        return
    sys.exit(
        "\nSnap can't start: LLM model file not found.\n"
        f"  expected at: {gguf}\n"
        "\nOne-time fix:\n"
        "  1. Download qwen3-4b-instruct-2507-q4_k_m.gguf from Hugging Face\n"
        "     (Qwen/Qwen3-4B-Instruct-2507-GGUF, the Q4_K_M quant)\n"
        f"  2. Place it in: {gguf.parent}\n"
        "  Smaller/faster alternative: uncomment the 1.7B fallback line (LLM_GGUF)\n"
        "  in src/snap/config.py. Full steps: README 'Quickstart', step 2."
    )


def _check_mic() -> None:
    try:
        import sounddevice as sd

        devices = [
            d for d in sd.query_devices() if getattr(d, "max_input_channels", 0) > 0
        ]
    except Exception as exc:  # noqa: BLE001 — any PortAudio failure is the same UX
        sys.exit(
            "\nSnap can't start: audio system unavailable.\n"
            f"  {exc}\n"
            "Mic mode needs a working audio stack — or use typed mode:\n"
            "  snap chat --text"
        )
    if not devices:
        sys.exit(
            "\nSnap can't start: no microphone found.\n"
            "Plug one in (check Settings > Sound), or use typed mode:\n"
            "  snap chat --text"
        )
