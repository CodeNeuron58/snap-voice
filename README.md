# Snap

[![CI](https://github.com/CodeNeuron58/snap-voice/actions/workflows/ci.yml/badge.svg)](https://github.com/CodeNeuron58/snap-voice/actions/workflows/ci.yml)

**Snap. On-device AI, in a snap.** A fully-offline, NPU-first voice assistant built for
**Snapdragon-powered HP PCs** (Windows on Snapdragon ARM64, HP OmniBook class): conversation plus
a small local tool layer (calculate, convert, remember & search notes, date & time), with every
heavy stage quantized and bound for the Hexagon NPU. Built as an entry for the
Snapdragon® AI Lab Build & Present Challenge 2026.

> Independent open-source project. Not affiliated with Snap Inc. or Qualcomm.

## Quickstart (Day-1 CPU spike)

```bash
# 1. environment (uv) — venv, then sync: installs ALL runtime deps + this package editable
uv venv
uv sync

# 2. models — put a GGUF in data/models/qwen3/ (e.g. qwen3-1.7b-q4_k_m.gguf from Hugging Face);
#    Whisper downloads automatically on first run.
#    Silero VAD model is not committed: fetch it once ->
#      curl -L -o src/snap/assets/models/silero_vad.onnx ^
#        https://github.com/snakers4/silero-vad/raw/master/src/silero-vad/data/silero_vad.onnx
#    (or copy it from any machine that has it). Optional: set PIPER_VOICE in src/snap/config.py
#    to a piper .onnx voice for real audio out (otherwise console fallback).
# 3. run
uv run snap chat --text            # typed spike: LLM + tools + sentence streaming (no audio deps)
uv run snap chat --mic             # FULL pipecat voice pipeline: Silero VAD, STT, LLM, TTS, barge-in
uv run snap chat --mic --quality   # Whisper-Large-V3-Turbo STT (better accuracy, TTFA ~1.4s)
uv run snap bench                  # fixed 10-prompt set -> per-stage medians -> bench_results.json
uv run snap chat --legacy --mic    # custom loop without pipecat (the ARM64 fallback path)

# later, for AI Hub profiling (challenge evidence): uv sync --extra aihub
```

## Architecture

```mermaid
flowchart TB
    U["User: speaks or types"] --> VAD["Silero VAD + SmartTurn end-of-turn<br/>(barge-in kills generation instantly)"]
    VAD --> STT["STT - faster-whisper<br/>small (default) or large-v3-turbo (--quality)"]
    STT --> PRE{"Arithmetic or unit<br/>conversion?"}
    PRE -- "yes" --> FAST["Deterministic answer<br/>0.1 ms, zero LLM tokens"]
    PRE -- "no" --> LLM["LLM - Qwen3-4B-Instruct GGUF via llama.cpp<br/>streams tokens; tool schemas ride in the system prompt"]
    LLM --> SEG{"SentenceSegmenter:<br/>think-blocks and tool tags stripped"}
    SEG -- "sentence closes" --> TTS["TTS - Piper, streaming per sentence<br/>(console fallback in text mode)"]
    SEG -- "tool call in stream" --> EXEC["Local tools, schema-driven:<br/>calculate / convert / search_notes / add_note / get_datetime"]
    EXEC -- "deterministic tool:<br/>template speaks, no 2nd pass" --> FAST
    EXEC -- "observation tool:<br/>result fed back (max 2 hops,<br/>errors self-correct)" --> LLM
    TTS --> OUT["Speaker"]
```

### Two interchangeable runtimes, one core

```mermaid
flowchart LR
    subgraph R["Both runtimes wrap the identical turn pipeline above"]
        direction TB
        A["pipecat runtime (default)<br/>snap chat --mic / --text<br/>transport, VAD, turn-taking, barge-in"]
        B["legacy runtime (ARM64 fallback)<br/>snap chat --legacy --mic / --text<br/>dependency-free custom loop"]
    end
    A --> C["Shared core:<br/>tool registry - SentenceSegmenter -<br/>Whisper - Qwen3 GGUF - timing/bench"]
    B --> C
```

pipecat owns the loop (transport, VAD, turn-taking, interruptions); Snap owns the two NPU-bound
stages as custom pipecat services plus per-stage latency probes. `snap chat --legacy` runs the
same models through a dependency-free custom loop — kept as the Windows-ARM64 fallback in case
pipecat's native deps can't go native-ARM64 on the Snapdragon machine (day-1 wheel check).

**Target hardware:** the Hexagon NPU in Snapdragon X / X2 Elite (the silicon of Snapdragon-powered
HP OmniBooks, Windows on Snapdragon ARM64). The current CPU spike runs the identical interfaces on
x64 today; the NPU runtimes slot in behind the same classes with no pipeline changes
(`scripts/profile_on_aihub.py` produces the hosted-device evidence).

### Version notes (pipecat moves fast — read before first import)

Coded against recent pipecat docs: `pipecat.transports.local.audio.LocalAudioTransport`,
`pipecat.audio.vad.silero.SileroVADAnalyzer`, `LLMService._process_context`, and the
`OpenAILLMContext`/`LLMContext` aggregator pair (both supported via a shim in
`snap/pc/services.py`). If an import fails on your installed version, the fix is almost
always the module path — check `pipecat.services.*` in your venv and adjust.

## Troubleshooting

- **"LLM model file not found"** — `data/models/` is gitignored, so a fresh clone has no
  models. Download `qwen3-4b-instruct-2507-q4_k_m.gguf` (Qwen/Qwen3-4B-Instruct-2507-GGUF,
  the Q4_K_M quant) into `data/models/qwen3-4b-instruct/`, or uncomment the 1.7B fallback
  line (`LLM_GGUF`) in `src/snap/config.py`. Details: Quickstart step 2.
- **"Whisper ... failed to load"** — Whisper auto-downloads from Hugging Face on first run.
  All HF traffic is routed via the `hf-mirror.com` mirror by default (`HF_ENDPOINT`
  override in `src/snap/config.py`); check connectivity, or pre-download the model.
- **"no microphone found" / audio errors** — check Settings → Sound for a working input
  device, or use typed mode: `snap chat --text`.
- **`silero_vad.onnx` not found** — the VAD model isn't committed; fetch it once (curl in
  Quickstart step 2).
- **pipecat import errors** — pipecat moves fast; see "Version notes" above. The fix is
  almost always the module path.
- **llama-cpp-python install is slow or fails** — PyPI has no Windows wheel, so
  `pyproject.toml` pins a prebuilt `win_amd64` wheel; on other platforms uv compiles it
  from source (~30 min, needs CMake + a C++ toolchain).
- **SmartTurn / turn-detection looks off** — the SmartTurn ONNX auto-downloads (~9 MB);
  `SNAP_SMART_TURN_URL` overrides the source. If unavailable, Snap falls back to
  fixed-silence end-of-turn and says so in the log.

## Repo layout

Modern `src/` layout — the installable package lives in `src/snap/`, runtime data stays at the root.

- `src/snap/pc/` — **the product path**: pipecat services (STT/LLM/TTS), mic pipeline, text spike, TTFA probes
- `src/snap/pipeline.py` — `--legacy` custom loop (no pipecat; ARM64 fallback), same interfaces
- `src/snap/tools.py` — the entire agentic surface (deliberately tiny, deterministic, local)
- `src/snap/stages/` — runtimes used by the legacy loop (CPU now, NPU after the gate)
- `src/snap/assets/` — package data shipped with the code (bundled Silero VAD ONNX)
- `src/snap/timing.py` — per-stage timers; `snap bench` emits the submission benchmark table
- `docs/` — `benchmarks.md` (methodology + numbers) and `evidence/` (raw AI Hub profiles)
- `scripts/` — `profile_on_aihub.py` (AI Hub hosted-device profiling harness)
- `data/` — runtime-only (gitignored models, sample notes) — never published

## Challenge docs

Strategy, build plan, submission checklist: kept separately in the ZCode workspace under
`snapdragon-challenge/` (`C:\Users\bipra\.zcode\workspace\default\snapdragon-challenge`).

## Credits

A few small utility modules (sentence segmentation, prosodic turn-end, VAD) are adapted from the
author's earlier MIT open-source project [Yumii](https://github.com/CodeNeuron58/Yumii); everything
else — the on-device model stack, pipeline, and tooling — was built new for this challenge.

## Development

```bash
uv sync               # full env incl. dev tools (ruff, pytest)
uv run pytest         # unit tests — pure logic only, no models or audio deps needed
uv run ruff check .   # lint (CI gates on this)
```

## License

MIT.
