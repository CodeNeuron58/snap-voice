# Snap

**Snap. On-device AI, in a snap.** A fully-offline, NPU-first voice assistant — conversation plus a
small local tool layer (calculate, convert, search your notes). Built as an entry for the
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

```
mic ──> pipecat LocalAudioTransport + Silero VAD (interruptions handled by the framework)
    ──> SnapWhisperSTT      faster-whisper small int8 (CPU spike) → QNN w8a16 on NPU (AI Hub compile)
    ──> SnapLlamaLLM        Qwen3-1.7B GGUF via llama.cpp (CPU spike) → GenieX/QAIRT on NPU
    │         └─ tool JSON? → local router: calculate / convert / search_notes
    │           (result spoken from a template — no second LLM pass, JSON never spoken raw)
    ──> piper TTS (CPU, chunked) ──> speaker
         └─ TranscriptionMark / FirstAudioMeasure probes record TTFA into the benchmark table
```

pipecat owns the loop (transport, VAD, turn-taking, interruptions); Snap owns the two NPU-bound
stages as custom pipecat services plus per-stage latency probes. `snap chat --legacy` runs the
same models through a dependency-free custom loop — kept as the Windows-ARM64 fallback in case
pipecat's native deps can't go native-ARM64 on the Snapdragon machine (day-1 wheel check).

### Version notes (pipecat moves fast — read before first import)

Coded against recent pipecat docs: `pipecat.transports.local.audio.LocalAudioTransport`,
`pipecat.audio.vad.silero.SileroVADAnalyzer`, `LLMService._process_context`, and the
`OpenAILLMContext`/`LLMContext` aggregator pair (both supported via a shim in
`snap/pc/services.py`). If an import fails on your installed version, the fix is almost
always the module path — check `pipecat.services.*` in your venv and adjust.

## Repo layout

Modern `src/` layout — the installable package lives in `src/snap/`, runtime data stays at the root.

- `src/snap/pc/` — **the product path**: pipecat services (STT/LLM/TTS), mic pipeline, text spike, TTFA probes
- `src/snap/pipeline.py` — `--legacy` custom loop (no pipecat; ARM64 fallback), same interfaces
- `src/snap/tools.py` — the entire agentic surface (deliberately tiny, deterministic, local)
- `src/snap/stages/` — runtimes used by the legacy loop (CPU now, NPU after the gate)
- `src/snap/assets/` — package data shipped with the code (bundled Silero VAD ONNX)
- `src/snap/timing.py` — per-stage timers; `snap bench` emits the submission benchmark table
- `data/` — runtime-only (gitignored models, sample notes) — never published

## Challenge docs

Strategy, build plan, submission checklist: kept separately in the ZCode workspace under
`snapdragon-challenge/` (`C:\Users\bipra\.zcode\workspace\default\snapdragon-challenge`).

## Lineage (adapted from the author's project Yumi, MIT)

- `snap/pc/sentence_stream.py` — SentenceSegmenter: token→sentence streaming with `<think>`-block
  stripping (Qwen3 thinks out loud otherwise) + clause-boundary flush; Hindi Danda added.
- `snap/turn.py` — Smart Turn v3 (pipecat-ai ONNX, ~9 MB): prosodic end-of-turn for `--legacy --mic`.
- `snap/vad.py` + `snap/assets/models/silero_vad.onnx` — torch-free Silero v5 VAD; keeps the legacy
  fallback path native-ARM64-capable (onnxruntime wheels exist where torch's may not).

## License

MIT.
