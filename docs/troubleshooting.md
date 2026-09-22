# Troubleshooting & dev notes

Detail that didn't earn a place in the README lives here.

## Setup problems

- **"LLM model file not found"** — `data/models/` is gitignored, so a fresh clone has no
  models. Download `qwen3-4b-instruct-2507-q4_k_m.gguf`
  ([Qwen/Qwen3-4B-Instruct-2507-GGUF](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507-GGUF),
  the Q4_K_M quant) into `data/models/qwen3-4b-instruct/`. The boot preflight
  (`src/snap/preflight.py`) prints this exact instruction and exits cleanly.
- **"Whisper … failed to load"** — Whisper auto-downloads from Hugging Face on first run.
  All HF traffic routes via the `hf-mirror.com` mirror by default (`HF_ENDPOINT` override in
  `src/snap/config.py`); check connectivity or pre-download the model.
- **"no microphone found" / audio errors** — check Settings → Sound for a working input
  device, or use typed mode: `snap chat --text`.
- **llama-cpp-python install is slow or fails** — PyPI has no Windows wheel, so
  `pyproject.toml` pins a prebuilt `win_amd64` wheel; on other platforms uv compiles it from
  source (~30 min, needs CMake + a C++ toolchain).
- **pipecat import errors** — pipecat moves fast; the fix is almost always the module path.
  See the version notes below.

## pipecat version notes

- Coded against pipecat 1.x: `pipecat.transports.local.audio.LocalAudioTransport`,
  `pipecat.audio.vad.silero.SileroVADAnalyzer`, `LLMService._process_context`, and the
  `LLMContext` / `OpenAILLMContext` aggregator pair (both supported via a shim in
  `snap/pc/services.py`).
- If an import fails on your installed version, check `pipecat.services.*` in your venv and
  adjust the module path.
- Silero VAD and SmartTurn v3 ship with pipecat 1.11's extras — nothing to fetch by hand.

## Benchmarks

- `snap bench` measures the default user path: the deterministic pre-router answers
  math/convert instantly, the LLM handles conversation and notes.
- `snap bench --no-preroute` is the diagnostic tier: every prompt is forced through the LLM's
  `<tool_call>` path, measuring real model tool adherence.
- Tables, methodology and honest labels: [benchmarks.md](benchmarks.md). Raw profile evidence
  (AI Hub jobs, screenshots) belongs in [evidence/](evidence/).

## Development

```bash
uv run pytest          # 65 unit tests — pure logic, no models or audio deps needed
uv run ruff check .    # lint gate (CI enforces)
```
