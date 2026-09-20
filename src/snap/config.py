"""Central configuration. Edit these instead of hunting through the code."""

import os
from pathlib import Path

# huggingface.co is unreachable on some networks (connection reset at the CDN).
# Route all HF traffic (Whisper auto-download, SmartTurn weights) through the
# community mirror unless the user has set their own endpoint.
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# src-layout: config.py sits at <repo>/src/snap/config.py → repo root is 3 levels up.
PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PACKAGE_ROOT / "data"
NOTES_DIR = DATA_DIR / "notes"
MODELS_DIR = DATA_DIR / "models"

# STT (locked 2026-09-14, AI Hub-measured on Snapdragon X2 Elite CRD):
#   small          -> encoder 60.5ms + 6.58ms/token (~190ms STT pass)  DEFAULT, TTFA ~0.9s
#   large-v3-turbo -> encoder 681ms + 2.67ms/token (~735ms)  QUALITY flag, TTFA ~1.4s
# Both multilingual (EN + HI). DeepSpeech2 rejected: 2015 CTC model, worse WER, English-only.
WHISPER_MODEL = "small"
WHISPER_MODEL_QUALITY = "large-v3-turbo"  # used by the --quality CLI flag
WHISPER_LANGUAGE = None          # None = auto-detect EN/HI

# LLM (upgraded 2026-09-21 to Qwen3-4B-Instruct-2507 — AI Hub X2 Elite CRD: prefill 2,831
# tok/s, decode 42.6 tok/s, on-device accuracy 86.4% vs 1.7B's 73%). Instruct-only = no
# think-mode (THINK_STRIP stays as a safety net). NPU build: `geniex infer
# ai-hub-models/Qwen3-4B-Instruct-2507`. Fallback: the 1.7B GGUF below (one-line swap).
LLM_GGUF = MODELS_DIR / "qwen3-4b-instruct" / "qwen3-4b-instruct-2507-q4_k_m.gguf"
# LLM_GGUF = MODELS_DIR / "qwen3" / "qwen3-1.7b-q4_k_m.gguf"  # fallback, smaller/faster
LLM_N_CTX = 4096
LLM_N_THREADS = 6
LLM_MAX_TOKENS = 256

# TTS (locked 2026-09-14): PiperTTS-EN from AI Hub, stays CPU. Hindi voice = roadmap.
# Point PIPER_VOICE at a local .onnx voice; empty = console fallback for the spike.
PIPER_VOICE = ""

# Turn taking (spike: energy-based; swap to a smart-turn model later)
SILENCE_END_OF_TURN_MS = 700
MAX_UTTERANCE_MS = 12_000
SAMPLE_RATE = 16_000

# Mic mode (Silero VAD + SmartTurn, adapted from Yumi)
VAD_SPEECH_THRESHOLD = 0.5         # Silero speech probability gate
SMART_TURN_ENABLED = True          # prosodic end-of-turn; auto-disables if models/deps missing
SMART_TURN_CHECK_SILENCE_MS = 250  # Deepgram-style aggressive: first check after 250ms of silence
SMART_TURN_RECHECK_MS = 150        # re-check cadence while silence continues

# pipecat pipeline: Silero VAD end-of-speech (Deepgram agents use ~300ms endpointing)
VAD_STOP_SECS = 0.45

# Latency hygiene
WARM_UP_ON_BOOT = True             # dummy Whisper pass + 1-token LLM completion at startup
PRINT_TURN_REPORT = True           # per-turn Deepgram-style latency attribution line
