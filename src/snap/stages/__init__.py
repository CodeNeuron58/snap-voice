"""Pipeline stage implementations. Each file behind a tiny duck-typed interface so
CPU and NPU runtimes swap without touching the pipeline:

    STT:       transcribe(pcm_16k: np.ndarray) -> str
    LLM:       stream_reply(messages, on_token) -> str
    TTS:       speak(text, stop) -> None  (blocking; must check `stop` frequently)
"""

from snap.stages.whisper_stt import FasterWhisperSTT
from snap.stages.llamacpp_llm import LlamaCppLLM
from snap.stages.tts import ConsoleTTS, PiperTTS

__all__ = ["FasterWhisperSTT", "LlamaCppLLM", "ConsoleTTS", "PiperTTS"]
