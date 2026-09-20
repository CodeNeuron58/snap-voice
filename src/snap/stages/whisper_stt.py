"""STT via faster-whisper (CTranslate2) — the CPU spike baseline.

NPU path (after AI Hub compile): export Whisper to ONNX -> compile w8a16 to a QNN
context binary -> wrap in a class with the same transcribe() signature using
ONNX Runtime + QNN Execution Provider. The pipeline must not change.
"""

import numpy as np
from snap import config
from snap.timing import TIMINGS


class FasterWhisperSTT:
    def __init__(self, model_size: str | None = None) -> None:
        from faster_whisper import WhisperModel  # heavy import: keep lazy

        self._model = WhisperModel(
            model_size or config.WHISPER_MODEL, device="cpu", compute_type="int8"
        )
        self._language = config.WHISPER_LANGUAGE

    def transcribe(self, pcm_16k: np.ndarray) -> str:
        pcm = pcm_16k.astype(np.float32) / 32768.0
        with TIMINGS.stage("stt"):
            segments, _ = self._model.transcribe(
                pcm, language=self._language, vad_filter=False, beam_size=1
            )
            return " ".join(s.text.strip() for s in segments).strip()

    def warm_up(self) -> None:
        """One silent pass so mel filterbanks and decoder state are hot before turn 1."""
        silence = np.zeros(config.SAMPLE_RATE // 2, dtype=np.float32)  # 0.5 s
        for _seg in self._model.transcribe(silence, language="en", beam_size=1)[0]:
            break
