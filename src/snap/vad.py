"""Silero VAD v5 on onnxruntime (no torch).

The model is bundled in snap/assets/models/silero_vad.onnx (~2 MB), stateful
(an RNN); reset_states() zeroes it before each capture. This is the VAD for the
`--legacy` loop and the ARM64 fallback: onnxruntime ships Windows-ARM64 wheels,
torch (pipecat's Silero extra) may not — so the fallback path stays native-ARM64
capable. The pipecat pipeline uses the framework's own Silero analyzer instead.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

log = logging.getLogger(__name__)

_MODEL_PATH = Path(__file__).parent / "assets" / "models" / "silero_vad.onnx"

# v5 hidden state shape: [2, batch, 128].
_STATE_SHAPE = (2, 1, 128)
# v5 prepends a 64-sample context (fed 576 samples); omitting it makes the model return ~0.
_CONTEXT_SIZE = 64


class SileroVAD:
    """Silero v5 VAD on ONNX; drop-in for the old torch model (reset_states() then vad(frame, sr))."""

    def __init__(self, model_path: str | Path = _MODEL_PATH) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(
                f"Silero VAD model not found at {model_path} — it ships with the repo "
                "(snap/assets/models/silero_vad.onnx); restore it or pass model_path."
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        opts.log_severity_level = 3  # quiet
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)
        log.info("silero_vad_ready backend=onnxruntime")

    def reset_states(self) -> None:
        """Clear the RNN hidden state + context (call before each capture)."""
        self._state = np.zeros(_STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT_SIZE), dtype=np.float32)

    def __call__(self, frame: np.ndarray, sr: int) -> float:
        """Return the speech probability (0..1) for one 512-sample frame at 16 kHz."""
        x = np.asarray(frame, dtype=np.float32)
        if x.ndim == 1:
            x = x[np.newaxis, :]
        # Prepend the previous frame's tail; feed 64 + 512 = 576 samples.
        x = np.concatenate([self._context, x], axis=1)
        out, self._state = self._session.run(
            None,
            {
                "input": x,
                "state": self._state,
                "sr": np.array(sr, dtype=np.int64),
            },
        )
        self._context = x[:, -_CONTEXT_SIZE:]
        return float(out[0, 0])
