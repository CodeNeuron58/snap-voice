"""Per-stage latency probes that ride inside a pipecat pipeline.

Place TranscriptionMark AFTER the STT service and FirstAudioMeasure between TTS and
transport.output. Together they record `utterance_to_first_audio` (TTFA — the only
latency that matters perceptually) straight into snap.timing.TIMINGS.
"""

import time

from pipecat.frames.frames import (
    AudioRawFrame,
    Frame,
    InterruptionFrame,
    TranscriptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor

from snap.timing import TIMINGS

_state: dict = {"t_utterance": None}


class TranscriptionMark(FrameProcessor):
    """Marks the moment a final transcript flows past (stage 0 of the TTFA ruler)."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            print(f"\nYou: {frame.text}")
            _state["t_utterance"] = time.perf_counter()
            _state["first_audio_done"] = False
        elif isinstance(frame, InterruptionFrame):
            _state["t_utterance"] = None
        await self.push_frame(frame, direction)


class FirstAudioMeasure(FrameProcessor):
    """Records TTFA on the first audio chunk after a marked utterance."""

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        if (
            isinstance(frame, AudioRawFrame)
            and _state.get("t_utterance")
            and not _state.get("first_audio_done")
        ):
            _state["first_audio_done"] = True
            TIMINGS.record(
                "utterance_to_first_audio",
                (time.perf_counter() - _state["t_utterance"]) * 1000,
            )
        await self.push_frame(frame, direction)
