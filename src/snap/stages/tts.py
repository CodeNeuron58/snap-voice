"""TTS: Piper if installed, console fallback otherwise. Stays CPU by design.

`speak()` must check `stop` (a threading.Event) between chunks — barge-in
depends on TTS stopping within ~50ms of the event being set.
"""

import threading

from snap.timing import TIMINGS


class ConsoleTTS:
    """Zero-dependency fallback: prints instead of speaking. Fine for the Day-1
    latency spike; swap in Piper (or any local streaming TTS) for audio."""

    def __init__(self) -> None:
        print("[tts] console fallback — install piper-tts for audio out")

    def speak(self, text: str, stop: threading.Event) -> None:
        with TIMINGS.stage("tts_console"):
            print(f"\n[Snap] {text}\n> ", end="", flush=True)


class PiperTTS:
    def __init__(self, voice_path: str) -> None:
        from piper import PiperVoice  # pip install piper-tts

        self._voice = PiperVoice.load(voice_path)
        self._sr = getattr(getattr(self._voice, "config", None), "sample_rate", 22050)
        try:
            import sounddevice as sd

            self._sd = sd
        except ImportError as exc:
            raise RuntimeError("piper needs sounddevice for playback") from exc

    def speak(self, text: str, stop: threading.Event) -> None:
        with TIMINGS.stage("tts"):
            # Stream in synth chunks so first audio starts early and barge-in
            # only loses the current chunk, not the whole reply.
            with self._sd.OutputStream(
                samplerate=self._sr, channels=1, dtype="int16"
            ) as stream:
                for chunk in self._voice.synthesize(text):
                    if stop.is_set():
                        return
                    audio = getattr(chunk, "audio_int16_bytes", None) or getattr(
                        chunk, "audio", b""
                    )
                    if audio:
                        stream.write(audio)
