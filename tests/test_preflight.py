"""Preflight tests: missing model / no mic must exit with guidance, never a traceback."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from snap import config
from snap.preflight import _check_llm, _check_mic, run


def _fake_sd(devices: list) -> SimpleNamespace:
    return SimpleNamespace(query_devices=lambda: devices)


def test_check_llm_passes_when_present(tmp_path: Path, monkeypatch) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(config, "LLM_GGUF", gguf)
    _check_llm()  # no SystemExit


def test_check_llm_missing_exits_with_guidance(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(config, "LLM_GGUF", tmp_path / "nope.gguf")
    with pytest.raises(SystemExit) as excinfo:
        _check_llm()
    message = str(excinfo.value)
    assert str(tmp_path / "nope.gguf") in message
    assert "Quickstart" in message  # points at the fix, not just the problem


def test_run_skips_mic_check_without_mic_flag(tmp_path: Path, monkeypatch) -> None:
    gguf = tmp_path / "model.gguf"
    gguf.write_bytes(b"x")
    monkeypatch.setattr(config, "LLM_GGUF", gguf)
    monkeypatch.setitem(sys.modules, "sounddevice", None)  # import must not even happen
    run(mic=False)  # no SystemExit


def test_check_mic_present_passes(monkeypatch) -> None:
    devices = [
        SimpleNamespace(max_input_channels=0),  # speaker
        SimpleNamespace(max_input_channels=1),  # mic
    ]
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devices))
    _check_mic()  # no SystemExit


def test_check_mic_absent_exits(monkeypatch) -> None:
    devices = [SimpleNamespace(max_input_channels=0)]  # output-only device
    monkeypatch.setitem(sys.modules, "sounddevice", _fake_sd(devices))
    with pytest.raises(SystemExit) as excinfo:
        _check_mic()
    assert "no microphone" in str(excinfo.value)


def test_check_mic_audio_system_failure_exits(monkeypatch) -> None:
    def boom():
        raise OSError("PortAudio not initialized")

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(query_devices=boom))
    with pytest.raises(SystemExit) as excinfo:
        _check_mic()
    assert "PortAudio not initialized" in str(excinfo.value)
