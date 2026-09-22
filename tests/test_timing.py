"""Timings tests — the benchmark backbone's math (median/p95/latency report)."""

from snap.timing import Timings


def test_record_and_summary_stats() -> None:
    t = Timings()
    for ms in (10, 20, 30, 40, 100):
        t.record("stt", ms)
    s = t.summary()["stt"]
    assert s["n"] == 5
    assert s["median_ms"] == 30.0
    assert s["min_ms"] == 10.0
    assert s["max_ms"] == 100.0
    assert s["p95_ms"] == 100.0  # n=5: index min(4, round(0.95*4)) = 4


def test_summary_p95_indexing() -> None:
    t = Timings()
    for ms in range(1, 21):  # 1..20
        t.record("llm", ms)
    s = t.summary()["llm"]
    assert s["p95_ms"] == 19.0  # index min(19, round(0.95*19)) = 18 -> value 19
    assert s["median_ms"] == 10.5


def test_stage_context_manager_records() -> None:
    t = Timings()
    with t.stage("tts"):
        pass
    assert t.last("tts") is not None
    assert t.last("tts") >= 0.0


def test_last_missing_stage_is_none() -> None:
    assert Timings().last("nope") is None


def test_clear() -> None:
    t = Timings()
    t.record("stt", 5.0)
    t.clear()
    assert t.summary() == {}


def test_latency_report_seconds_and_missing_keys() -> None:
    t = Timings()
    t.record("llm_first_token", 1500.0)
    t.record("utterance_to_first_audio", 900.0)
    report = t.latency_report()
    assert report["ttt_text_latency_s"] == 1.5
    assert report["total_latency_s"] == 0.9
    assert report["stt_latency_s"] is None
    assert report["tts_latency_s"] is None


def test_markdown_table() -> None:
    t = Timings()
    t.record("stt", 12.0)
    table = t.markdown()
    assert table.startswith("| Stage |")
    assert "stt" in table
