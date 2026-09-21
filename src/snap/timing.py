"""Per-stage timing — the benchmark backbone.

Every pipeline stage wraps its work in `time_stage("stt")`. At the end of a run,
`report()` emits the markdown table that goes straight into the submission docs.
"""

import json
import statistics
import time
from collections import defaultdict
from contextlib import contextmanager


class Timings:
    def __init__(self) -> None:
        self._samples: defaultdict[str, list[float]] = defaultdict(list)

    @staticmethod
    def now_ms() -> float:
        return time.perf_counter() * 1000.0

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._samples[name].append((time.perf_counter() - start) * 1000.0)

    def record(self, name: str, ms: float) -> None:
        self._samples[name].append(ms)

    def last(self, name: str) -> float | None:
        xs = self._samples.get(name)
        return xs[-1] if xs else None

    def clear(self) -> None:
        self._samples.clear()

    def latency_report(self) -> dict[str, float | None]:
        """Deepgram-style per-turn attribution (seconds, latest values).

        Mirrors their LatencyReport field names so the benchmark write-up can
        say the observability model matches theirs: stt_latency, ttt_text_latency
        (LLM first token), tts_latency, and end-to-end utterance_to_first_audio.
        """
        def seconds(name: str) -> float | None:
            v = self.last(name)
            return round(v / 1000.0, 3) if v is not None else None

        return {
            "stt_latency_s": seconds("stt"),
            "ttt_text_latency_s": seconds("llm_first_token"),
            "llm_total_s": seconds("llm_total"),
            "tts_latency_s": seconds("tts") or seconds("tts_console"),
            "tool_answer_no_llm_s": seconds("tool_answer_no_llm"),
            "total_latency_s": seconds("utterance_to_first_audio"),
        }

    def summary(self) -> dict[str, dict[str, float]]:
        out = {}
        for name, xs in self._samples.items():
            ordered = sorted(xs)
            p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
            out[name] = {
                "n": len(xs),
                "median_ms": round(statistics.median(xs), 1),
                "min_ms": round(min(xs), 1),
                "max_ms": round(max(xs), 1),
                "p95_ms": round(p95, 1),
            }
        return out

    def markdown(self) -> str:
        rows = [
            "| Stage | n | median (ms) | p95 | min | max |",
            "|---|---|---|---|---|---|",
        ]
        for name, s in self.summary().items():
            rows.append(
                f"| {name} | {s['n']} | {s['median_ms']} | {s['p95_ms']} | {s['min_ms']} | {s['max_ms']} |"
            )
        return "\n".join(rows)

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.summary(), f, indent=2)


TIMINGS = Timings()
