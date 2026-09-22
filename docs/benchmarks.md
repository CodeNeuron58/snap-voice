# Benchmarks

Snap's numbers, with exactly how each was obtained — measured values vs.
profiled values are always labeled. `snap bench` prints this table shape and
refreshes `bench_results*.json`; paste new runs here.

## Methodology

- Fixed 10-prompt set (5 EN, 1 HI, 3 tool-intended, 1 creative), 3 runs each
  (`snap bench --runs 3`), warm-up turn excluded, **fresh conversation per run**
  (a 30-turn session degrades small-model coherence; demo reality is short
  fresh conversations).
- Per-stage attribution via `snap.timing` (`llm_first_token` = prefill-inclusive
  time to first token, `llm_total` = full reply, `tool_answer_no_llm` = the
  deterministic pre-router path — zero LLM tokens). Medians + p95 reported.
- **Two-tier tool measurement.** By default the regex pre-router answers
  arithmetic/conversion prompts deterministically (0 LLM tokens) — those rows
  measure the *router*, by design. `snap bench --no-preroute` disables it and
  runs the same prompts through the LLM's Hermes-style tool path (JSON-schema
  tools, `<tool_call>` protocol, `<tool_response>` feedback, max 2 hops),
  measuring real model tool adherence and its latency. Report both; never mix.
- CPU numbers below: dev laptop, Windows x64, Qwen3-4B-Instruct-2507 Q4_K_M via
  llama.cpp. Not the target hardware — the target is Snapdragon NPU.

## Measured (CPU baseline, 2026-09-21)

| Stage | n | median | p95 | Source |
|---|---|---|---|---|
| LLM first token (prefill-inclusive) | 18 | 2,785 ms | 18,651 ms | `snap bench` (measured) |
| LLM full reply (~100–150 tok) | 18 | 10,708 ms | 22,762 ms | `snap bench` (measured) |
| Tool answers (pre-routed, no LLM) | 12 | 0.1 ms | 1.2 ms | `snap bench` (measured) |
| STT (3 s utterance) | — | — | — | not in text bench |
| TTFA (end-to-end voice) | — | — | — | pending mic-path probe |

## Profiled (Snapdragon NPU — AI Hub, X2 Elite CRD)

To be filled from `scripts/profile_on_aihub.py` runs; raw job JSON + screenshots
go to `docs/evidence/` first, summarized here after.

| Stage | Expected (AI Hub model pages) | Profile job | Delta vs CPU |
|---|---|---|---|
| STT (Whisper-Small, w8a16) | encoder 60.5 ms + 6.58 ms/tok ≈ 190 ms/pass | pending | — |
| LLM prefill (Qwen3-4B-Instruct-2507) | ~150–400 ms (2,831 tok/s) | pending | ~10–19× |
| LLM decode | 42.6 tok/s (vs ~14 tok/s CPU) | pending | ~3× |
| TTFA (composed) | ~0.9 s target | pending | — |

\* AI Hub page numbers are Qualcomm's published hosted-device measurements; our
own profile jobs will replace/augment them. Never mix the two without a label.
