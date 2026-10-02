# Benchmark result artefacts

- `profiles.jsonl` — quick profile ladder, pre-fast-variant (one line per
  profile; the full per-prompt JSON lives on the measurement host)
- `fast-ladder.jsonl` — the five-leg Swift 1.0 fast-variant comparison
  (2026-09-15, same night, REPS=4, medians)
- `swift15-ladder.jsonl` — the five-leg Swift 1.5 ladder (2026-10-01,
  REPS=4, medians; leg `B2` re-measures the published Swift 1.0 fast build
  the same night as the drift control)

Labels: `swift`/`swiftfast` = Swift-Qwen3.8-27B (1.0) prepared with the
int8-heads layout / this repo's fast variant; `swift15`/`swift15fast` =
the same for Swift 1.5 on UkisAI's official AWQ; `qwen` = base
Qwen3.8-27B on the upstream fast variant; profiles `mtp-long`
(SPEC=mtp, 114,688 ctx), `off-long` (no speculation) and `dflash2-fast`
(SPEC=dflash2, 46,080 ctx).
Reproduce with `bench/swift_ab.py` (see its docstring); numbers and how to
read them: [docs/benchmarks.md](../../docs/benchmarks.md).
