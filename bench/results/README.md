# Benchmark result artefacts

- `profiles.jsonl` — quick profile ladder, pre-fast-variant (one line per
  profile; the full per-prompt JSON lives on the measurement host)
- `fast-ladder.jsonl` — the five-leg fast-variant comparison (same night,
  REPS=4, medians)

Labels: `swift` = Swift-Qwen3.8-27B W4A16 prepared with the int8-heads
layout; `swiftfast` = the fast variant this repo builds; `qwen` = base
Qwen3.8-27B on the upstream fast variant; profiles `mtp-long`
(SPEC=mtp, 114,688 ctx) and `dflash2-fast` (SPEC=dflash2, 46,080 ctx).
Reproduce with `bench/swift_ab.py` (see its docstring); numbers and how to
read them: [docs/benchmarks.md](../../docs/benchmarks.md).
