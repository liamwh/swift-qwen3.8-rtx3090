# swift-qwen3.8-rtx3090

Reproducibly turn a compatible Swift-Qwen3.8-27B W4A16 checkpoint into a
[syv](https://github.com/syv-ai/qwen38-27b-rtx3090)-optimised single-GPU
**fast variant** with a target-calibrated speculative draft vocabulary,
GPTQ int4 lm_head/MTP, structural verification, and RTX 3090 serving
benchmarks.

A focused companion project, not a fork: it drives the syv stack's own
`drafter/` recipe unmodified and adds what was missing — deriving the
speculative artefacts from the *target* model's own outputs instead of
reusing base-Qwen's.

## Why

The syv stack serves Qwen3.8-27B on a single RTX 3090 in three tiers:

1. generic third-party W4A16 checkpoints can be made servable (int8 heads),
   but every model-specific artefact — the draft vocabulary, the GPTQ
   calibration — still comes from base Qwen;
2. base Qwen itself has a highly optimised fast variant (int4-GPTQ heads,
   own-output draft vocab), worth ~15% plus materially better speculative
   acceptance;
3. nothing existed for a third-party finetune of that family.

That gap is not cosmetic. Speculative artefacts are
distribution-dependent: the MTP drafter scores a ~40k-row slice of
lm_head, and a token outside that slice can never be drafted — every miss
is a guaranteed rejection that truncates the chain. On our workload, the
shipped base-Qwen id list covered **96.7%** of what Swift actually emits;
a list counted over Swift's own outputs covers **99.8%** (99.86% on code),
with only 18,701 of 40,960 ids shared between the two lists. GPTQ
calibration is the same story: Hessians from the target's own hidden
states give an int4 lm_head at KL 0.00234 vs 0.00707 for round-to-nearest.

Measured result (all legs same night, medians, one RTX 3090 — see
[docs/benchmarks.md](docs/benchmarks.md)):

| | decode tok/s | MTP acceptance | quote acceptance |
|---|---|---|---|
| Swift int8 (baseline) | 94.0 | 0.630 | 0.962 |
| **Swift-fast (this pipeline)** | **98.4** | **0.660** | **0.998** |
| Qwen fast (upstream, reference) | 98.2 | 0.634 | — |

Swift-fast reaches base-Qwen-fast parity on the daily profile with better
acceptance, and the DFlash2 acceptance deficit vs Qwen disappears.

## What's here

```text
config/workload.example.json   workload mix (categories, weights, think rates)
swift_qwen_syv/
  make_corpus.py               workload-weighted prompt corpus (deterministic)
  vocab_build.py               target draft vocab + held-out coverage/convergence
  quant_heads_multishard.py    int8 heads for multi-shard AWQ checkpoints
  gptq_lm_head.py              GPTQ int4 lm_head, target-calibrated
  requant_mtp.py               GPTQ int4 MTP + fast-variant assembly
  build_fast.sh                end-to-end orchestrator (pinned containers)
verifier/
  verify_model.py              structural model-dir verifier (stdlib core)
  test/                        selftest + fixtures, incl. the stale-tensor
                               collision regression
bench/
  swift_ab.py                  A/B harness: suite/prefix/long/quote/reason/agent
                               modes + acceptance via /metrics counters
  quality_ab.py                9-task quality battery (mechanical checks)
  results/                     committed summary artefacts (the real numbers)
docs/                          methodology, benchmarks, troubleshooting
examples/systemd, examples/nix fail-closed serving-gate patterns
```

## Prerequisites

- One NVIDIA GPU (24 GB; everything here was measured on a 250 W RTX 3090,
  sm86)
- The [syv stack](https://github.com/syv-ai/qwen38-27b-rtx3090) checked
  out and working (`bash verify.sh --install` passes) — its container image
  and `drafter/` tooling are the runtime for the GPU phases
- ~90 GB free disk for the checkpoint, variant dirs and hidden-state
  capture (74 GB memmap, deleted after the build)
- Independent access to a compatible W4A16 checkpoint (see
  [Licensing](#licensing); Swift checkpoints are gated)
- For the verifier selftest only: Python 3.10+ stdlib — nothing else

## Quick start

```bash
# 0. selftest the verifier (stdlib only, ~1 s)
python3 verifier/test/selftest.py

# 1. fetch the checkpoint into the syv checkout (gated; see Licensing)
cd $SYV_REPO && python prepare/fetch_thirdparty.py TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ

# 2. int8 heads (multi-shard aware) — the model now serves; keep as rollback
python swift_qwen_syv/quant_heads_multishard.py $SYV_REPO/models/Swift-Qwen3.8-27B-W4A16

# 3. your workload corpus — EDIT THE MIX FIRST (config/workload.example.json)
python swift_qwen_syv/make_corpus.py --config my-workload.json \
    --out $SYV_REPO/drafter/data/prompts.jsonl

# 4. generate outputs with the target (upstream, ~1.7 h GPU)
cd $SYV_REPO && venv/bin/python drafter/gen_data.py

# 5-10. draft vocab -> capture -> Hessians -> int4 heads -> assemble -> verify
SRC_MODEL=$SYV_REPO/models/Swift-Qwen3.8-27B-W4A16 \
FAST_MODEL=$SYV_REPO/models/Swift-Qwen3.8-27B-W4A16-fast \
bash swift_qwen_syv/build_fast.sh

# serve it through the syv launchers; verify any time:
python3 verifier/verify_model.py $SYV_REPO/models/Swift-Qwen3.8-27B-W4A16-fast
```

Customise the workload mix before trusting any downstream artefact: the
draft vocabulary is only as representative as the corpus. The defaults
approximate one user's coding-agent profile (Rust/TypeScript-heavy) and
are documented as such, not as a universal mix.

### Benchmarking

```bash
BASE=http://127.0.0.1:8090/v1 KEY=... LABEL=swiftfast \
  python3 bench/swift_ab.py suite out.json      # decode/TTFT/acceptance
BASE=... python3 bench/swift_ab.py quote out.json   # 8k-token reproduction
BASE=... THINKING=1 python3 bench/quality_ab.py out.json
```

Acceptance is read from vLLM's `/metrics` spec-decode counters (accepted /
drafted / drafts), so numbers are the server's own view, not a prompt-level
approximation.

## Verification

`verifier/verify_model.py` is the fail-closed replacement for pointing the
upstream verifier at a fast-variant dir (upstream's asserts an int8
lm_head, falsely rejecting the int4-GPTQ layout its own pipeline
produces). It auto-detects the layout and checks:

- config/architecture self-consistency (layer counts vs layer_types,
  attention-interval pattern, untied embeddings)
- compressed-tensors groups: targets, per-group bit widths, symmetry,
  group size, zero-point rules
- packed/scale/shape geometry for lm_head, embeddings, all 8 MTP linears
  and the draft head — checked against bf16 geometry x declared bits
- draft-vocab artefact: unique, in-range ids; rows == draft head rows;
  optional provenance cross-check against your corpus-derived list
- index completeness and a **whole-directory duplicate-tensor scan**: every
  index-mapped name must live in exactly its mapped shard

That last check exists because of a real bug: a hardlinked shard still
carrying superseded int8 MTP tensors passed every existence check and
collided with the int4 tensors at load time. The selftest
(`verifier/test/selftest.py`) includes that exact corruption as a
regression fixture, plus bit-width mismatch, draft-row mismatch,
duplicate-id and missing-shard cases.

Machine-specific concerns (systemd, API keys, ports) are not part of the
library; `examples/` shows the serving-gate pattern instead.

## Upstream relationship

- Generic fixes belong in syv and are contributed there, not carried here:
  verify.sh's int8-only lm_head assertion, and the duplicate-tensor shard
  collision check, are upstreamed separately (see
  [docs/upstream.md](docs/upstream.md) for status).
- Target-specific methodology (workload corpus, own-output draft vocab,
  calibrated GPTQ heads) stays in this companion repo: it is a recipe
  around upstream scripts, not a change to them.
- `quant_heads_multishard.py` generalises upstream's
  `prepare/quant_heads_stream.py` to checkpoints whose heads live in
  different shards; it lives here until/unless the generalisation lands
  upstream.

## Licensing

- **This repo's code**: Apache-2.0 (see LICENSE, NOTICE). It contains no
  model weights, no checkpoint-derived tensors, and no generated outputs —
  only tooling, configs, docs and benchmark summaries.
- **Swift-Qwen3.8-27B** (ukisai) and its quantisations (e.g.
  TheUnderscore's W4A16 AWQ) are **gated** under the Swift Open License
  v1.0: free for personal/research/evaluation use and for organisations
  under US$1M annual recurring revenue; above that, commercial use needs a
  separate Swift Enterprise License. The pipeline here therefore never
  redistributes derived artefacts (int4 heads, MTP weights, draft-vocab id
  lists, fast-variant shards) — you build them locally from a checkpoint
  you are authorised to use. Regeneration is exactly what this repo makes
  reproducible.
- Not legal advice; read the checkpoint card before use.

## Known limitations

- Numbers are for one 27B AWQ body on one RTX 3090 with one stack pin
  (vLLM 0.28.0 + the syv patches). Different body quant, GPU or workload:
  re-measure (that's what `bench/` is for).
- The MTP-vs-DFlash2 "daily profile" conclusion reflects a coding-agent
  workload with long contexts and thinking-on turns; DFlash2 wins raw
  short-context tok/s and remains the better specialised mode there.
- The draft vocab id list is a function of your corpus; a workload mix
  that doesn't match what you serve wastes the coverage gain.
- English-language engineering workload; multilingual output distributions
  will differ (upstream's own experience with a Danish-chat mix is the
  cautionary tale that motivated this repo's existence).

## Attribution

- [syv-ai/qwen38-27b-rtx3090](https://github.com/syv-ai/qwen38-27b-rtx3090)
  (Apache-2.0) — the serving stack, the drafter recipe, and the math this
  pipeline reuses. Three scripts here are derived from it (NOTICE).
- [ukisai/Swift-Qwen3.8-27b](https://huggingface.co/ukisai/Swift-Qwen3.8-27b)
  and
  [TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ](https://huggingface.co/TheUnderscore/Swift-Qwen3.8-27b-W4A16-AWQ)
  — the target checkpoint (gated, Swift Open License v1.0).
