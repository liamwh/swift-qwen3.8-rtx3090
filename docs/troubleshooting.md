# Troubleshooting

Every failure below was hit for real during development. Most produce
errors that name neither the file nor the cause.

## "have/got" shape assert or wrong tensors at load, model boots fine

**The stale-tensor collision.** vLLM reads every key of every shard it
opens, not just the index-mapped ones. If a shard in the served dir still
carries a superseded tensor with the same name as one mapped elsewhere
(our case: a hardlinked `model-nonquant.safetensors` still carrying the
int8 `mtp.fc.weight_packed` from the int8 build, colliding with the int4
one in `model_extra_tensors.safetensors`), you get a shape assert at best
and silently wrong weights at worst. Every "does the mapped tensor exist"
check passes.

Fix: `requant_mtp.py` rewrites `model-nonquant` norms-only (own inode);
detect with `verifier/verify_model.py` (the duplicate/stale tensor scan is
the check that exists for exactly this). The selftest ships a fixture of
this corruption.

## Hardlink corruption: edits land in both dirs

Body shards are hardlink-shared to save disk. Any in-place rewrite of a
shared shard (e.g. `save_file()` over a hardlinked path, which
`build_draft_vocab.py` does to `model_extra_tensors.safetensors`)
truncates and rewrites the shared inode — silently modifying the OTHER
model dir. `gptq_lm_head.py` copies `model_extra_tensors.safetensors`
instead of hardlinking for this reason. Rule: hardlink body shards, copy
anything a later step rewrites.

## verify.sh fails a valid int4 fast variant

Upstream's verify.sh requires the lm_head group to be `num_bits == 8`, so
it rejects the int4-GPTQ layout its own drafter pipeline produces. Use
`verifier/verify_model.py` (auto-detects the layout and validates
self-consistency), or pin the layout with `--expect int4|int8`.

## Upstream quant scripts reject the checkpoint

`prepare/quant_lm_head.py` + friends require the base-Qwen single-shard
layout; third-party AWQ exports split lm_head/embed/MTP across shards.
`quant_heads_multishard.py` handles that (one streaming pass per head
shard; ~10 GB peak RSS on a 27B checkpoint).

## Draft ids artefact and the 40,960 assumption

The draft vocab does NOT need exactly 40,960 ids. A target-derived list
has as many rows as the corpus produced distinct output tokens (ours:
25,879). What must hold: the draft head's rows == the ids artefact's
count, ids unique and within vocab range — which is what the verifier
checks. If your ids artefact is `.pt` and torch is unavailable, the
verifier type-checks nothing (WARN); export `.json` or run with torch
installed.

## Server dies on "ReasoningConfig: failed to tokenize reasoning strings"

The model dir has no usable `tokenizer.json`; transformers silently hands
back a 1-token vocabulary instead of failing. Copy `tokenizer.json` +
`tokenizer_config.json` in from the base checkpoint dir. The verifier
checks for the files; upstream verify.sh additionally encodes `<think>`.

## Container build phases fail on GPU memory

`build_fast.sh` refuses to start if >4 GB is already allocated — stop the
serving stack first (both phases want the whole card). `capture.py`'s
hidden-state memmap needs ~74 GB of free DISK for a 27B model at 4M tokens.

## KV-cache boot lottery

vLLM's activation-profile draw swings the available KV pool (~0.9 GiB
observed between attempts). A long-context max_model_len near the pool
limit fails roughly one boot in five and succeeds on the container/unit
retry. Pinning `--kv-cache-memory` (a successful boot logs the suggestion)
makes boots deterministic at a slightly smaller pool — the retry cycle is
the documented convention upstream.

## Acceptance looks fine but decode is slow

Check which profile you actually booted (the harness reads `/metrics`
spec-decode counters; a SPEC=off control decodes ~50 tok/s where MTP-long
does ~100). And remember the last ~7% of DFlash2-profile decode speed on
an AWQ body is weight-read bytes, not drafter quality — no vocab fixes it.
