#!/usr/bin/env bash
# Fast-variant build orchestrator for a third-party Qwen3.8-family checkpoint.
# Runs AFTER the syv stack's drafter/gen_data.py has produced
# drafter/data/gen.jsonl from your corpus. Executes, in order (each GPU phase
# in its own container; the GPU must be free — this script refuses otherwise):
#
#   1. vocab_build.py            (CPU)  target draft vocab + coverage report
#   2. capture.py                (GPU)  hidden states for every token
#   3. train_mtp.py --eval-only --dump-hessians (GPU, upstream recipe verbatim)
#   4. gptq_lm_head.py           (GPU)  int4 lm_head -> <tmp-dir>
#   5. build_draft_vocab.py      (CPU)  int4 draft head from your ids
#   6. requant_mtp.py            (GPU)  int4 MTP -> <fast-dir>
#   7. verify                    (CPU)  verifier/verify_model.py against the new dir
#
# The source model dir is never modified. Configure everything through the
# environment (or edit the defaults):
#   SYV_REPO     the syv stack checkout (default ~/git/qwen38-27b-rtx3090)
#   COMPANION    this repository (default: script's parent dir)
#   IMAGE        the pinned syv container image
#   SRC_MODEL    dir of the prepared (int8-heads) third-party model
#   FAST_MODEL   output fast-variant dir
#   TMP_DIR      intermediate dir for the int4-lm_head stage
#   IDS          your draft-vocab ids file (from vocab_build.py)
set -euo pipefail

SYV_REPO="${SYV_REPO:-$HOME/git/qwen38-27b-rtx3090}"
COMPANION="$(cd "$(dirname "$(readlink -f "$0")")/.." && pwd)"
IMAGE="${IMAGE:-ghcr.io/syv-ai/qwen38-27b-rtx3090:latest}"
SRC_MODEL="${SRC_MODEL:?set SRC_MODEL to the prepared third-party model dir}"
FAST_MODEL="${FAST_MODEL:?set FAST_MODEL to the fast-variant output dir}"
TMP_DIR="${TMP_DIR:-$(dirname "$SRC_MODEL")/tmp-lm4}"
IDS="${IDS:-$SYV_REPO/drafter/data/draft_vocab_ids.json}"
BITS="${BITS:-4}"
CALIB_ROWS="${CALIB_ROWS:-300000}"

mkdir -p "$SYV_REPO/drafter/runs"

run() {  # run <phase-name> <cmd...>   (container phases)
  local name=$1; shift
  echo "===== $name ====="
  docker run --rm --name "fast-$name" --gpus all --ipc host \
    -v "$SYV_REPO:/repo" -v "$COMPANION:/companion:ro" -v qwen38-27b-rtx3090_qwen-cache:/cache \
    -e SYV_REPO=/repo -e HOME=/cache -e PATH=/app/venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
    -e VLLM_USE_FLASHINFER_SAMPLER=0 \
    -w /repo --entrypoint bash "$IMAGE" -c "$*" 2>&1 | tee "$SYV_REPO/drafter/runs/$name.log" | tail -5
}

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
[ "$USED" -gt 4000 ] && { echo "GPU busy ($USED MiB) — stop the serving stack first"; exit 1; }

echo "===== 1. vocab (container, CPU) ====="
docker run --rm -v "$SYV_REPO:/repo" -v "$COMPANION:/companion:ro" -e HOME=/tmp \
  --entrypoint bash "$IMAGE" -c "/app/venv/bin/python /companion/swift_qwen_syv/vocab_build.py \
    --gen /repo/drafter/data/gen.jsonl --tokenizer '$SRC_MODEL' \
    --out-ids '$IDS' --out-report /repo/drafter/data/vocab_report.json" \
  2>&1 | tee "$SYV_REPO/drafter/runs/vocab.log" | tail -30

run capture   "pip -q install ninja && /app/venv/bin/python drafter/capture.py"

run hessians  "/app/venv/bin/python drafter/train_mtp.py --out drafter/runs/target --eval-only 1 \
                 --draft-ids '$IDS' --max-seqs 400 --val-frac 0.4 --depths 2 \
                 --dump-hessians drafter/runs/target/mtp_hessians.pt"

run gptq      "/app/venv/bin/python /companion/swift_qwen_syv/gptq_lm_head.py \
                 '$SRC_MODEL' '$TMP_DIR' --bits $BITS --calib-rows $CALIB_ROWS"

run draftvocab "/app/venv/bin/python prepare/build_draft_vocab.py '$TMP_DIR' --ids '$IDS'"

run assemble  "/app/venv/bin/python /companion/swift_qwen_syv/requant_mtp.py \
                 '$TMP_DIR' '$FAST_MODEL' drafter/runs/target/mtp_hessians.pt --bits $BITS \
                 --orig '$SRC_MODEL'"

echo "===== 7. verify (CPU, stdlib core) ====="
python3 "$COMPANION/verifier/verify_model.py" "$FAST_MODEL"

echo "===== build complete: $FAST_MODEL ====="
echo "Regenerate calibration data (hidden.npy) by re-running capture.py over"
echo "gen.jsonl if you deleted it; the Hessians under drafter/runs/target/ are kept."
