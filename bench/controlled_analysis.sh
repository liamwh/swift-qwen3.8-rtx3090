#!/usr/bin/env bash
# Regenerates bench/results/controlled-summary.json from the committed raw legs
# of the 2026-10-02 controlled session (bench/results/controlled/*.json and
# bench/results/controlled-ladder.jsonl, produced by
# infra hosts/zeus/swift-qwen/controlled_ladder.py).
#
# Leg ids, in run order:
#   c01 t10  c02 t15  c03 t15  c04 t10          ABBA, 1.0 and 1.5 fast, MTP long
#   c05 t10off c06 t15off c07 t15off c08 t10off  no-speculation controls
#   c09 i15                                      1.5 int8 heads (not first)
#   c10 v10  c11 vu  c12 t15  c13 t15  c14 vu  c15 v10   1.5 target, three lists
#   c16 t15  c17 t10  c18 t10  c19 t15           second order block, B A A B
#   c20 t10  c21 x15  c22 xu  c23 xu  c24 x15  c25 t10   1.0 target, three lists
# Configs: t10/t15 = published 1.0 / current 1.5 fast build with their own
# lists; v10 = 1.5 target + 1.0 list; vu = 1.5 target + union; x15 = 1.0 target
# + 1.5 list; xu = 1.0 target + union; i15 = 1.5 int8 heads + base-Qwen list.
# Each leg has a T=0.7 suite (suffix -t07) and a T=0 suite (suffix -t0).
set -euo pipefail
cd "$(dirname "$0")/.."
R=bench/results
g() { # g <name> <legs...> -> --group name=leg-t07,... for both sampling tags
  local name=$1; shift
  local a="" b=""
  for leg in "$@"; do a+="${leg}-t07,"; b+="${leg}-t0,"; done
  printf -- '--group %s=%s --group %s_t0=%s ' "$name" "${a%,}" "$name" "${b%,}"
}
python3 bench/analyze_runs.py --ladder $R/controlled-ladder.jsonl --suite-dir $R/controlled \
  $(g t10 c01 c04 c17 c18) $(g t15 c02 c03 c16 c19) \
  $(g off10 c05 c08) $(g off15 c06 c07) $(g i15 c09) \
  $(g l15 c12 c13) $(g l10 c10 c15) $(g lu c11 c14) \
  $(g m10 c20 c25) $(g m15 c21 c24) $(g mu c22 c23) \
  --compare t15:t10 --compare t15_t0:t10_t0 \
  --compare off15:off10 --compare off15_t0:off10_t0 \
  --compare l10:l15 --compare lu:l15 --compare l10_t0:l15_t0 --compare lu_t0:l15_t0 \
  --compare m15:m10 --compare mu:m10 --compare m15_t0:m10_t0 --compare mu_t0:m10_t0 \
  --out $R/controlled-summary.json
