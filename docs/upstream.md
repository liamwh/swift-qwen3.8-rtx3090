# Upstream relationship and contribution status

Decision rule applied to every piece of this work: **generic fixes belong
in syv; target-specific methodology belongs here.**

| piece | where it belongs | status |
| verify.sh rejects valid int4-GPTQ fast variants (`num_bits == 8` hard-coded for lm_head) | upstream | [PR #122](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/122): accept 4 or 8 when the packed/scale geometry is self-consistent with the declared bits |
| duplicate-tensor / shard-collision detection | upstream | [PR #123](https://github.com/syv-ai/qwen38-27b-rtx3090/pull/123): whole-dir scan — a mapped tensor present in more than its mapped shard is a real corruption class (stale hardlinked tensors) that every existence check misses |
| multi-shard head requant (`quant_heads_multishard.py`) | upstream candidate | kept here: generalises `prepare/quant_heads_stream.py` to checkpoints whose lm_head/embed/MTP live in different shards; offered upstream if the maintainers want third-party layout coverage there |
| target-calibrated fast-variant pipeline (corpus -> own-output vocab -> GPTQ heads) | here | a recipe around upstream scripts; upstreaming it would add target-specific opinion (workload mixes) to a stack that is deliberately model-agnostic. If syv later grows a generic "fast variant for any compatible checkpoint" track, this repo's pipeline is the reference implementation to fold in |
| serving-gate systemd/Nix patterns | here (examples/) | deployment opinion, not stack functionality |

The upstream PRs are deliberately narrow and independent: no Zeus/private
infrastructure, no Swift-specific naming — the int4 fast-variant layout
they validate is the one upstream's own `drafter/` produces and
`prepare/fetch_fast_variant.py` distributes, so the false-rejection bug
bites upstream users today without any third-party checkpoint involved.
