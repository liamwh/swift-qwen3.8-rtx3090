#!/usr/bin/env python3
"""Assemble the fast variant: GPTQ int4 MTP (Hessians from the TARGET's own
states) into a dst dir that already carries the int4 lm_head + int4 draft
head (gptq_lm_head.py + prepare/build_draft_vocab.py output).

Layout-adapted port of the syv stack's drafter/requant_mtp_gptq.py (same
math, same Hessian consumption). Differences forced by third-party layouts:
  - tolerant small-file copy list (no quantization_config.json /
    processor_config.json in third-party dirs)
  - keeps hardlinking model-nonquant.safetensors, then REWRITES it norms-only
    (own inode): the hardlink source still carries the stale int8 MTP packed
    tensors, and vLLM reads every key in an opened shard, so they would
    collide with the int4 ones in model_extra_tensors ("have/got" shape
    assert at best, silent wrong weights at worst). This exact collision is
    what the verifier's duplicate-tensor scan exists to catch.
  - rewrites the dst index so the eight MTP linears' packed tensors map to
    model_extra_tensors.safetensors (the base-Qwen layout has MTP there from
    the start; a third-party index still points at model-nonquant).

Usage (inside the syv stack container, from its repo root):
  python requant_mtp.py <src-with-int4-lm_head> <dst-fast-dir> <mtp_hessians.pt> \
      --bits 4 --orig <int8-model-dir> [--bf16-mtp <file-with-bf16-mtp-weights>]

The bf16 MTP weights are read from --bf16-mtp, else
<orig>/model_extra_tensors.safetensors.bak-mtp (the backup convention of
quant_heads_multishard.py's predecessor flow).
"""
import json, os, shutil, sys

DRAFTER = os.path.join(os.environ.get("SYV_REPO", "."), "drafter")
sys.path.insert(0, DRAFTER)
import torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from gptq_utils import gptq_quantize, dequant

S, D, HP = sys.argv[1].rstrip("/") + "/", sys.argv[2].rstrip("/") + "/", sys.argv[3]
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 4
ORIG = (sys.argv[sys.argv.index("--orig") + 1] if "--orig" in sys.argv else S).rstrip("/") + "/"
BF16_MTP = sys.argv[sys.argv.index("--bf16-mtp") + 1] if "--bf16-mtp" in sys.argv else None
GROUP = 128
LIN = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj", "mtp.layers.0.mlp.up_proj",
       "mtp.layers.0.self_attn.q_proj", "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
       "mtp.layers.0.self_attn.o_proj"]

os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f.endswith(".safetensors") and not f.startswith("model_extra") and not os.path.exists(D + f):
        os.link(S + f, D + f)                        # body shards + model-nonquant + lm_head shard
for f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "preprocessor_config.json", "vocab.json",
          "merges.txt", "special_tokens_map.json", "mtp_draft_vocab_ids.pt",
          "mtp_draft_vocab_ids.json"]:
    if os.path.exists(S + f):
        shutil.copy(S + f, D + f)

src_bf16 = BF16_MTP or next((ORIG + "model_extra_tensors.safetensors" + s
                             for s in (".bak-mtp", ".bak-orig") if os.path.exists(ORIG + "model_extra_tensors.safetensors" + s)), None)
if not src_bf16:
    sys.exit("no bf16 MTP source found; pass --bf16-mtp <safetensors with mtp.*.weight>")
with safe_open(src_bf16, "pt") as f:
    bf = {k: f.get_tensor(k) for k in f.keys()}
missing = [m + ".weight" for m in LIN if m + ".weight" not in bf]
if missing:
    sys.exit(f"bf16 MTP source {src_bf16} lacks {missing}")
tensors = {}
with safe_open(S + "model_extra_tensors.safetensors", "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        if any(k.startswith(m + ".") for m in LIN):
            continue
        tensors[k] = f.get_tensor(k)
HS = torch.load(HP)
for m in LIN:
    w = bf[m + ".weight"].cuda()
    q, scale = gptq_quantize(w, HS[m].cuda(), bits=BITS, group=GROUP, blocksize=GROUP)
    rel = ((dequant(q, scale) - w.float()).norm() / w.float().norm()).item()
    print(f"  GPTQ {m}: {tuple(w.shape)} int{BITS} rel err {rel:.4f}")
    out_f, in_f = w.shape
    tensors[m + ".weight_packed"] = pack_to_int32(q.cpu(), BITS, packed_dim=1).contiguous()
    tensors[m + ".weight_scale"] = scale.to(torch.float16).cpu().contiguous()
    tensors[m + ".weight_shape"] = torch.tensor([out_f, in_f], dtype=torch.int64)
    del w, q, scale
    torch.cuda.empty_cache()
if os.path.exists(D + "model_extra_tensors.safetensors"):
    os.remove(D + "model_extra_tensors.safetensors")
save_file(tensors, D + "model_extra_tensors.safetensors", metadata=meta or {"format": "pt"})

# index: MTP packed tensors now live in model_extra_tensors.safetensors
idx = json.load(open(S + "model.safetensors.index.json"))
wm = idx["weight_map"]
moved = 0
for m in LIN:
    for suf in ("weight_packed", "weight_scale", "weight_shape"):
        k = f"{m}.{suf}"
        if k in wm and wm[k] != "model_extra_tensors.safetensors":
            wm[k] = "model_extra_tensors.safetensors"
            moved += 1
json.dump(idx, open(D + "model.safetensors.index.json", "w"), indent=2)
print(f"index remapped ({moved} entries -> model_extra_tensors.safetensors)")

c = json.load(open(S + "config.json"))
qc = c["quantization_config"]
g3 = qc["config_groups"]["group_3"]
assert g3["targets"] == ["re:^mtp\\..*"], g3["targets"]
g3["weights"]["num_bits"] = BITS
g1 = qc["config_groups"]["group_1"]
assert g1["weights"]["num_bits"] == BITS, "src dir must already have the int4 lm_head"
json.dump(c, open(D + "config.json", "w"), indent=2)
# model-nonquant is hardlinked from the int8 model and may still carry its
# STALE int8 mtp.* packed tensors; vLLM reads every key in an opened shard,
# so the int8 mtp.fc would collide with the int4 one in extras ("have/got"
# shape assert). Rewrite it norms-only (own inode).
_stale = D + "model-nonquant.safetensors"
if os.path.exists(_stale):
    with safe_open(_stale, "pt") as _f:
        _meta = _f.metadata()
        _keep = {k: _f.get_tensor(k) for k in _f.keys()
                 if not any(k.endswith(s) for s in (".weight_packed", ".weight_scale", ".weight_shape"))}
    os.remove(_stale)
    save_file(_keep, _stale, metadata=_meta or {"format": "pt"})
    print(f"model-nonquant rewritten norms-only ({len(_keep)} tensors)")

print("done", D)
