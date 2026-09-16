#!/usr/bin/env python3
"""GPTQ-quantize an lm_head to int4 (g128, sym) using captured hidden states
from the TARGET model's own outputs — a layout-adapted port of the syv
stack's drafter/gptq_lm_head.py (same math via the same gptq_utils; reads
the same data/hidden.npy).

Why calibrate on the target's hidden states: the lm_head Hessian is a
property of the distribution the model actually emits. Round-to-nearest int4
cost measurably more KL than GPTQ calibrated on target outputs (measured
0.00707 RTN vs 0.00234 GPTQ on our Swift run; the base-Qwen fast variant
shipped at 0.0029 on base-Qwen states).

Differences from upstream, forced by third-party checkpoint layouts:
  - tolerant small-file copy list (third-party dirs have no
    quantization_config.json / processor_config.json; config.json embeds
    quantization_config)
  - model-nonquant.safetensors (the bf16 MTP/norm shard) is hardlinked too
  - model_extra_tensors.safetensors is COPIED, not hardlinked: upstream
    hardlinks it and a later build_draft_vocab.py save_file() over the path,
    truncating the shared inode — identical-content luck for the base model,
    silent corruption of the source model for a variant dir.

Usage (inside the syv stack container, from its repo root):
  python gptq_lm_head.py <src-model-dir> <dst-model-dir> \
      --bits 4 --calib-rows 300000 [--bf16-shard <file-with-bf16-lm_head>] [--mse-clip]

The bf16 lm_head is read from a pre-quantisation backup: --bf16-shard, else
<mapped-shard>.bak, else <mapped-shard>.bak-orig (what
quant_heads_multishard.py leaves behind).
"""
import json, os, shutil, sys, time

DRAFTER = os.path.join(os.environ.get("SYV_REPO", "."), "drafter")
sys.path.insert(0, DRAFTER)
import numpy as np, torch
from safetensors import safe_open
from safetensors.torch import save_file
from compressed_tensors.compressors.pack_quantized.base import pack_to_int32
from gptq_utils import accumulate_hessian, gptq_quantize, rtn_quantize, dequant

S = sys.argv[1].rstrip("/") + "/"
D = sys.argv[2].rstrip("/") + "/"
BITS = int(sys.argv[sys.argv.index("--bits") + 1]) if "--bits" in sys.argv else 4
NCAL = int(sys.argv[sys.argv.index("--calib-rows") + 1]) if "--calib-rows" in sys.argv else 400000
MSE = "--mse-clip" in sys.argv
BF16 = sys.argv[sys.argv.index("--bf16-shard") + 1] if "--bf16-shard" in sys.argv else None
DATA = os.path.join(DRAFTER, "data")
GROUP = 128
dev = "cuda"

idx = json.load(open(S + "model.safetensors.index.json"))
wm = idx["weight_map"]
shard = wm["lm_head.weight_packed"]
src_bf16 = BF16 or next((S + shard + s for s in (".bak", ".bak-orig")
                         if os.path.exists(S + shard + s)), None)
if not src_bf16:
    sys.exit(f"no bf16 lm_head backup found for {shard}; pass --bf16-shard")
with safe_open(src_bf16, "pt") as f:
    W = f.get_tensor("lm_head.weight").to(dev)      # bf16 [V,K]
V, K = W.shape

hid = np.load(f"{DATA}/hidden.npy", mmap_mode="r")
T = hid.shape[0]
rng = np.random.default_rng(0)
rows = np.sort(rng.choice(T, size=min(NCAL + 20000, T), replace=False))
cal, held = rows[:NCAL], rows[NCAL:]
H = torch.zeros(K, K, device=dev)
n_seen = 0
t0 = time.time()
for a in range(0, len(cal), 32768):
    x = torch.from_numpy(np.array(hid[cal[a:a + 32768]])).view(torch.bfloat16).to(dev)
    H, n_seen = accumulate_hessian(H, x, n_seen)
print(f"hessian from {n_seen} rows in {time.time()-t0:.0f}s")
xh = torch.from_numpy(np.array(hid[held])).view(torch.bfloat16).to(dev)


def kl_eval(Wq):
    tot = 0.0
    n = 0
    for a in range(0, xh.shape[0], 1024):
        x = xh[a:a + 1024]
        lp = torch.log_softmax((x @ W.t()).float(), -1)
        lq = torch.log_softmax((x @ Wq.to(torch.bfloat16).t()).float(), -1)
        tot += (lp.exp() * (lp - lq)).sum().item()
        n += x.shape[0]
    return tot / n


q_rtn, s_rtn = rtn_quantize(W, BITS, GROUP)
print(f"RTN int{BITS}: KL {kl_eval(dequant(q_rtn, s_rtn)):.5f}")
del q_rtn, s_rtn
torch.cuda.empty_cache()
t0 = time.time()
qs, ss = [], []
for r0 in range(0, V, 16384):          # rows are independent under GPTQ; chunk to bound memory
    q_, s_ = gptq_quantize(W[r0:r0 + 16384], H, bits=BITS, group=GROUP, blocksize=GROUP, mse_clip=MSE)
    qs.append(q_.cpu()); ss.append(s_.cpu()); del q_, s_
    torch.cuda.empty_cache()
q, s = torch.cat(qs).to(dev), torch.cat(ss).to(dev)
dq = dequant(q, s)
print(f"GPTQ int{BITS}{' +mse' if MSE else ''}: KL {kl_eval(dq):.5f}  ({time.time()-t0:.0f}s)")
rel = ((dq - W.float()).norm() / W.float().norm()).item()
del dq
torch.cuda.empty_cache()
print(f"GPTQ round-trip rel error {rel:.4f}")

# ---- write variant dir ----
os.makedirs(D, exist_ok=True)
for f in os.listdir(S):
    if f.endswith(".safetensors") and f != shard and not f.startswith("model_extra") \
            and not os.path.exists(D + f):
        os.link(S + f, D + f)                      # body + model-nonquant shards
shutil.copy(S + "model_extra_tensors.safetensors", D + "model_extra_tensors.safetensors")
for f in ["mtp_draft_vocab_ids.pt", "mtp_draft_vocab_ids.json",
          "tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
          "generation_config.json", "preprocessor_config.json", "vocab.json",
          "merges.txt", "special_tokens_map.json", "README.md"]:
    if os.path.exists(S + f) and not os.path.exists(D + f):
        shutil.copy(S + f, D + f)
tensors = {}
with safe_open(S + shard, "pt") as f:
    meta = f.metadata()
    for k in f.keys():
        if k != "__metadata__":
            tensors[k] = f.get_tensor(k)
tensors["lm_head.weight_packed"] = pack_to_int32(q.cpu(), BITS, packed_dim=1).contiguous()
tensors["lm_head.weight_scale"] = s.to(torch.float16).cpu().contiguous()
tensors["lm_head.weight_shape"] = torch.tensor([V, K], dtype=torch.int64)
if os.path.exists(D + shard):
    os.remove(D + shard)
save_file(tensors, D + shard, metadata=meta or {"format": "pt"})
json.dump(idx, open(D + "model.safetensors.index.json", "w"), indent=2)
c = json.load(open(S + "config.json"))
g1 = c["quantization_config"]["config_groups"]["group_1"]
assert g1["targets"] == ["re:.*lm_head$"], g1["targets"]
g1["weights"]["num_bits"] = BITS
json.dump(c, open(D + "config.json", "w"), indent=2)
print("wrote", D, "(draft head still needs prepare/build_draft_vocab.py --ids)")
