#!/usr/bin/env python3
"""Create a fast-variant directory that differs from an existing one ONLY in
its draft vocabulary and draft-head rows.

Used for vocabulary ablations. Everything except the draft vocabulary is
shared with the source directory by hardlink (body shards, int4 lm_head,
embeddings, MTP weights, tokenizer, config), so a variant costs a few hundred
MB at most and cannot drift from the source in any other way.

The draft head is rebuilt the same way prepare/build_draft_vocab.py builds it:
the rows `ids` of the source directory's own int4 GPTQ `lm_head`, packed
words and scales alike. No row is ever copied from another model's head.

What changes in the destination:
  * mtp_draft_vocab_ids.pt           the new sorted id tensor (int64)
  * mtp.draft_lm_head.*              rebuilt in the shard that holds them
  * model.safetensors.index.json     rewritten copy (same mapping)

Run it where the model files are readable, which in practice means inside the
pinned syv image (the build phases leave root-owned 0600 files), e.g.

    docker run --rm -v ~/git/qwen38-27b-rtx3090:/repo -v <this repo>:/companion:ro \\
      --entrypoint /app/venv/bin/python <image> \\
      /companion/swift_qwen_syv/build_vocab_variant.py \\
        --src /repo/models/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast \\
        --dst /repo/models/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast-vocab-union \\
        --ids /companion/data/draft_vocab_ids/swift15_draft_vocab_ids.json \\
              /companion/data/draft_vocab_ids/swift_draft_vocab_ids.json --union

Without --union a single ids file is required and used as is. With --union
the sorted union of all given files is used. The tool prints the id count,
the intersection and the union so nothing is hard-coded.
"""
import argparse
import json
import os
import shutil
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REWRITE = {"model.safetensors.index.json", "mtp_draft_vocab_ids.pt"}
DRAFT_KEYS = ("mtp.draft_lm_head.weight_packed", "mtp.draft_lm_head.weight_scale", "mtp.draft_lm_head.weight_shape")


def load_ids(path):
    data = json.load(open(path))
    return sorted(set(data["ids"] if isinstance(data, dict) else data))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--ids", nargs="+", required=True)
    ap.add_argument("--union", action="store_true", help="use the union of all --ids files")
    args = ap.parse_args()

    src, dst = args.src.rstrip("/"), args.dst.rstrip("/")
    if os.path.exists(dst):
        sys.exit(f"refusing to overwrite existing {dst}")
    lists = [load_ids(p) for p in args.ids]
    if args.union:
        ids = sorted(set().union(*map(set, lists)))
        inter = set(lists[0]).intersection(*map(set, lists[1:]))
        print(f"union of {len(lists)} lists: {len(ids)} ids (intersection {len(inter)}; sizes {[len(l) for l in lists]})")
    else:
        if len(lists) != 1:
            sys.exit("give exactly one --ids file, or pass --union")
        ids = lists[0]
        print(f"{len(ids)} ids from {args.ids[0]}")

    index = json.load(open(f"{src}/model.safetensors.index.json"))
    wm = index["weight_map"]
    head_shard = wm["lm_head.weight_packed"]
    draft_shards = {wm[k] for k in DRAFT_KEYS}
    if len(draft_shards) != 1:
        sys.exit(f"draft head tensors are spread over several shards: {draft_shards}")
    draft_shard = draft_shards.pop()

    with safe_open(f"{src}/{head_shard}", framework="pt") as f:
        wp = f.get_tensor("lm_head.weight_packed")  # [vocab, K/8] int32
        ws = f.get_tensor("lm_head.weight_scale")  # [vocab, K/group]
        shape = f.get_tensor("lm_head.weight_shape")
    if ids[-1] >= wp.shape[0]:
        sys.exit(f"id {ids[-1]} is outside the {wp.shape[0]}-row lm_head")
    ids_t = torch.tensor(ids, dtype=torch.int64)
    sub_p = wp.index_select(0, ids_t).contiguous()
    sub_s = ws.index_select(0, ids_t).contiguous()
    sub_shape = torch.tensor([len(ids), int(shape[1])], dtype=torch.int64)
    print(f"draft head: packed {tuple(sub_p.shape)} {sub_p.dtype}, scales {tuple(sub_s.shape)} {sub_s.dtype}, "
          f"{(sub_p.numel() * sub_p.element_size() + sub_s.numel() * sub_s.element_size()) / 1e6:.0f} MB")

    tensors, meta = {}, None
    with safe_open(f"{src}/{draft_shard}", framework="pt") as f:
        meta = f.metadata()
        for k in f.keys():
            tensors[k] = f.get_tensor(k)
    tensors["mtp.draft_lm_head.weight_packed"] = sub_p
    tensors["mtp.draft_lm_head.weight_scale"] = sub_s
    tensors["mtp.draft_lm_head.weight_shape"] = sub_shape

    os.makedirs(dst)
    linked = copied = 0
    for name in sorted(os.listdir(src)):
        s, d = f"{src}/{name}", f"{dst}/{name}"
        if name in REWRITE or name == draft_shard or os.path.isdir(s):
            continue
        if name.endswith((".bak", ".bak-orig", ".bak-quant", ".bak-draft", ".bak-mtp")) or ".bak-" in name:
            continue  # rollback copies of the source build, not part of the model
        if name.endswith(".safetensors"):
            os.link(s, d)
            linked += 1
        else:
            shutil.copy2(s, d)
            copied += 1
    save_file(tensors, f"{dst}/{draft_shard}", metadata=meta or {"format": "pt"})
    json.dump(index, open(f"{dst}/model.safetensors.index.json", "w"), indent=2)
    torch.save(ids_t, f"{dst}/mtp_draft_vocab_ids.pt")
    print(f"{dst}: {linked} shards hardlinked, {copied} small files copied, rewrote {draft_shard}, index and ids")


if __name__ == "__main__":
    main()
