"""Tiny stdlib safetensors writer + synthetic model-dir fixtures for the
verifier selftest. Zero dependencies: shapes are what the verifier reads, so
tensor payloads can be zero bytes of the right dtype/size.

The 'corrupt' variants reproduce the real failure classes this verifier was
built around, at toy scale:
  - stale-collision: a served shard carrying a superseded duplicate packed
    tensor (the historical hardlinked-int8-MTP-in-model-nonquant bug)
  - bits-mismatch: config declares int4 but tensors carry int8 geometry
  - draft-rows: draft head rows != draft ids count
  - dup-ids: duplicate ids in the draft vocab artefact
"""
import json
import os
import shutil
import struct

I32, F16, I64 = 4, 2, 8
ITEMSIZE = {"I32": 4, "F16": 2, "I64": 8, "BF16": 2, "F32": 4, "I8": 1, "U8": 1}

VOCAB, HIDDEN, GROUP = 1000, 256, 128
N_DRAFT = 50
MTP_TINY = {
    "mtp.fc": (256, 512),
    "mtp.layers.0.mlp.down_proj": (256, 512),
    "mtp.layers.0.mlp.gate_proj": (512, 256),
    "mtp.layers.0.mlp.up_proj": (512, 256),
    "mtp.layers.0.self_attn.q_proj": (384, 256),
    "mtp.layers.0.self_attn.k_proj": (64, 256),
    "mtp.layers.0.self_attn.v_proj": (64, 256),
    "mtp.layers.0.self_attn.o_proj": (256, 320),
}


def write_safetensors(path, tensors):
    """tensors: {name: (dtype, shape)} — payload is zeros."""
    header = {"__metadata__": {"format": "pt"}}
    offset = 0
    blobs = []
    for name, (dtype, shape) in tensors.items():
        n = 1
        for d in shape:
            n *= d
        size = n * ITEMSIZE[dtype]
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [offset, offset + size]}
        blobs.append(b"\x00" * size)
        offset += size
    hj = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def quant_triples(name, out_f, in_f, bits):
    return {
        f"{name}.weight_packed": ("I32", [out_f, in_f * bits // 32]),
        f"{name}.weight_scale": ("F16", [out_f, in_f // GROUP]),
        f"{name}.weight_shape": ("I64", [2]),
    }


def build_dir(root, lm_bits=4, mtp_bits=4, stale_collision=False, bits_mismatch=False,
              draft_rows=None, dup_ids=False, drop_shard=False, json_ids=True):
    """Assemble a toy model dir. bits_mismatch builds int8 tensors for a config
    that declares int4 (for the lm_head). draft_rows overrides the draft head's
    row count. Returns the dir path."""
    D = os.path.join(root, "model")
    os.makedirs(D, exist_ok=True)
    w = {}  # filename -> {tensor: (dtype, shape)}

    emb_bits = 8
    shard_a = "model-00001-of-00002.safetensors"
    shard_b = "model-00002-of-00002.safetensors"
    extras = "model_extra_tensors.safetensors"
    nonquant = "model-nonquant.safetensors"
    w[shard_a] = quant_triples("model.language_model.embed_tokens", VOCAB, HIDDEN, emb_bits)
    lm_build_bits = 8 if bits_mismatch else lm_bits
    w[shard_b] = quant_triples("lm_head", VOCAB, HIDDEN, lm_build_bits)
    w[extras] = {}
    for m, (o, i) in MTP_TINY.items():
        w[extras].update(quant_triples(m, o, i, mtp_bits))
    rows = N_DRAFT if draft_rows is None else draft_rows
    w[extras].update(quant_triples("mtp.draft_lm_head", rows, HIDDEN, lm_bits))
    w[nonquant] = {"model.language_model.norm.weight": ("F16", [HIDDEN])}
    if stale_collision:
        # the historical bug: model-nonquant (hardlinked from the int8 dir)
        # still carries the superseded int8 mtp.fc packed tensors
        w[nonquant].update(quant_triples("mtp.fc", *MTP_TINY["mtp.fc"], 8))

    wm = {}
    for fname, tensors in w.items():
        for t in tensors:
            wm[t] = fname
        write_safetensors(os.path.join(D, fname), tensors)
    if drop_shard:
        os.remove(os.path.join(D, shard_b))
        wm = {k: v for k, v in wm.items() if v != shard_b}

    ids = sorted(set(range(100, 100 + N_DRAFT + (1 if dup_ids else 0))))
    if dup_ids:
        ids = sorted(ids[: N_DRAFT - 1] + [ids[0]])  # N_DRAFT entries, one duplicated
    if json_ids:
        json.dump(ids, open(os.path.join(D, "mtp_draft_vocab_ids.json"), "w"))
    else:  # minimal .pt: torch.save layout is versioned zip; emulate via pickle protocol
        import pickle
        with open(os.path.join(D, "mtp_draft_vocab_ids.pt"), "wb") as f:
            pickle.dump({"ids": ids}, f)  # NOT a real torch artefact; only for load-failure paths

    json.dump({"metadata": {"total_size": 0}, "weight_map": wm},
              open(os.path.join(D, "model.safetensors.index.json"), "w"))

    lt = ["linear_attention"] * 3 + ["full_attention"]
    groups = {
        "group_0": {"targets": "Linear", "weights": {"num_bits": 4, "symmetric": False, "group_size": GROUP}},
        "group_1": {"targets": ["re:.*lm_head$"], "weights": {"num_bits": lm_bits, "symmetric": True, "group_size": GROUP, "zp_dtype": None}},
        "group_2": {"targets": ["re:.*embed_tokens$"], "weights": {"num_bits": emb_bits, "symmetric": True, "group_size": GROUP, "zp_dtype": None}},
        "group_3": {"targets": ["re:^mtp\\..*"], "weights": {"num_bits": mtp_bits, "symmetric": True, "group_size": GROUP, "zp_dtype": None}},
    }
    cfg = {
        "architectures": ["Qwen3_5ForConditionalGeneration"],
        "model_type": "qwen3_5",
        "text_config": {"hidden_size": HIDDEN, "vocab_size": VOCAB, "num_hidden_layers": 4,
                        "layer_types": lt * 1, "full_attention_interval": 4,
                        "tie_word_embeddings": False},
        "quantization_config": {"format": "compressed-tensors", "config_groups": groups, "ignore": []},
    }
    json.dump(cfg, open(os.path.join(D, "config.json"), "w"), indent=1)
    json.dump({"tokenizer": True}, open(os.path.join(D, "tokenizer.json"), "w"))
    json.dump({"think": "<think>"}, open(os.path.join(D, "tokenizer_config.json"), "w"))

    geo = os.path.join(root, "geometry.json")
    json.dump({k: list(v) for k, v in MTP_TINY.items()}, open(geo, "w"))
    return D, geo


def clean(root):
    shutil.rmtree(root, ignore_errors=True)
