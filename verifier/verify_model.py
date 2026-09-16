#!/usr/bin/env python3
"""Structural verifier for syv-prepared third-party Qwen3.8-family model
dirs (int8-heads layout) and their fast variants (int4-GPTQ heads + target
draft vocab).

Why this exists: the syv stack's verify.sh asserts an int8 lm_head
(`num_bits == 8`), so it falsely rejects the VALID int4-GPTQ fast-variant
layout its own drafter/ pipeline produces. This verifier accepts either
layout and validates SELF-CONSISTENCY instead of one prescribed geometry:
the bit widths declared in config.json's quantization groups, the packed
tensor / scale / shape math implied by those bits, and the draft head's rows
against the ids artefact that build_draft_vocab.py consumed.

It exists because the failure mode it catches is dangerous and quiet: a
hardlinked shard still carrying superseded int8 MTP packed tensors passes
every "does the mapped tensor exist" check, boots, and collides with the
int4 tensors at load time. The duplicate-tensor scan below catches exactly
that class before anything is served.

Checks (exit 1 on any FAIL; warnings do not fail):
  config/architecture self-consistency (layers, interval pattern, untied
  embeddings, vocab/hidden), quantisation groups (targets, bit widths per
  layout, symmetry, group size, zero-point rules), packed tensor geometry
  for lm_head + embed + the 8 MTP linears + the draft head (shape-checked
  against bf16 geometry x bits), draft ids artefact (type, range, rows ==
  draft head rows; provenance cross-check when --ids-source is given),
  index/shard completeness, a whole-directory duplicate-tensor scan, and —
  only with --runtime, inside the serving environment — vLLM version, syv
  patch markers, and reasoning/tool parser registries.

Machine-specific concerns (systemd, API keys, ports) are deliberately NOT
checked here; upstream verify.sh keeps the key-presence check.

Usage:
  python verify_model.py <model-dir> [--expect {int4,int8,auto}]
      [--mtp-geometry geometry.json] [--ids-source ids.json] [--runtime]

mtp_geometry.json (optional): {"mtp.fc": [out, in], ...}. Without it the
built-in Qwen3.8-27B table is used; with --mtp-geometry infer, geometry is
read from the weight_shape tensors themselves (pure self-consistency).
"""
import argparse, json, os, struct, sys

GROUP = 128
MTP_LINEARS = ["mtp.fc", "mtp.layers.0.mlp.down_proj", "mtp.layers.0.mlp.gate_proj",
               "mtp.layers.0.mlp.up_proj", "mtp.layers.0.self_attn.q_proj",
               "mtp.layers.0.self_attn.k_proj", "mtp.layers.0.self_attn.v_proj",
               "mtp.layers.0.self_attn.o_proj"]
# bf16 (out, in) geometry of the Qwen3.8-27B MTP module — the only
# model-specific default; override with --mtp-geometry for other family members.
DEFAULT_MTP_GEOMETRY = {
    "mtp.fc": (5120, 10240),
    "mtp.layers.0.mlp.down_proj": (5120, 17408),
    "mtp.layers.0.mlp.gate_proj": (17408, 5120),
    "mtp.layers.0.mlp.up_proj": (17408, 5120),
    "mtp.layers.0.self_attn.q_proj": (12288, 5120),
    "mtp.layers.0.self_attn.k_proj": (1024, 5120),
    "mtp.layers.0.self_attn.v_proj": (1024, 5120),
    "mtp.layers.0.self_attn.o_proj": (5120, 6144),
}

FAILS = 0


def ok(m):
    print(f"  PASS  {m}")


def warn(m):
    print(f"  WARN  {m}")


def fail(m):
    global FAILS
    print(f"  FAIL  {m}")
    FAILS += 1


def read_header(path):
    """safetensors header: u64 LE length + JSON. No deps."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def header_keys(path):
    return {k: v for k, v in read_header(path).items() if k != "__metadata__"}


def main():
    ap = argparse.ArgumentParser(description="structural verifier for syv-prepared third-party model dirs")
    ap.add_argument("dir")
    ap.add_argument("--expect", choices=["int4", "int8", "auto"], default="auto",
                    help="required head layout (default: auto-detect from the lm_head group)")
    ap.add_argument("--mtp-geometry", default=None,
                    help="JSON {name: [out, in]} or 'infer' to read shapes from weight_shape tensors")
    ap.add_argument("--ids-source", default=None,
                    help="expected draft-id list (e.g. your vocab_build.py output); cross-checks provenance")
    ap.add_argument("--runtime", action="store_true",
                    help="also check the runtime this runs inside (vLLM version, syv patches, parsers)")
    a = ap.parse_args()
    D = a.dir.rstrip("/") + "/"

    print(f"== dir {D}")
    if not os.path.isdir(D):
        print("  FAIL  model dir not found")
        sys.exit(1)

    # ---- config / architecture (self-consistency, not fixed values) ----
    print("== config / architecture")
    try:
        c = json.load(open(D + "config.json"))
    except Exception as e:
        fail(f"config.json unreadable: {e}")
        sys.exit(1)
    tc = c.get("text_config", c)
    if c.get("architectures"):
        ok(f"architectures {c['architectures']}")
    else:
        fail("architectures empty")
    if tc.get("hidden_size") and tc.get("vocab_size"):
        VOCAB, HIDDEN = int(tc["vocab_size"]), int(tc["hidden_size"])
        ok(f"hidden_size {HIDDEN}, vocab_size {VOCAB}")
    else:
        fail("hidden_size/vocab_size missing from config")
        sys.exit(1)
    lt = tc.get("layer_types") or []
    nh = tc.get("num_hidden_layers")
    if lt and nh:
        (ok if len(lt) == nh else fail)(f"layer_types ({len(lt)}) == num_hidden_layers ({nh})")
    elif lt or nh:
        warn("only one of layer_types/num_hidden_layers present; count not cross-checked")
    else:
        fail("no layer count information in config")
    iv = tc.get("full_attention_interval")
    if iv and lt:
        pat_ok = all((t == "full_attention") == ((i % iv) == iv - 1) for i, t in enumerate(lt))
        (ok if pat_ok else fail)(f"full_attention_interval {iv} matches the layer_types pattern")
    if tc.get("tie_word_embeddings") is False:
        ok("untied embeddings (separate lm_head present)")
    elif tc.get("tie_word_embeddings") is True:
        fail("tied embeddings: no separate lm_head to quantize")
    else:
        warn("tie_word_embeddings not stated")

    # ---- quantisation groups ----
    print("== quantization_config groups")
    qc = c.get("quantization_config", {})
    fmt = qc.get("format") or qc.get("quant_method")
    if fmt:
        (ok if fmt == "compressed-tensors" else fail)(f"quant format {fmt}")
    else:
        warn("quantization_config carries no format/quant_method")
    groups = qc.get("config_groups", {})

    def gbits(g):
        g = groups.get(g)
        return g["weights"]["num_bits"] if g else None

    lm_bits = gbits("group_1")
    EXPECT = a.expect
    if EXPECT == "auto":
        EXPECT = "int4" if lm_bits == 4 else "int8"
    if lm_bits not in (4, 8):
        fail(f"group_1 (lm_head) num_bits {lm_bits} not in (4, 8)")
    elif EXPECT == "int4" and lm_bits != 4:
        fail(f"--expect int4 but group_1 num_bits is {lm_bits}")
    elif EXPECT == "int8" and lm_bits != 8:
        fail(f"--expect int8 but group_1 num_bits is {lm_bits}")
    else:
        ok(f"layout: {EXPECT} (lm_head group_1 bits={lm_bits})")

    def want(g, bits, targets, sym=True):
        gg = groups.get(g)
        if not gg:
            fail(f"{g} missing")
            return None
        w = gg["weights"]
        if gg.get("targets") != targets:
            fail(f"{g} targets {gg.get('targets')} != {targets}")
        else:
            ok(f"{g} targets {targets}")
        if w.get("num_bits") != bits:
            fail(f"{g} num_bits {w.get('num_bits')} != {bits}")
        else:
            ok(f"{g} num_bits {bits}")
        if w.get("symmetric") is not sym:
            fail(f"{g} symmetric != {sym}")
        else:
            ok(f"{g} symmetric={sym}")
        if w.get("group_size") != GROUP:
            fail(f"{g} group_size != {GROUP}")
        else:
            ok(f"{g} group_size {GROUP}")
        if w.get("zp_dtype") is not None and sym:
            fail(f"{g} zp_dtype set on symmetric tensors")
        return w.get("num_bits")

    lm_b = want("group_1", {"int8": 8, "int4": 4}[EXPECT], ["re:.*lm_head$"]) or {"int8": 8, "int4": 4}[EXPECT]
    emb_b = want("group_2", 8, ["re:.*embed_tokens$"]) or 8
    mtp_b = want("group_3", {"int8": 8, "int4": 4}[EXPECT], ["re:^mtp\\..*"]) or {"int8": 8, "int4": 4}[EXPECT]
    g0 = groups.get("group_0")
    if g0 and g0["weights"].get("num_bits") == 4:
        ok(f"group_0 body 4-bit (symmetric={g0['weights'].get('symmetric')}, AWQ asym ok)")
    else:
        fail("group_0 body not 4-bit")

    # ---- index + shard files ----
    print("== index / shards")
    try:
        idx = json.load(open(D + "model.safetensors.index.json"))
        wm = idx["weight_map"]
    except Exception as e:
        fail(f"index unreadable: {e}")
        sys.exit(1)
    shards = sorted(set(wm.values()))
    for s in shards:
        if not os.path.exists(D + s):
            fail(f"mapped shard missing: {s}")
    ok(f"{len(shards)} mapped shards present")

    files = [f for f in os.listdir(D) if f.endswith(".safetensors") and ".bak" not in f and ".tmp" not in f]
    try:
        keys_per_file = {f: set(header_keys(D + f)) for f in files}
    except Exception as e:
        fail(f"safetensors header unreadable: {e}")
        sys.exit(1)

    # ---- duplicate / stale tensor scan ----
    # Every index-mapped name must live in EXACTLY ONE file, and it must be
    # the mapped one. This is the stale-int8-MTP collision class: a hardlinked
    # shard still carrying superseded packed tensors that vLLM (which reads
    # every key of an opened shard) loads on top of the real ones.
    print("== duplicate / stale tensor scan")
    dup = 0
    for name, mapped in wm.items():
        holders = [f for f, ks in keys_per_file.items() if name in ks]
        if len(holders) > 1 or (holders and holders[0] != mapped):
            fail(f"{name}: present in {holders}, index maps {mapped}")
            dup += 1
    if not dup:
        ok(f"all {len(wm)} index tensors exist in exactly their mapped shard")
    stale_warned = 0
    for f in shards:
        for k in keys_per_file.get(f, set()) - set(wm):
            if k.endswith((".weight_packed", ".weight_scale", ".weight_shape")):
                warn(f"unmapped packed tensor {k} left in served shard {f} (stale quant residue)")
                stale_warned += 1
    if not stale_warned:
        ok("no unmapped packed tensors in served shards")

    # ---- packed tensor shapes ----
    print("== packed tensor shapes")
    hdr_cache = {}

    def tensor_header(name):
        for f in files:
            if name in keys_per_file.get(f, ()):  # noqa: B023
                if f not in hdr_cache:
                    hdr_cache[f] = header_keys(D + f)
                return hdr_cache[f].get(name)
        return None

    def check_quant_tensor(name, out_f, in_f, bits):
        for suf, shape in (
            ("weight_packed", (out_f, in_f * bits // 32)),
            ("weight_scale", (out_f, in_f // GROUP)),
            ("weight_shape", (2,)),
        ):
            k = f"{name}.{suf}"
            h = tensor_header(k)
            if h is None:
                fail(f"{k} missing from all shards")
                continue
            actual = tuple(h["shape"])
            if actual == shape:
                ok(f"{k} {list(actual)}")
            else:
                fail(f"{k} {list(actual)} != {list(shape)}")

    if "lm_head.weight_packed" not in wm:
        fail("lm_head.weight_packed not in index")
    else:
        check_quant_tensor("lm_head", VOCAB, HIDDEN, lm_b)
    emb_key = next((k for k in wm if k.endswith("embed_tokens.weight_packed")), None)
    if not emb_key:
        fail("embed_tokens.weight_packed not in index")
    else:
        check_quant_tensor(emb_key[: -len(".weight_packed")], VOCAB, HIDDEN, emb_b)

    if a.mtp_geometry == "infer":
        mtp_geometry = {}
        for m in MTP_LINEARS:
            h = tensor_header(f"{m}.weight_shape")
            mtp_geometry[m] = tuple(h["shape"]) if h else None
        ok("MTP geometry inferred from weight_shape tensors")
    elif a.mtp_geometry:
        mtp_geometry = {k: tuple(v) for k, v in json.load(open(a.mtp_geometry)).items()}
        ok(f"MTP geometry from {a.mtp_geometry}")
    else:
        mtp_geometry = DEFAULT_MTP_GEOMETRY
        ok("MTP geometry: built-in Qwen3.8-27B table (pass --mtp-geometry for other family members)")
    for m in MTP_LINEARS:
        if f"{m}.weight_packed" not in wm:
            fail(f"{m}.weight_packed not in index")
        elif mtp_geometry.get(m):
            o, i = mtp_geometry[m]
            check_quant_tensor(m, o, i, mtp_b)

    # ---- draft head + vocab ----
    print("== draft head / vocab artefacts")
    ids_path_pt = D + "mtp_draft_vocab_ids.pt"
    ids_path_json = D + "mtp_draft_vocab_ids.json"
    ids = None
    if os.path.exists(ids_path_pt):
        try:
            import torch  # optional dependency; only needed for .pt artefacts
            t = torch.load(ids_path_pt, map_location="cpu")
            if not isinstance(t, torch.Tensor) or t.dtype != torch.int64:
                fail(f"draft ids wrong type {type(t)}")
            else:
                ids = t.tolist()
        except ImportError:
            warn("torch unavailable; .pt draft ids not type-checked (install torch or use a .json artefact)")
    elif os.path.exists(ids_path_json):
        ids = json.load(open(ids_path_json))
        (ok if all(isinstance(i, int) for i in ids) else fail)("draft ids (json) all ints")
    else:
        fail("mtp_draft_vocab_ids.{pt,json} missing")
    N_DRAFT = None
    if ids is not None:
        if not 0 < len(ids) <= 65536:
            fail(f"draft ids count {len(ids)} outside (0, 65536]")
        elif max(ids) >= VOCAB or min(ids) < 0:
            fail("draft ids out of vocab range")
        elif len(ids) != len(set(ids)):
            fail("draft ids contain duplicates")
        else:
            N_DRAFT = len(ids)
            cap = "" if len(ids) <= 40960 else " (above the ~40,960 coverage plateau; still self-consistent)"
            ok(f"draft ids: {N_DRAFT:,} unique ints in range{cap}")
        if a.ids_source and os.path.exists(a.ids_source):
            want_ids = sorted(set(json.load(open(a.ids_source))))
            if sorted(set(ids)) == want_ids:
                ok(f"draft ids match source list {a.ids_source}")
            else:
                fail("draft ids do not match the expected source list")
        elif a.ids_source:
            warn(f"--ids-source {a.ids_source} not found; provenance not cross-checked")
    if "mtp.draft_lm_head.weight_packed" not in wm:
        fail("mtp.draft_lm_head.weight_packed not in index")
    elif N_DRAFT is None:
        fail("draft head rows unverifiable (ids artefact failed above)")
    else:
        check_quant_tensor("mtp.draft_lm_head", N_DRAFT, HIDDEN, {"int4": 4, "int8": 8}[EXPECT])

    # ---- tokenizer (structural; upstream verify.sh does the encode test) ----
    print("== tokenizer")
    tok_json = os.path.exists(D + "tokenizer.json")
    tok_cfg = os.path.exists(D + "tokenizer_config.json")
    if tok_json and tok_cfg:
        ok("tokenizer.json + tokenizer_config.json present")
        try:
            body = open(D + "tokenizer.json").read()
            (ok if "<think>" in body else warn)("tokenizer.json contains <think>")
        except Exception as e:
            warn(f"tokenizer.json unreadable: {e}")
    else:
        fail(f"tokenizer files missing (json={tok_json}, config={tok_cfg}) — a dir without "
             f"tokenizer.json hands transformers a 1-token vocabulary and the server dies far "
             f"downstream on 'failed to tokenize reasoning strings'")

    # ---- runtime / patches / parsers (only when asked; we run inside the image) ----
    if a.runtime:
        print("== runtime / syv patches / parsers")
        try:
            import vllm
            ver = vllm.__version__
            (ok if ver.startswith("0.28") else warn)(f"vllm {ver}"
                                                     + ("" if ver.startswith("0.28") else " (patches written against 0.28.x)"))
        except Exception as e:
            fail(f"vllm import: {e}")
        import glob as _glob
        mtp_py = _glob.glob(os.path.join(os.path.dirname(__import__("vllm").__file__),
                                         "model_executor", "models", "qwen3_5_mtp.py"))
        if mtp_py and "draft_lm_head" in open(mtp_py[0]).read():
            ok("qwen3_5-mtp-draft-vocab patch present (draft_lm_head in qwen3_5_mtp.py)")
        elif mtp_py:
            fail("mtp draft-vocab patch missing (drafter needs it to score the id slice)")
        else:
            warn("qwen3_5_mtp.py not found at the default path; patch presence not checked")
        try:
            from vllm.reasoning import ReasoningParserManager
            ReasoningParserManager.get_reasoning_parser("qwen3")
            ok("reasoning parser qwen3 resolves")
        except Exception as e:
            fail(f"reasoning parser qwen3 unavailable: {e}")
        try:
            from vllm.tool_parsers import ToolParserManager
            ToolParserManager.get_tool_parser("qwen3_coder")
            ok("tool parser qwen3_coder resolves")
        except Exception as e:
            fail(f"tool parser qwen3_coder unavailable: {e}")

    print()
    if FAILS:
        print(f"verify_model: {FAILS} FAILURE(S)")
        sys.exit(1)
    print("verify_model: OK")


if __name__ == "__main__":
    main()
