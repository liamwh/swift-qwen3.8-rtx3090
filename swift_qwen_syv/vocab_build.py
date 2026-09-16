#!/usr/bin/env python3
"""Build a target-specific draft-vocab id list from the model's OWN outputs
and measure held-out coverage against a baseline list.

The single highest-leverage artefact in this pipeline. The MTP drafter scores
a small slice of lm_head (prepare/build_draft_vocab.py --ids); a token outside
that slice can never be drafted, so every miss is a guaranteed rejection and
truncates the speculation chain. Reusing a draft vocab counted over a
*different* model's (or workload's) outputs leaves that coverage on the table.

Reads gen_data.py output ({"id","src","think","prompt_ids","output_ids",...}).
Splits sequences 90/10 (deterministic), counts output-token frequencies on the
90%, builds the N-row id list (same specials handling as
prepare/build_draft_vocab.py --ids mode: specials forced in, everything else
by frequency), and reports:

  - coverage of BOTH lists on the held-out 10% (all tokens, and code-ish
    sources only, if your corpus uses src tags like rust/ts/debug/edit/agent)
  - convergence: held-out coverage of lists built from the first
    25/50/75/100% of the train split (is more generation still buying
    coverage? — our run levelled off within ~3k sequences)

Note: the list may legitimately be shorter than N when the corpus contains
fewer distinct output tokens (do not "fix" this by padding); the verifier
checks self-consistency, not an arbitrary row count.

Usage:
  python vocab_build.py --gen gen.jsonl --tokenizer <model-dir>
      --out-ids draft_vocab_ids.json --out-report vocab_report.json
      [--n 40960] [--base-ids baseline_ids.json] [--seed 20260915]
"""
import argparse, collections, json, os, random, sys

CODE_SRC = {"rust", "ts", "debug", "edit", "agent"}  # corpus src tags to treat as "code"
SPECIALS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<think>", "</think>",
            "<tool_call>", "</tool_call>", "<tool_response>", "</tool_response>"]


def main():
    ap = argparse.ArgumentParser(description="target-output draft-vocab builder")
    ap.add_argument("--gen", required=True, help="gen_data.py output JSONL")
    ap.add_argument("--tokenizer", required=True, help="model dir (or HF id) for special-token ids")
    ap.add_argument("--out-ids", required=True, help="output id list JSON (the --ids file)")
    ap.add_argument("--out-report", required=True, help="output coverage report JSON")
    ap.add_argument("--n", type=int, default=40960, help="draft vocab size")
    ap.add_argument("--base-ids", default=None,
                    help="baseline id list to compare held-out coverage against")
    ap.add_argument("--seed", type=int, default=20260915)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    special_ids = set()
    for name in SPECIALS:
        tid = tok.convert_tokens_to_ids(name)
        if isinstance(tid, int) and tid >= 0:
            special_ids.add(tid)
    special_ids |= set(tok.all_special_ids)

    recs = [json.loads(l) for l in open(a.gen)]
    R = random.Random(a.seed)
    R.shuffle(recs)
    cut = int(len(recs) * 0.9)
    train_recs, held_recs = recs[:cut], recs[cut:]
    N = a.n

    def count(recs, frac=1.0):
        c = collections.Counter()
        n = 0
        lim = int(len(recs) * frac)
        for r in recs[:lim]:
            c.update(r["output_ids"])
            n += len(r["output_ids"])
        return c, n

    def build_list(counter):
        top = [t for t, _ in counter.most_common() if t not in special_ids][: N - len(special_ids)]
        return sorted(set(top) | special_ids)

    def coverage(ids, recs, only_src=None):
        s = set(ids)
        tot = inn = 0
        for r in recs:
            if only_src and r.get("src") not in only_src:
                continue
            for t in r["output_ids"]:
                tot += 1
                if t in s:
                    inn += 1
        return (inn / tot if tot else None), tot

    full_counter, ntok = count(train_recs)
    own_ids = build_list(full_counter)

    rep = {"gen_seqs": len(recs), "train_seqs": len(train_recs), "held_seqs": len(held_recs),
           "train_tokens": ntok, "n_requested": N, "n_built": len(own_ids)}

    lists = [("own", own_ids)]
    if a.base_ids:
        base_ids = sorted(set(json.load(open(a.base_ids))))
        rep.update({"overlap": len(set(own_ids) & set(base_ids)),
                    "own_only": len(set(own_ids) - set(base_ids)),
                    "base_only": len(set(base_ids) - set(own_ids))})
        lists.append(("baseline", base_ids))

    for name, ids in lists:
        cov_all, tot = coverage(ids, held_recs)
        cov_code, tot_c = coverage(ids, held_recs, CODE_SRC)
        rep[f"{name}_heldout_coverage_all"] = round(cov_all, 4) if cov_all is not None else None
        rep[f"{name}_heldout_coverage_code"] = round(cov_code, 4) if cov_code is not None else None
        rep[f"{name}_heldout_tokens"] = tot

    rep["convergence"] = []
    for frac in (0.25, 0.5, 0.75, 1.0):
        c, n = count(train_recs, frac)
        ids = build_list(c)
        cov, _ = coverage(ids, held_recs)
        top_n = {t for t, _ in c.most_common() if t not in special_ids}  # rank-stability vs full
        full_top = {t for t, _ in full_counter.most_common() if t not in special_ids}
        rep["convergence"].append({
            "frac": frac, "counted_tokens": n,
            "heldout_coverage": round(cov, 4) if cov is not None else None,
            "top_n_jaccard_vs_full": round(len(top_n & full_top) / len(top_n | full_top), 4)
                                     if top_n or full_top else None,
        })

    json.dump(own_ids, open(a.out_ids, "w"))
    json.dump(rep, open(a.out_report, "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
