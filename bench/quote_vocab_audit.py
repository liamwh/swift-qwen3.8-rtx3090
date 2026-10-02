#!/usr/bin/env python3
"""Audit the quote workload against draft-vocabulary lists.

The quote benchmark asks the model to reproduce a synthetic ~8,000-token
document verbatim. A token that is not in the draft vocabulary can never be
drafted, so every occurrence is a guaranteed rejection and ends the
speculation chain. This script tokenises the exact benchmark document
(bench/swift_ab.py build_doc(8000)) and reports, per list, how many document
tokens fall outside it and which ids those are.

Usage (needs the `tokenizers` package; `uv run --with tokenizers` works):

    uv run --with tokenizers python bench/quote_vocab_audit.py \\
        --tokenizer ~/git/qwen38-27b-rtx3090/models/Swift-1.5-Qwen3.8-27B-W4A16-syv-fast/tokenizer.json \\
        --list base-Qwen=~/git/qwen38-27b-rtx3090/prepare/draft_vocab_ids.json \\
        --list swift-1.0=data/draft_vocab_ids/swift_draft_vocab_ids.json \\
        --list swift-1.5=data/draft_vocab_ids/swift15_draft_vocab_ids.json \\
        --out bench/results/quote-vocab-audit.json

Optional --ladder FILE.jsonl (repeatable) prints each leg's quote-workload
speculative counters next to the list results so the two can be compared.
"""
import argparse
import collections
import hashlib
import importlib.util
import json
import os
import sys


def load_build_doc():
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("swift_ab", os.path.join(here, "swift_ab.py"))
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    return mod.build_doc


def load_ids(path):
    raw = open(os.path.expanduser(path), "rb").read()
    data = json.loads(raw)
    ids = data["ids"] if isinstance(data, dict) else data
    return set(ids), hashlib.sha256(raw).hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--list", action="append", required=True, metavar="NAME=PATH")
    ap.add_argument("--target-tokens", type=int, default=8000)
    ap.add_argument("--top", type=int, default=8, help="most frequent missing ids to show per list")
    ap.add_argument("--ladder", action="append", default=[], help="jsonl with acceptance_quote per leg")
    ap.add_argument("--out")
    args = ap.parse_args()

    from tokenizers import Tokenizer

    doc = load_build_doc()(args.target_tokens)
    tk = Tokenizer.from_file(os.path.expanduser(args.tokenizer))
    # The harness compares text.strip() to doc.strip(), so count the stripped document.
    ids = tk.encode(doc.strip()).ids
    result = {
        "target_tokens": args.target_tokens,
        "doc_sha1_12": hashlib.sha1(doc.encode()).hexdigest()[:12],
        "doc_tokens": len(ids),
        "tokenizer_sha256": hashlib.sha256(open(os.path.expanduser(args.tokenizer), "rb").read()).hexdigest(),
        "lists": {},
    }
    print(f"document {result['doc_sha1_12']}: {len(ids)} tokens")
    sets = {}
    for spec in args.list:
        name, path = spec.split("=", 1)
        s, digest = load_ids(path)
        sets[name] = s
        missing = collections.Counter(i for i in ids if i not in s)
        entry = {
            "list_ids": len(s),
            "list_sha256": digest,
            "missing_tokens": sum(missing.values()),
            "missing_fraction": sum(missing.values()) / len(ids),
            "distinct_missing_ids": len(missing),
            "top_missing": [
                {"id": i, "text": tk.decode([i]), "count": c, "in_other_lists": [n for n, o in sets.items() if n != name and i in o]}
                for i, c in missing.most_common(args.top)
            ],
        }
        result["lists"][name] = entry
        print(f"{name:14s} {len(s):6d} ids  misses {entry['missing_tokens']:4d} of {len(ids)} ({100 * entry['missing_fraction']:.2f}%)"
              f"  distinct {len(missing)}")
        for t in entry["top_missing"]:
            print(f"    {t['count']:4d} x {t['text']!r} (id {t['id']})")
    # Cross-list membership for the ids that matter, now that every list is loaded.
    for name, entry in result["lists"].items():
        for t in entry["top_missing"]:
            t["in_other_lists"] = [n for n, o in sets.items() if n != name and t["id"] in o]
    if args.ladder:
        result["quote_counters"] = {}
        print("\nquote-workload counters (accepted / drafted / steps):")
        for path in args.ladder:
            for line in open(os.path.expanduser(path)):
                row = json.loads(line)
                q = row.get("acceptance_quote")
                if q:
                    key = f"{os.path.basename(path)}:{row['label']}"
                    result["quote_counters"][key] = {k: q[k] for k in ("accepted", "drafted", "steps")}
                    print(f"  {key:50s} {q['accepted']:6d} / {q['drafted']:6d} / {q['steps']:5d}  rate {q['acceptance_rate']:.4f}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
            f.write("\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    sys.exit(main())
