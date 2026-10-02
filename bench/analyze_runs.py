#!/usr/bin/env python3
"""Paired analysis of swift_ab.py suite legs.

Groups are sets of legs that measure the same configuration (for example the
four Swift 1.0 legs of an ABBA block). For each group the script reports
decode-rate dispersion, the speculative counters pooled over the group's
legs, and the decomposition

    decode tok/s = tokens per verify step x verify steps per second

where tokens per step comes from vLLM's own counters,
(accepted + steps) / steps, and steps per second is the pooled decode rate
divided by it.

Two groups can be compared prompt by prompt. The six suite prompts differ in
speed far more than any two builds do, so the comparison is the ratio of the
per-prompt mean decode rates, averaged over prompts. The bootstrap interval
resamples the repetitions within each prompt and, separately, the prompts
themselves. With six prompts the prompt-level interval is wide; both are
printed and neither should be read as more than it is.

Usage:
    python bench/analyze_runs.py \\
        --ladder bench/results/swift15-ladder.jsonl \\
        --suite-dir ~/git/infra/hosts/zeus/llm-bench/results/profiles \\
        --group g15=G-swift15fast-mtp-long \\
        --group g10=B2-swift10fast-mtp-long \\
        --compare g15:g10 --out summary.json

A --group value is a comma-separated list of leg labels. Each label needs
<suite-dir>/<label>-suite.json and a row with that label in a --ladder file
(for the speculative counters; no-speculation legs have none).
"""
import argparse
import json
import os
import random
import statistics


def quantiles(xs):
    q = statistics.quantiles(xs, n=4)
    return q[0], q[2]


def load_leg(suite_dir, label):
    path = os.path.join(os.path.expanduser(suite_dir), f"{label}-suite.json")
    return json.load(open(path))["runs"]


def pooled_rate(runs):
    tokens = sum(r["completion_tokens"] for r in runs)
    seconds = sum(r["total_s"] - r["ttft_s"] for r in runs)
    return tokens / seconds


def summarise_group(name, labels, suite_dir, ladder):
    runs = []
    for label in labels:
        runs += load_leg(suite_dir, label)
    rates = [r["decode_tps"] for r in runs]
    q1, q3 = quantiles(rates)
    out = {
        "legs": labels,
        "runs": len(runs),
        "decode_tps": {
            "median": statistics.median(rates),
            "mean": statistics.mean(rates),
            "q1": q1,
            "q3": q3,
            "min": min(rates),
            "max": max(rates),
        },
        "pooled_decode_tps": pooled_rate(runs),
        "ttft_s_median": statistics.median(r["ttft_s"] for r in runs),
    }
    counters = [ladder[l]["acceptance_suite"] for l in labels if ladder.get(l, {}).get("acceptance_suite")]
    if counters:
        accepted = sum(c["accepted"] for c in counters)
        drafted = sum(c["drafted"] for c in counters)
        steps = sum(c["steps"] for c in counters)
        tok_per_step = (accepted + steps) / steps
        out["speculation"] = {
            "accepted": accepted,
            "drafted": drafted,
            "steps": steps,
            "acceptance": accepted / drafted,
            "tokens_per_step": tok_per_step,
            "steps_per_s": out["pooled_decode_tps"] / tok_per_step,
            "ms_per_step": 1000 * tok_per_step / out["pooled_decode_tps"],
        }
    else:
        out["steps_per_s"] = out["pooled_decode_tps"]
        out["ms_per_step"] = 1000 / out["pooled_decode_tps"]
    out["by_prompt"] = {}
    for pid in sorted({r["id"] for r in runs}):
        xs = [r["decode_tps"] for r in runs if r["id"] == pid]
        out["by_prompt"][pid] = {"n": len(xs), "mean": statistics.mean(xs), "sd": statistics.stdev(xs) if len(xs) > 1 else 0.0}
    return out, runs


def paired(runs_a, runs_b, draws=5000, seed=1):
    """Ratio a/b of per-prompt mean decode rates, averaged over prompts."""
    ids = sorted({r["id"] for r in runs_a} & {r["id"] for r in runs_b})
    xa = {i: [r["decode_tps"] for r in runs_a if r["id"] == i] for i in ids}
    xb = {i: [r["decode_tps"] for r in runs_b if r["id"] == i] for i in ids}
    per_prompt = {i: statistics.mean(xa[i]) / statistics.mean(xb[i]) - 1 for i in ids}
    point = statistics.mean(per_prompt.values())
    rng = random.Random(seed)

    def one(resample_prompts):
        chosen = [rng.choice(ids) for _ in ids] if resample_prompts else ids
        diffs = []
        for i in chosen:
            ma = statistics.mean(rng.choices(xa[i], k=len(xa[i])))
            mb = statistics.mean(rng.choices(xb[i], k=len(xb[i])))
            diffs.append(ma / mb - 1)
        return statistics.mean(diffs)

    def interval(resample_prompts):
        d = sorted(one(resample_prompts) for _ in range(draws))
        return d[int(0.025 * draws)], d[int(0.975 * draws)]

    return {
        "prompts": len(ids),
        "per_prompt_ratio_minus_1": per_prompt,
        "prompts_a_slower": sum(1 for v in per_prompt.values() if v < -0.005),
        "prompts_level": sum(1 for v in per_prompt.values() if abs(v) <= 0.005),
        "prompts_a_faster": sum(1 for v in per_prompt.values() if v > 0.005),
        "mean_ratio_minus_1": point,
        "ci95_within_prompt_reps": interval(False),
        "ci95_resampling_prompts_too": interval(True),
        "draws": draws,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ladder", action="append", default=[], help="jsonl with per-leg speculative counters")
    ap.add_argument("--suite-dir", required=True)
    ap.add_argument("--group", action="append", required=True, metavar="NAME=LABEL[,LABEL...]")
    ap.add_argument("--compare", action="append", default=[], metavar="A:B", help="paired comparison of group A against group B")
    ap.add_argument("--out")
    args = ap.parse_args()

    ladder = {}
    for path in args.ladder:
        for line in open(os.path.expanduser(path)):
            row = json.loads(line)
            ladder[row["label"]] = row

    summary, runs = {"groups": {}, "comparisons": {}}, {}
    for spec in args.group:
        name, labels = spec.split("=", 1)
        summary["groups"][name], runs[name] = summarise_group(name, labels.split(","), args.suite_dir, ladder)
    for spec in args.compare:
        a, b = spec.split(":")
        summary["comparisons"][spec] = paired(runs[a], runs[b])

    for name, g in summary["groups"].items():
        d = g["decode_tps"]
        line = f"{name:12s} runs {g['runs']:3d}  median {d['median']:7.2f}  mean {d['mean']:7.2f}  IQR {d['q1']:6.1f}-{d['q3']:6.1f}  range {d['min']:.1f}-{d['max']:.1f}  pooled {g['pooled_decode_tps']:7.2f}"
        s = g.get("speculation")
        if s:
            line += f"  acc {s['acceptance']:.3f}  tok/step {s['tokens_per_step']:.3f}  steps/s {s['steps_per_s']:.2f} ({s['ms_per_step']:.2f} ms)"
        else:
            line += f"  steps/s {g['steps_per_s']:.2f} ({g['ms_per_step']:.2f} ms)"
        print(line)
    for spec, c in summary["comparisons"].items():
        lo, hi = c["ci95_within_prompt_reps"]
        plo, phi = c["ci95_resampling_prompts_too"]
        print(f"\n{spec}: mean per-prompt ratio {100 * c['mean_ratio_minus_1']:+.2f}% "
              f"(reps-only 95% {100 * lo:+.2f}..{100 * hi:+.2f}%, prompts too {100 * plo:+.2f}..{100 * phi:+.2f}%); "
              f"first slower on {c['prompts_a_slower']}, level on {c['prompts_level']}, faster on {c['prompts_a_faster']} of {c['prompts']} prompts")
        for pid, v in c["per_prompt_ratio_minus_1"].items():
            print(f"    {pid:20s} {100 * v:+6.2f}%")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(summary, f, indent=2)
            f.write("\n")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
