#!/usr/bin/env python3
"""Generate the benchmark charts as standalone SVGs (stdlib only).

Outputs to docs/charts/:
  bench-ladder.svg    decode tok/s + speculative acceptance, five legs
  vocab-coverage.svg  held-out draft-vocab coverage vs counted tokens
  gptq-kl.svg         lm_head KL to the bf16 head, RTN vs GPTQ

Numbers are transcribed from the measured result files (see
docs/benchmarks.md and the fast-ladder/vocab-report provenance noted
below); regenerating after a re-measure means editing this file, which is
the point.

Style: opaque white card, dark text, works on GitHub light/dark and on
veloxide.dev's light/dark themes alike.
"""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "charts")
os.makedirs(OUT, exist_ok=True)

FONT = "ui-sans-serif, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
INK = "#1f2937"        # primary text
MUTED = "#6b7280"      # secondary text
GRID = "#e5e7eb"
BG = "#ffffff"
BLUE = "#2563eb"       # Swift-fast
LIGHT_BLUE = "#93c5fd" # Swift int8
GREY = "#9ca3af"       # Qwen reference

def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def txt(x, y, s, size=13, fill=INK, anchor="start", weight=400):
    return (f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" '
            f'fill="{fill}" text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

def rect(x, y, w, h, fill, rx=3):
    return f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" rx="{rx}" fill="{fill}"/>'

def line(x1, y1, x2, y2, stroke, w=1, dash=""):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{stroke}" stroke-width="{w}"{d}/>'

def svg_open(w, h):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" '
            f'viewBox="0 0 {w} {h}" role="img">')

def card(w, h):
    return f'<rect width="{w}" height="{h}" rx="10" fill="{BG}"/>'

# ---------------------------------------------------------------- bench ----
# fast-ladder.jsonl (2026-09-15, REPS=4 medians, same night, RTX 3090)
BENCH = [
    # (label, decode tok/s, acceptance, tok/step, colour, group)
    ("Swift int8 heads",        94.0, 0.630, 2.89, LIGHT_BLUE, "MTP · 114k context"),
    ("Swift-fast (this build)", 98.4, 0.660, 2.98, BLUE,       "MTP · 114k context"),
    ("Qwen fast (upstream)",    98.2, 0.634, 2.90, GREY,       "MTP · 114k context"),
    ("Swift-fast (this build)", 156.4, 0.385, 3.70, BLUE,      "DFlash2 · 46k context"),
    ("Qwen fast (upstream)",    167.4, 0.387, 3.71, GREY,      "DFlash2 · 46k context"),
]

def bench_chart():
    W, H = 880, 470
    L, R = 220, 190
    T, B = 64, 40
    plot_w = W - L - R
    vmax = 180.0
    row_h = (H - T - B) / len(BENCH)
    bar_h = 26
    parts = [svg_open(W, H), card(W, H)]
    parts.append(txt(24, 34, "Decode throughput and speculative acceptance", 17, INK, weight=600))
    parts.append(txt(24, 52, "RTX 3090, four repetitions, medians, same night", 12, MUTED))
    for gv in (0.0, 45.0, 90.0, 135.0, 180.0):
        x = L + plot_w * gv / vmax
        parts.append(line(x, T - 6, x, H - B, GRID))
        parts.append(txt(x, H - B + 18, f"{gv:.0f}", 11, MUTED, "middle"))
    parts.append(txt(L + plot_w / 2, H - 10, "decode tok/s", 11, MUTED, "middle"))
    group_y = None
    for i, (label, dec, acc, tps, col, group) in enumerate(BENCH):
        y = T + i * row_h + (row_h - bar_h) / 2
        if group != group_y and group_y is not None:
            gy = T + i * row_h
            parts.append(line(L - 12, gy, W - 16, gy, GRID))
        group_y = group
        parts.append(txt(L - 12, y + bar_h / 2 + 4, label, 13, INK, "end"))
        bw = plot_w * dec / vmax
        parts.append(rect(L, y, bw, bar_h, col))
        parts.append(txt(L + bw + 8, y + bar_h / 2 + 0.5, f"{dec:.1f}", 13, INK, "start", 600))
        parts.append(txt(L + bw + 46, y + bar_h / 2 + 0.5, f"acc {acc:.3f}", 12, MUTED))
        parts.append(txt(L + bw + 112, y + bar_h / 2 + 0.5, f"{tps:.2f} tok/step", 12, MUTED))
    parts.append(txt(W - 16, T - 14, BENCH[0][5].upper(), 10, MUTED, "end", 600))
    parts.append(txt(W - 16, T + 3 * row_h - 14, BENCH[3][5].upper(), 10, MUTED, "end", 600))
    parts.append("</svg>")
    open(os.path.join(OUT, "bench-ladder.svg"), "w").write("\n".join(parts))

# --------------------------------------------------------------- vocab ----
# swift_vocab_report.json: convergence over the counting split, plus the
# base-Qwen list's held-out coverage (flat).
CONV = [(946179, 0.9917), (1901468, 0.9958), (2852357, 0.9973), (3811120, 0.9981)]
BASE = 0.9669

def vocab_chart():
    W, H = 880, 430
    L, R = 78, 200
    T, B = 64, 52
    plot_w, plot_h = W - L - R, H - T - B
    ymin, ymax = 96.0, 100.0
    xmin, xmax = 0.9e6, 3.9e6

    def X(t): return L + plot_w * (t - xmin) / (xmax - xmin)
    def Y(v): return T + plot_h * (1 - (v * 100 - ymin) / (ymax - ymin))

    parts = [svg_open(W, H), card(W, H)]
    parts.append(txt(24, 34, "Held-out draft-vocabulary coverage", 17, INK, weight=600))
    parts.append(txt(24, 52, "share of held-out Swift output tokens the 40k id list can draft", 12, MUTED))
    for v in (0.96, 0.97, 0.98, 0.99, 1.00):
        y = Y(v)
        parts.append(line(L, y, W - R, y, GRID))
        parts.append(txt(L - 10, y + 4, f"{v*100:.0f}%", 11, MUTED, "end"))
    for t in (1e6, 2e6, 3e6):
        x = X(t)
        parts.append(txt(x, H - B + 18, f"{t/1e6:.0f}M", 11, MUTED, "middle"))
    parts.append(txt(L + plot_w / 2, H - 12, "tokens counted into the vocabulary", 11, MUTED, "middle"))
    # base list: flat
    parts.append(line(L, Y(BASE), W - R, Y(BASE), GREY, 2, "6 5"))
    parts.append(txt(W - R + 12, Y(BASE) + 4, f"base-Qwen 40k list  {BASE*100:.1f}%", 12, "#6b7280"))
    # swift list: rising
    pts = [(X(t), Y(v)) for t, v in CONV]
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(f'<path d="{path}" fill="none" stroke="{BLUE}" stroke-width="2.5"/>')
    for x, y in pts:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{BLUE}"/>')
    parts.append(txt(W - R + 12, Y(CONV[-1][1]) + 4, f"Swift-derived  {CONV[-1][1]*100:.2f}%", 12, BLUE, weight=600))
    parts.append(txt(X(CONV[0][0]) - 6, Y(CONV[0][1]) - 10, f"{CONV[0][1]*100:.2f}%", 11, MUTED, "end"))
    parts.append(txt(L + 4, T + 14, "25,879 ids (not 40,960): Swift emits fewer distinct tokens", 11, MUTED))
    parts.append("</svg>")
    open(os.path.join(OUT, "vocab-coverage.svg"), "w").write("\n".join(parts))

# ----------------------------------------------------------------- gptq ----
# drafter/runs/gptq.log on this machine; the shipped-fast-variant figure is
# upstream's own documented number for the base model.
KL = [
    ("round-to-nearest int4", 0.00707, GREY),
    ("GPTQ, Swift's own Hessian", 0.00234, BLUE),
]
SHIPPED = 0.0029  # upstream's shipped fast variant (base Qwen)

def gptq_chart():
    W, H = 880, 360
    L, R = 60, 150
    T, B = 64, 46
    plot_w = W - L - R
    vmax = 0.008
    bar_h = 40
    parts = [svg_open(W, H), card(W, H)]
    parts.append(txt(24, 34, "lm_head quantisation quality", 17, INK, weight=600))
    parts.append(txt(24, 52, "KL divergence to the bf16 head on held-out states, lower is better", 12, MUTED))
    for v in (0.000, 0.002, 0.004, 0.006, 0.008):
        x = L + plot_w * v / vmax
        parts.append(line(x, T - 6, x, H - B, GRID))
        parts.append(txt(x, H - B + 18, f"{v:.3f}", 11, MUTED, "middle"))
    parts.append(txt(L + plot_w / 2, H - 10, "KL(bf16 head || int4 head)", 11, MUTED, "middle"))
    xs = L + plot_w * SHIPPED / vmax
    parts.append(line(xs, T - 6, xs, H - B, "#374151", 1.5, "5 4"))
    parts.append(txt(xs + 6, T + 6, f"upstream's shipped fast variant  {SHIPPED}", 11, MUTED))
    for i, (label, v, col) in enumerate(KL):
        y = T + 34 + i * (bar_h + 46)
        bw = plot_w * v / vmax
        parts.append(rect(L, y, bw, bar_h, col))
        parts.append(txt(L, y - 8, label, 13, INK))
        parts.append(txt(L + bw + 10, y + bar_h / 2 + 5, f"{v:.5f}", 13, INK, "start", 600))
    parts.append("</svg>")
    open(os.path.join(OUT, "gptq-kl.svg"), "w").write("\n".join(parts))

if __name__ == "__main__":
    bench_chart(); vocab_chart(); gptq_chart()
    for f in sorted(os.listdir(OUT)):
        print(os.path.join(OUT, f), os.path.getsize(os.path.join(OUT, f)), "bytes")
