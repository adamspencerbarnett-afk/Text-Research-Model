"""Two figures for the one-time side-by-side: held-out bits per byte at 600 s for the
traditional models against ours (bars), and the learning curves of the runs that share the
validation split (lines).

    python compiler/experiments/plot_side_by_side.py OUT_DIR
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (validated). Emphasis form: ours in blue, the traditional models in gray.
BLUE, ORANGE, AQUA, GRAYBAR = "#2a78d6", "#eb6834", "#1baf7a", "#c3c2b7"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def style(ax):
    ax.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def main() -> int:
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    l1 = json.load(open("results/ladder1.json"))["configs"]
    l2 = json.load(open("results/ladder2a.json"))["configs"]
    sb = json.load(open("results/side_by_side.json"))["configs"]
    rows = [  # label, run, ours?
        ("raw bytes\n(no tokenizer)", l1["bytes"], False),
        ("standard BPE 8k\n(traditional tokenizer)", sb["bpe_8k"], False),
        ("compiler v0 8k\n(words and pieces)", l1["v0_8k_plain"], True),
        ("compiler + hashed\ninput tables", l2["v0_8k_table20_tri"], True),
        ("compiler + exact\nn-gram tables", sb["v0_8k_ng"], True),
        ("compiler + both\nkinds of table", sb["v0_8k_table20_tri_ng"], True),
    ]
    # ---- figure 1: bars
    fig, ax = plt.subplots(figsize=(9, 4.6), dpi=150); fig.patch.set_facecolor(SURFACE); style(ax)
    vals = [r["eval"]["heldout_overall"]["bits_per_byte"] for _, r, _ in rows]
    colors = [BLUE if ours else GRAYBAR for _, _, ours in rows]
    bars = ax.bar(range(len(rows)), vals, color=colors, width=0.55)
    for b in bars:  # 4px-ish rounded data end is not native to matplotlib bars; keep thin bars and a clean baseline
        pass
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([l for l, _, _ in rows], fontsize=8.5, color=INK2)
    ax.set_ylim(1.6, 2.3); ax.set_yticks([1.6, 1.8, 2.0, 2.2]); ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    for i, (v, (_, r, _)) in enumerate(zip(vals, rows)):
        p = r["model"]["params"]["total"]
        ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=9.5, color=INK)
        ax.text(i, 1.615, f"{p/1e6:.1f}M params" if p < 1e8 else f"{p/1e6:.0f}M params", ha="center", va="bottom", fontsize=7.5, color=MUTED)
    ax.set_ylabel("held-out bits per byte (lower is better)", color=INK2, fontsize=10)
    ax.set_title("Same 0.79M-parameter core, same text, 600 s of CPU training: traditional (gray) against ours (blue)",
                 color=INK, fontsize=10.5, loc="left")
    fig.tight_layout(); fig.savefig(out / "side_by_side_bars.png", facecolor=SURFACE)

    # ---- figure 2: curves of the runs that share the validation split
    fig, ax = plt.subplots(figsize=(9, 5), dpi=150); fig.patch.set_facecolor(SURFACE); style(ax)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    series = [("standard BPE 8k (traditional)", sb["bpe_8k"], GRAYBAR), ("compiler + exact n-gram tables", sb["v0_8k_ng"], BLUE),
              ("compiler + both kinds of table", sb["v0_8k_table20_tri_ng"], ORANGE)]
    for label, r, c in series:
        hist = r["training"]["history"]; xs = [h["train_s"] for h in hist]; ys = [h["quick_bits_per_byte"] for h in hist]
        ax.plot(xs, ys, color=c if c != GRAYBAR else "#898781", linewidth=2, label=label, solid_capstyle="round")
        ax.plot(xs[-1], ys[-1], "o", color=c if c != GRAYBAR else "#898781", markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
        ax.annotate(f"{ys[-1]:.3f}", xy=(xs[-1], ys[-1]), xytext=(xs[-1] + 10, ys[-1]), fontsize=9, color=INK2, va="center")
    ax.set_xlim(0, 700); ax.set_xlabel("training time, seconds (same 4-core CPU)", color=INK2, fontsize=10)
    ax.set_ylabel("validation bits per byte (lower is better)", color=INK2, fontsize=10)
    ax.set_title("Learning curves on the shared validation split (2% of the training stream, never trained on)", color=INK, fontsize=10.5, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout(); fig.savefig(out / "side_by_side_curves.png", facecolor=SURFACE)
    print("wrote", out / "side_by_side_bars.png", out / "side_by_side_curves.png")
    print("| model | params | held-out bits/byte | vs BPE | out-of-domain | talk-back B/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    bpe = sb["bpe_8k"]["eval"]["heldout_overall"]["bits_per_byte"]
    for label, r, _ in rows:
        h = r["eval"]["heldout_overall"]["bits_per_byte"]
        print(f"| {label.replace(chr(10), ' ')} | {r['model']['params']['total']/1e6:.1f}M | {h:.3f} | {100*(h-bpe)/bpe:+.1f}% | {r['eval']['ood_overall']['bits_per_byte']:.2f} | {r['generation']['bytes_per_s']:.0f} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
