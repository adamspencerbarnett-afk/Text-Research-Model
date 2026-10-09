"""Draw learning curves (validation bits per byte against training time) for chosen runs.

    python compiler/experiments/plot_curves.py OUT.png "label=results/file.json:config" ...

Each series is one run's validation curve. The y-axis is bits per original byte on the
validation slice (lower is better); the x-axis is training seconds. A table of the same
numbers follows on stdout, so the values are available without the picture.
"""
from __future__ import annotations

import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference palette (validated): categorical slots in fixed order, chrome in text tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"


def load(spec: str):
    label, rest = spec.split("=", 1)
    path, config = rest.rsplit(":", 1)
    r = json.load(open(path))
    run = r["configs"][config] if "configs" in r else r
    hist = run["training"]["history"]
    return label, [h["train_s"] for h in hist], [h["quick_bits_per_byte"] for h in hist], run


def main() -> int:
    out, specs = sys.argv[1], sys.argv[2:]
    series = [load(s) for s in specs]
    if len(series) > len(SERIES):
        raise SystemExit("at most 8 series: fold the rest or draw small multiples")
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(AXIS); ax.spines[spine].set_linewidth(1)
    ax.grid(axis="y", color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)
    for (label, xs, ys, _), color in zip(series, SERIES):
        ax.plot(xs, ys, color=color, linewidth=2, solid_joinstyle="round", solid_capstyle="round", label=label)
        ax.plot(xs[-1], ys[-1], "o", color=color, markersize=6, markeredgecolor=SURFACE, markeredgewidth=2)
    # direct end labels with leader lines, spread so they do not collide
    ends = sorted(((ys[-1], xs[-1], label) for label, xs, ys, _ in series), reverse=True)
    ymin, ymax = ax.get_ylim(); gap = (ymax - ymin) * 0.045
    placed = []
    for y, x, label in ends:
        ty = y
        if placed and placed[-1] - ty < gap:
            ty = placed[-1] - gap
        placed.append(ty)
        ax.annotate(f"{label}  {y:.3f}", xy=(x, y), xytext=(x + 12, ty), textcoords="data", fontsize=9, color=INK2,
                    va="center", arrowprops=dict(arrowstyle="-", color=AXIS, linewidth=0.8, shrinkA=0, shrinkB=3))
    ax.set_xlim(0, max(xs[-1] for _, xs, _, _ in series) * 1.38)
    ax.set_xlabel("training time, seconds (same 4-core CPU)", color=INK2, fontsize=10)
    ax.set_ylabel("validation bits per byte (lower is better)", color=INK2, fontsize=10)
    ax.set_title("Learning curves: the same 0.79M-parameter core on different encodings", color=INK, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE)
    print(f"wrote {out}")
    print("| series | " + " | ".join(f"{int(x)} s" for x in series[0][1]) + " |")
    for label, xs, ys, _ in series:
        print(f"| {label} | " + " | ".join(f"{y:.3f}" for y in ys) + " |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
