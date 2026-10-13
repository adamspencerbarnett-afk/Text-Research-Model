"""Print a ladder result file as Markdown tables: the equal-time view, the equal-bytes view
(validation bits per byte read off each run's curve at the same number of training bytes),
per-file bits per byte, hash fidelity and the talk-back samples.

    python compiler/experiments/summarize_ladder.py results/ladder1.json [--at-mb 2.0]
"""
from __future__ import annotations

import argparse
import json

import numpy as np


def at_bytes(history: list[dict], target: float) -> float | None:
    xs = np.array([h["bytes_seen"] for h in history], dtype=float)
    ys = np.array([h["quick_bits_per_byte"] for h in history], dtype=float)
    if len(xs) < 2 or target > xs[-1] or target < xs[0]:
        return None
    return float(np.interp(target, xs, ys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path"); ap.add_argument("--at-mb", type=float, default=None)
    a = ap.parse_args()
    r = json.load(open(a.path))
    cfgs = r["configs"]
    print(f"Budget {r['budget_s']:.0f} s per run, seed {r['seed']}, {r['corpus']['train_books']} training books "
          f"({r['corpus']['train_bytes']/1e6:.1f} MB), held-out {', '.join(r['corpus']['heldout'])}.\n")
    print("| Encoding | Params | IDs/KB | Held-out bits/byte | OOD bits/byte | MB seen | Steps | Train IDs/s | Gen bytes/s |")
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for name, c in cfgs.items():
        h, o, t, g = c["eval"]["heldout_overall"], c["eval"]["ood_overall"], c["training"], c["generation"]
        print(f"| {name} | {c['model']['params']['total']/1e6:.2f}M | {h['ids_per_kb']:.1f} | {h['bits_per_byte']:.3f} | "
              f"{o['bits_per_byte']:.3f} | {t['bytes_seen']/1e6:.2f} | {t['steps']} | {t['tokens_per_s']:.0f} | {g['bytes_per_s']:.0f} |")
    # equal-bytes view: the largest byte count every run reached
    common = min(c["training"]["bytes_seen"] for c in cfgs.values())
    target = (a.at_mb * 1e6) if a.at_mb else common
    print(f"\nEqual-bytes view: validation bits/byte (150 KB slice of the first held-out book) at {target/1e6:.2f} MB of training text, interpolated on each run's curve.\n")
    print("| Encoding | bits/byte at equal bytes | seconds to reach it |")
    print("| --- | ---: | ---: |")
    for name, c in cfgs.items():
        hist = c["training"]["history"]
        v = at_bytes(hist, target)
        secs = None
        if v is not None:
            xs = [h["bytes_seen"] for h in hist]; ts = [h["train_s"] for h in hist]
            secs = float(np.interp(target, xs, ts))
        print(f"| {name} | {v:.3f} | {secs:.0f} |" if v is not None else f"| {name} | not reached | |")
    print("\nPer-file held-out bits/byte:\n")
    files = list(next(iter(cfgs.values()))["eval"]["heldout"].keys())
    print("| Encoding | " + " | ".join(files) + " |"); print("| --- |" + " ---: |" * len(files))
    for name, c in cfgs.items():
        print(f"| {name} | " + " | ".join(f"{c['eval']['heldout'][f]['bits_per_byte']:.3f}" for f in files) + " |")
    for name, c in cfgs.items():
        fid = [(f, row["fidelity"]) for f, row in c["eval"]["heldout"].items() if "fidelity" in row]
        if fid:
            print(f"\nHash fidelity for {name} (share of held-out bytes that decode back exactly):")
            for f, x in fid:
                print(f"- {f}: exact {100*x['exact_share']:.2f}%, unseen {100*x['unseen_share']:.2f}%, collided {100*x['collided_share']:.3f}%")
            print(f"- training units: {c['encoding']['distinct_units']:,} distinct, {c['encoding']['units_lost']} lost to collisions "
                  f"({100*c['encoding']['unit_mass_lost']:.3f}% of unit occurrences)")
    print("\nTalk-back samples (prompt \"We went to the park\", 200 tokens, temperature 0.8, top-k 40):\n")
    for name, c in cfgs.items():
        g = c["generation"]
        sample = g["sample"].replace("\n", " ")[:300]
        print(f"- **{name}** ({g['bytes']} bytes, {g['bytes_per_s']:.0f} B/s): {sample}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
