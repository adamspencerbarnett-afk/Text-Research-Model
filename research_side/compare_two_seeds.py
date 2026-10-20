"""Compare candidate models with the current best (recipes/BEST.json), front by front, pooling seeds; 2 standard errors.
    python research_side/compare_two_seeds.py CAND_s1 [CAND_s2 ...] [--best v24a_s1,v24a_s2]
Reads results/suite_<name>_ev2full.json. Prints every front, marks significant wins (+) and losses (-), and the verdict
under the rule: a new best must not lose any front beyond noise and must win more fronts than it loses."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from chat22.tests.gate22 import fronts  # noqa: E402

SKIP = ("latency", "cache", "probe lookup")     # probe lookup asks about A's Alpaca store: store-bound, not a model front


def load(n):
    p = ROOT / "results" / f"suite_{n}_ev2full.json"
    return fronts(json.loads(p.read_text())) if p.exists() else None


def pool(xs):
    a = xs[0]
    if a[0] == "share":
        return ("share", sum(x[1] for x in xs), sum(x[2] for x in xs))
    if a[0].startswith("lower"):
        return (a[0], sum(x[1] for x in xs) / len(xs), a[2] if len(a) > 2 else 0)
    return a


def sc(x):
    return x[1] / max(1, x[2]) if x[0] == "share" else (-x[1] if x[0].startswith("lower") else float(x[1]))


def fmt(x):
    return f"{x[1]}/{x[2]}" if x[0] == "share" else f"{x[1]:.3f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cands", nargs="+"); ap.add_argument("--best", default=None)
    a = ap.parse_args()
    best = (a.best or ",".join(json.loads((ROOT / "recipes" / "BEST.json").read_text())["best"]["seeds_confirmed"])).split(",")
    B = [f for f in (load(n) for n in best) if f]; C = [f for f in (load(n) for n in a.cands) if f]
    if not B or not C:
        print("missing suite results"); return 1
    wins = losses = 0; rows = []
    for k in sorted(set.intersection(*[set(f) for f in B + C])):
        if any(s in k for s in SKIP):
            continue
        pb, pc = pool([f[k] for f in B]), pool([f[k] for f in C]); d = sc(pc) - sc(pb)
        if pb[0] == "share":
            se = math.sqrt(max(sc(pb) * (1 - sc(pb)), 1e-4) / max(1, pb[2]) + max(sc(pc) * (1 - sc(pc)), 1e-4) / max(1, pc[2])); sig = abs(d) > 2 * se
        else:
            sig = abs(d) > 0.02 * max(1, abs(sc(pb)))
        wins += sig and d > 0; losses += sig and d < 0
        rows.append(f"{k[:38]:38}{fmt(pb):>12}{fmt(pc):>12}{('  +' if d > 0 else '  -') if sig else ''}")
    print(f"{'front':38}{'best':>12}{'candidate':>12}   best = {best}  candidate = {a.cands}")
    print("\n".join(rows))
    verdict = "NEW BEST (no front lost, more wins)" if losses == 0 and wins > 0 else "not a new best"
    print(f"significant wins {wins}, losses {losses}  ->  {verdict}" + ("" if len(C) > 1 else "  (one seed: confirm with a second)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
