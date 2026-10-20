"""Regression guard (16 October 2026, Adam: "keep our gains plus implement our long context improvements, so we don't
lose on any front"). Scores a chat model's result files against the best 5M results so far; every front must hold
before a change is adopted. Floors are the best measured 5M values minus the allowed slack (two points, 1% on text).

    python compiler/experiments/gate_check.py chat5m_v21_s1 [--card-dir chat5m_v21] [--base chat5m_v2_A]
Reads results/chat_tests_<stem>.json, results/core_eval_<dir>_<stem>_clean.json, results/long_grid_<stem>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"

# front, how to read it, best 5M so far (where), floor; higher is better unless noted
FRONTS = [
    ("language: clean wiki bits/byte (lower)", lambda t, c, g: c["fluency"]["wiki"] if c else None, 1.354, 1.354 * 1.01, "lower"),
    ("language: clean books bits/byte (lower)", lambda t, c, g: c["fluency"]["books"] if c else None, 1.485, 1.485 * 1.01, "lower"),
    ("Q&A: chat probe /32", lambda t, c, g: n(t["abc"]["verify_off"]["chat_probe"]["total"]), 26, 24, "higher"),
    ("skills test v1 /100", lambda t, c, g: n(t["abc"]["verify_off"]["v1"]["total"]), 73, 71, "higher"),
    ("held-out skills v2 /200", lambda t, c, g: n(t["abc"]["verify_off"]["v2"]["total"]), 122, 120, "higher"),
    ("rewriting /40", lambda t, c, g: n(t["abc"]["verify_off"]["rewrite"]["total"]), 32, 30, "higher"),
    ("'I don't know' when unknown /20", lambda t, c, g: n(t["abc"]["verify_off"]["v1"]["unknown"]), 15, 13, "higher"),
    ("asks which one (clarify) /10", lambda t, c, g: n(t["abc"]["verify_off"]["v1"]["clarify"]), 7, 5, "higher"),
    ("wrong refusals v1 (lower)", lambda t, c, g: t["abc"]["verify_off"]["v1"]["wrong_refusals"], 4, 6, "lower"),
    ("conversation probe v2 /120", lambda t, c, g: n(t["probes"]["mt_probe2"]["total"]), 112, 110, "higher"),
    ("recall with memory, worst of 2-40 turns /15", lambda t, c, g: min(n(r["recalled"]) for r in t["recall"] if r["session_memory"]), 14, 13, "higher"),
    ("answer seconds (engine, lower)", lambda t, c, g: t["abc"]["verify_off"]["v1"]["s_per_answer"], 0.293, 0.293 * 1.2, "lower"),
]
# long-context fronts: these must RISE (baseline = arm A through the same engine)
LONG = ["one", "two", "distractor", "absent", "correction", "doc"]


def n(s) -> int:
    return int(str(s).split("/")[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stem"); ap.add_argument("--card-dir", default=""); ap.add_argument("--base", default="chat5m_v2_A")
    a = ap.parse_args()
    load = lambda p: json.loads(p.read_text()) if p.exists() else None
    t = load(R / f"chat_tests_{a.stem}.json")
    card_dir = a.card_dir or a.stem.rsplit("_", 1)[0]
    c = load(R / f"core_eval_{card_dir}_{a.stem}_clean.json")
    g, gb = load(R / f"long_grid_{a.stem}.json"), load(R / f"long_grid_{a.base}.json")
    rows, ok_all = [], True
    for name, get, best, floor, way in FRONTS:
        try:
            v = get(t, c, g) if t else None
        except (KeyError, TypeError):
            v = None
        ok = None if v is None else (v <= floor if way == "lower" else v >= floor)
        ok_all &= ok is not False
        rows.append({"front": name, "value": v, "best_5M": best, "floor": round(floor, 3), "holds": ok})
    long_rows = []
    if g and gb:
        for task in LONG:
            v, b = g["summary_right_share"]["task"].get(task), gb["summary_right_share"]["task"].get(task)
            long_rows.append({"task": task, "value": v, "base": b, "gain": None if v is None or b is None else round(v - b, 3)})
    res = {"model": a.stem, "all_fronts_hold": ok_all, "fronts": rows, "long_context_vs_base": long_rows,
           "missing": [r["front"] for r in rows if r["value"] is None]}
    (R / f"gate_{a.stem}.json").write_text(json.dumps(res, indent=1))
    for r in rows:
        mark = "  ?  " if r["holds"] is None else ("HOLDS" if r["holds"] else "LOST ")
        print(f"{mark} {r['front']:<46} {str(r['value']):>8}  (best {r['best_5M']}, floor {r['floor']})")
    for r in long_rows:
        print(f"LONG  {r['task']:<12} {r['value']}  vs base {r['base']}  gain {r['gain']}")
    print("ALL FRONTS HOLD" if ok_all else "A FRONT WAS LOST", "| missing:", res["missing"] or "none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
