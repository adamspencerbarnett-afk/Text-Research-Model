"""The guard for chat22 (16 October: no front may be lost). Reads only results/suite_*.json.

For each front, the new model is compared with the BEST reference model on that front (the earlier 5M models: arm A
and v2.1 seeds by default). A share holds if it is not lower than the reference by more than 2 standard errors of the
difference (scoring.diff_holds); text bits/byte holds within +1%; latency within +20%; the cache must be identical on
every turn. A front the new model lacks FAILS (the old guard counted it as holding).

    python -m chat22.tests.gate22 v22_s1 [--refs chat5m_v2_A,chat5m_v21_s1,chat5m_v21_s2]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from chat22.tests import scoring as S

ROOT = Path(__file__).resolve().parents[2]


def fronts(r: dict) -> dict:
    """front name -> ("share", k, n) | ("lower", value) | ("exact", ok)"""
    f = {}
    get = lambda *ks: _dig(r, ks)
    for name in ("wiki", "books"):
        v = get("text", name)
        if v is not None:
            f[f"text {name} bits/byte"] = ("lower", v, 0.01)
    for sec in ("skills", "skills_stored"):
        for s in ("heldout_v2", "rewrite"):
            t = get(sec, s, "total")
            if t:
                f[f"{sec.replace('_', ' ')} {s}"] = ("share", t["k"], t["n"])
    for g in ("unknown", "clarify", "followup", "update", "nearmiss"):
        t = get("skills", "heldout_v2", "groups", g)
        if t:
            f[f"skills v2 {g}"] = ("share", t["k"], t["n"])
    for s in ("lookup", "not_lookup"):
        t = get("probe", s)
        if t:
            f[f"probe {s}"] = ("share", t["k"], t["n"])
    for s in ("generated", "real_phrasing"):
        t = get("convo", s, "total")
        if t:
            f[f"conversation {s}"] = ("share", t["k"], t["n"])
    for k, t in (get("long", "by_task") or {}).items():
        f[f"long {k}"] = ("share", t["k"], t["n"])
    for N, m in (get("memory") or {}).items():
        f[f"memory {N} turns recall"] = ("share", m["recall"]["k"], m["recall"]["n"])
        f[f"memory {N} turns never told"] = ("share", m["never_told_abstained"]["k"], m["never_told_abstained"]["n"])
    gr = get("grounding")
    if gr:
        f["grounding: right with memory on"] = ("share", gr["right"], gr["n"])
        f["grounding: invented share (lower)"] = ("lower_abs", gr["invented_share"], 0.03)
        f["grounding: copied a wrong memory (lower)"] = ("lower_abs", gr["copied_wrong_share"], 0.03)
    t = get("story", "questions")
    if t:
        f["story questions"] = ("share", t["k"], t["n"])
    v = get("story", "contradictions_per_scene")
    used = get("story", "character_facts_used")
    if v is not None and used and used["n"] >= 5:     # a model that never names a character cannot pass by saying nothing
        f["story contradictions per scene"] = ("lower_abs", v, 0.1)
    if used:
        f["story character facts used"] = ("share", used["k"], used["n"])
    v = get("latency", "first_code_ms_p95")
    if v is not None:
        f["latency first code p95 ms"] = ("lower", v, 0.2)
    c = get("cache")
    if c:
        f["cache identical"] = ("exact", c["identical_turns"] == c["turns"])
    return f


def _dig(d, ks):
    for k in ks:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--refs", default="chat5m_v2_A,chat5m_v21_s1,chat5m_v21_s2")
    a = ap.parse_args()
    load = lambda s: json.loads((ROOT / "results" / f"suite_{s}.json").read_text()) if (ROOT / "results" / f"suite_{s}.json").exists() else None
    new = load(a.model)
    if new is None:
        print("no suite result for", a.model); return 1
    refs = {s: fronts(r) for s in a.refs.split(",") if (r := load(s))}
    fn = fronts(new)
    names = sorted(set(fn) | set().union(*[set(v) for v in refs.values()]))
    rows, all_ok = [], True
    for name in names:
        mine = fn.get(name)
        cands = [(s, v[name]) for s, v in refs.items() if name in v]
        if mine is None:
            rows.append((name, "MISSING", None, None)); all_ok = False; continue
        if not cands:
            shown = f"{mine[1]}/{mine[2]}" if mine[0] == "share" else mine[1]
            rows.append((name, "new", shown, None)); continue
        kind = mine[0]
        if kind == "share":
            best = max(cands, key=lambda c: c[1][1] / max(1, c[1][2]))
            ok = S.diff_holds(mine[1], mine[2], best[1][1], best[1][2])
            rows.append((name, "HOLDS" if ok else "LOST", f"{mine[1]}/{mine[2]}", f"{best[1][1]}/{best[1][2]} ({best[0]})"))
        elif kind in ("lower", "lower_abs"):
            best = min(cands, key=lambda c: c[1][1])
            lim = best[1][1] * (1 + mine[2]) if kind == "lower" else best[1][1] + mine[2]
            ok = mine[1] <= lim
            rows.append((name, "HOLDS" if ok else "LOST", mine[1], f"{best[1][1]} ({best[0]})"))
        else:
            ok = mine[1]
            rows.append((name, "HOLDS" if ok else "LOST", mine[1], "must be identical"))
        all_ok &= ok
    for name, verdict, v, ref in rows:
        print(f"{verdict:8s} {name:42s} {str(v):>12s}   ref {ref}")
    print("ALL FRONTS HOLD" if all_ok else "A FRONT WAS LOST OR MISSING")
    (ROOT / "results" / f"gate22_{a.model}.json").write_text(json.dumps({"model": a.model, "refs": list(refs), "all_hold": all_ok,
                                                                       "rows": [dict(zip(("front", "verdict", "value", "ref"), r)) for r in rows]}, indent=1))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
