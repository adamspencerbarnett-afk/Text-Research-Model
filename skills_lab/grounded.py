"""Grounded scorecard (15 October 2026, Adam): where each answer came from, and whether it made things up.

    .venv/Scripts/python.exe -B skills_lab/grounded.py MODEL.pt --name NAME [--mode k1|k3close] [--device cuda]
        [--key-head K.pt | --dense 0]

For every answer to the skills tests (v1, held-out v2) and the 32-question chat probe:
  support   share of the answer's content words found in what memory returned (the shown exchanges) or in the
            question itself; an answer with support >= 0.8 is GROUNDED, below 0.5 it is the network's own text
  correct   the test's keywords (abstaining counts as correct only where nothing was stored)
and the four outcomes:
  right from memory    correct and grounded            (the design working)
  right from network   correct, not grounded           (general knowledge in the weights)
  wrong from memory    wrong, grounded                 (copied the wrong evidence: retrieval or selection failed)
  made up              wrong, not grounded             (invented)
plus, on questions whose answer is NOT stored, how often the model invented one instead of saying so.
Writes skills_lab/results/grounded_<name>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
import torch  # noqa: E402

import decision as D  # noqa: E402
import engine as E  # noqa: E402
import eval_skills as V  # noqa: E402
import testset as T  # noqa: E402

m2_qa = E.m2_qa


def support(ans: bytes, q: bytes, evidence) -> float:
    aw = D.cwords(ans)
    if not aw:
        return 1.0
    pool = set(D.cwords(q))
    for eq, ea in evidence:
        pool |= set(D.cwords(eq)) | set(D.cwords(ea))
    return sum(w in pool for w in aw) / len(aw)


def score_set(eng, items, k, policy):
    rows = []
    for group, q, kw, extra in items:
        out, info = eng.ask_k(q.encode(), k=k, policy=policy)
        ev = [eng.mem.exchanges[i] for i in info.get("idx", [])]
        text = out.decode("utf-8", "replace")
        sup = support(out, q.encode(), ev)
        abst = V.abstains(text)
        if group == "unknown":
            correct = abst
        elif group == "clarify":
            correct = "?" in text and V.has(text, kw)
        else:
            correct = V.has(text, kw) if kw else None
        rows.append({"group": group, "question": q, "answer": text, "support": round(sup, 3), "abstained": abst, "correct": correct,
                     "stored": group not in ("unknown",)})
    return rows


def summarise(rows):
    s = {"answers": len(rows)}
    graded = [r for r in rows if r["correct"] is not None and not r["abstained"]]
    g = lambda r: r["support"] >= 0.8
    s["grounded_share"] = round(sum(g(r) for r in rows if not r["abstained"]) / max(1, sum(not r["abstained"] for r in rows)), 3)
    s["right_from_memory"] = sum(r["correct"] and g(r) for r in graded)
    s["right_from_network"] = sum(r["correct"] and not g(r) for r in graded)
    s["wrong_from_memory"] = sum((not r["correct"]) and g(r) for r in graded)
    s["made_up"] = sum((not r["correct"]) and r["support"] < 0.8 for r in graded)
    unk = [r for r in rows if r["group"] == "unknown"]
    s["unknown_questions"] = len(unk)
    s["invented_on_unknown"] = sum(not r["abstained"] for r in unk)
    known = [r for r in rows if r["group"] not in ("unknown", "clarify")]
    s["refused_known"] = sum(r["abstained"] for r in known)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--name", required=True); ap.add_argument("--mode", default="k1", choices=["k1", "k3close"])
    ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dense", type=float, default=2.0); ap.add_argument("--key-head", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    t0 = time.time()
    eng = E.SkillEngine(str(ROOT / a.model), native=str(ROOT / "data" / "cv0.exe"), device=a.device,
                        dense=0.0 if a.key_head else a.dense, log=lambda *x: None)
    if a.key_head or a.dense == 0:
        eng.mem.learned = None
    if a.key_head:
        eng.mem.attach_learned(m2_qa.LearnedKeys(str(ROOT / a.key_head), str(ROOT / "data" / "cv0.exe"), a.device))
    v2 = json.loads((ROOT / "data_skills" / "test_v2.json").read_text())
    for q, ans in V.chat_probe.TEACH + T.facts_in_order() + [tuple(f) for f in v2["facts"]]:
        eng.add(q.encode(), ans.encode())
    k, policy = (1, "close") if a.mode == "k1" else (3, "close")
    sets = {"test_v1": [i for i in T.items() if i[0] != "clarify"] + [i for i in T.items() if i[0] == "clarify"],
            "test_v2": [tuple(i) for i in v2["items"]],
            "chat_probe": [(sec, q, kw, {}) for sec, q, kw in V.chat_probe.PROBES]}
    res = {"model": a.model, "name": a.name, "mode": a.mode, "retrieval": a.key_head or (f"MiniLM x{a.dense}" if a.dense else "literal only"),
           "date": time.strftime("%Y-%m-%d %H:%M"), "sets": {}}
    allrows = []
    for name, items in sets.items():
        rows = score_set(eng, items, k, policy)
        allrows += rows
        res["sets"][name] = {"summary": summarise(rows), "rows": rows}
        print(a.name, name, json.dumps(res["sets"][name]["summary"]), flush=True)
    res["all"] = summarise(allrows)
    res["seconds"] = round(time.time() - t0)
    print(a.name, "ALL", json.dumps(res["all"]))
    (LAB / "results" / f"grounded_{a.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
