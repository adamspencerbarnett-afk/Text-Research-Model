"""Multi-turn probe v2 (15 October 2026): 120 generated conversations about invented facts, so the conversation
score is not decided by a handful of items (the 27-conversation probe ranked the session 31 arms opposite to history
gain on 1,000 real conversations). Uses gen_skills' held-out split (reserved names and ways of asking, never in any
training), played live through the Q&A engine with history, as mt_probe.py does.

Groups, 30 conversations each, keyword-graded on the final answer:
  recall      the user states a fact, then 1-2 unrelated questions, then asks for the fact in other words
  correction  the user states a fact, later corrects it ("Actually, ..."), then asks: the corrected value is right
  select      the user states facts about two people (same field), then asks about one of them
  return      a fact, then three unrelated turns, then the question

    python compiler/experiments/mt_probe2.py MODEL.pt --device cuda --graph --precision bf16 [--tag _x]
Writes results/mtprobe2_<model><tag>.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import m2_qa  # noqa: E402

OTHER = ["What is the capital of Italy?", "Name a famous painter.", "What is a good book to read on a plane?", "How do I make pancakes?",
         "What is the tallest mountain in the world?", "Give me a tip for staying focused.", "What is the boiling point of water?",
         "Name three colors of the rainbow.", "Who wrote Hamlet?", "What is a healthy breakfast?", "How many continents are there?",
         "Tell me a fun fact about dolphins."]


def statement(answer: str) -> str:
    return answer


def build(seed: int = 31):
    G3.setup(test=True)                      # reserved names and ways of asking only
    rng = np.random.default_rng(seed)
    convs = []
    for _ in range(30):                      # recall
        f = G.Fact(rng)
        turns = [statement(f.stored()[1])] + list(rng.choice(OTHER, int(rng.integers(1, 3)), replace=False))
        convs.append(("recall", turns, f.ask(), [[f.value.lower()]], []))
    n = 0
    while n < 30:                            # correction
        f = G.Fact(rng)
        if not f.can_update():
            continue
        old = f.value; first = statement(f.stored()[1]); ask = f.ask()
        (_, new_ans), _ = f.update()
        corr = "Actually, " + new_ans[0].lower() + new_ans[1:]
        turns = [first] + list(rng.choice(OTHER, int(rng.integers(0, 2)), replace=False)) + [corr]
        convs.append(("correction", turns, ask, [[f.new_value.lower()]], [old.lower()])); n += 1
    n = 0
    while n < 30:                            # select between two people
        f = G.Fact(rng, "p")
        g = G.Fact(rng, "p", f.field, G.person(rng), extra={"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]})
        if g.value == f.value or g.ent["id"] == f.ent["id"]:
            continue
        pair = [statement(f.stored()[1]), statement(g.stored()[1])]
        rng.shuffle(pair)
        target = f if rng.random() < 0.5 else g
        other = g if target is f else f
        convs.append(("select", pair, target.ask(), [[target.value.lower()]], [other.value.lower()])); n += 1
    for _ in range(30):                      # return after three unrelated turns
        f = G.Fact(rng)
        turns = [statement(f.stored()[1])] + list(rng.choice(OTHER, 3, replace=False))
        convs.append(("return", turns, f.ask(), [[f.value.lower()]], []))
    return convs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--graph", action="store_true"); ap.add_argument("--precision", default="fp32", choices=["fp32", "bf16", "int8"])
    ap.add_argument("--max-codes", type=int, default=60); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    native = ROOT / "data" / "cv0.exe"
    eng = m2_qa.QAEngine(a.model, native=str(native) if native.exists() else None, device=a.device, precision=a.precision, log=lambda *x: None)
    fast = "graph" if a.graph else True
    rows, t0 = [], time.perf_counter()
    for group, turns, final, keys, wrong in build():
        history = []
        for u in turns:
            r, _ = eng.ask(u.encode(), a.max_codes, 3, fast=fast, history=list(history) or None)
            history.append((u.encode(), r))
        ans, _ = eng.ask(final.encode(), a.max_codes, 3, fast=fast, history=history)
        alone, _ = eng.ask(final.encode(), a.max_codes, 3, fast=fast, use_memory=False)
        text, low = ans.decode("utf-8", "replace"), ans.decode("utf-8", "replace").lower()
        ok = all(any(k in low for k in g) for g in keys)
        rows.append({"group": group, "turns": turns, "question": final, "answer": text, "ok": ok,
                     "wrong_value": any(w in low for w in wrong),
                     "ok_without_history": all(any(k in alone.decode("utf-8", "replace").lower() for k in g) for g in keys)})
    summary = {}
    for r in rows:
        s = summary.setdefault(r["group"], {"n": 0, "ok": 0, "ok_without_history": 0, "wrong_value": 0})
        s["n"] += 1; s["ok"] += r["ok"]; s["ok_without_history"] += r["ok_without_history"]; s["wrong_value"] += r["wrong_value"]
    total = {"n": len(rows), "ok": sum(r["ok"] for r in rows), "ok_without_history": sum(r["ok_without_history"] for r in rows)}
    stem = Path(a.model).stem + a.tag
    (ROOT / "results" / f"mtprobe2_{stem}.json").write_text(json.dumps({"model": a.model, "total": total, "summary": summary,
                                                                         "seconds": round(time.perf_counter() - t0, 1), "rows": rows}, indent=1))
    print(json.dumps({"total": total, "summary": summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
