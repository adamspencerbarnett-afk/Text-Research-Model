"""Build the skills-arm dataset (version 3) and the held-out test v2 (14 October 2026).

    .venv/Scripts/python.exe -B skills_lab/build_skills_data.py train --n 100000     -> data_skills/train_v3.jsonl + manifest
    .venv/Scripts/python.exe -B skills_lab/build_skills_data.py test                 -> data_skills/test_v2.json (200 questions)

Training windows never use the reserved names or each field's reserved way of asking; test v2 uses only those, and no
test v1 name (testset.TEST_NAMES) appears in either. 10% of training questions carry typing mistakes.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
import numpy as np  # noqa: E402

import engine as E  # noqa: E402
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402

OUT = ROOT / "data_skills"


def train(n: int, seed: int = 11):
    G3.setup(test=False)
    rng = np.random.default_rng(seed)
    alp = E.m2_qa.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
    alpaca = lambda r: tuple(x.decode("utf-8", "replace") for x in alp[int(r.integers(len(alp)))])
    OUT.mkdir(exist_ok=True)
    kinds, words = collections.Counter(), 0
    with open(OUT / "train_v3.jsonl", "w", encoding="utf-8") as f:
        for _ in range(n):
            w = G3.make(rng, alpaca)
            if rng.random() < 0.10:
                w["q"] = E.m2_qa.add_typos(w["q"].encode(), rng).decode("utf-8", "replace")
            w["cands"] = [list(c) for c in w["cands"]]; w["turns"] = [list(t) for t in w["turns"]]
            kinds[w["kind"]] += 1
            words += sum(len(x.split()) for x in [w["q"], w["target"]] + [y for c in w["cands"] + w["turns"] for y in c])
            f.write(json.dumps(w) + "\n")
    man = {"version": 3, "windows": n, "seed": seed, "kinds": dict(kinds), "words": words,
           "fields": {"person": sorted(G.P), "thing": sorted(G.TH)}, "names_used": len(G.NAMES), "names_reserved_for_test": len(G3.RESERVED)}
    (OUT / "manifest_v3.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man))


def kw(v: str):
    return [[v.lower()]]


def test(seed: int = 23):
    G3.setup(test=True)
    rng = np.random.default_rng(seed)
    facts, upd, items = [], [], []
    for _ in range(40):
        f = G.Fact(rng); facts.append(f.stored()); items.append(["taught", G3.wrap(rng, f.ask()), kw(f.value), {}])
    for j in range(40):
        f = G.Fact(rng)
        while not f.can_update() or (j >= 30 and f.spec[5] is None):
            f = G.Fact(rng)
        old_v = f.value; facts.append(f.stored()); ask = f.ask()
        (nq, na), prev = f.update()
        upd.append((nq, na))
        items.append(["update", ask if j < 30 else prev, kw(f.new_value if j < 30 else old_v),
                      {"kind": "current" if j < 30 else "earlier", "wrong": [old_v.lower()] if j < 30 else []}])
    for _ in range(20):
        f = G.Fact(rng, "p")
        g = G.Fact(rng, "p", f.field, G.person(rng), extra={"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]})
        if g.value == f.value:
            continue
        facts += [f.stored(), g.stored()]
        items += [["nearmiss", f.ask(), kw(f.value), {}], ["nearmiss", g.ask(), kw(g.value), {}]]
    for _ in range(40):
        f = G.Fact(rng)
        if f.kind == "p" and rng.random() < 0.5:
            facts.append(G.Fact(rng, "p", G.pick(rng, [k for k in G.P if k != f.field]), dict(f.ent)).stored())
        items.append(["unknown", f.ask(), None, {}])
    n_cl = 0
    while n_cl < 20:
        w = G.make(rng)
        if w["kind"] not in ("clarify", "followup"):
            continue
        cands = w["cands"]
        if w["kind"] == "followup":
            ask, clar = w["turns"][0]
        else:
            ask, clar = w["q"], w["target"]
        labels = clar.removeprefix("Do you mean ").rstrip("?").split(" or ")
        labels = [l.removeprefix("the ").split(" ")[0] for l in labels]
        facts += [tuple(c) for c in cands]
        j = int(rng.integers(2))
        fu = f"I mean {labels[j]}." if labels[j][:1].isupper() else f"The {labels[j]} one."
        mine, other = cands[j][1].rstrip(".").split(" "), set(cands[1 - j][1].rstrip(".").lower().split(" "))
        key = [w for w in mine if w.lower() not in other and w.lower() != labels[j].lower()] or [mine[-1]]
        items.append(["clarify", ask, [[labels[0].lower()], [labels[1].lower()]], {"followup": fu, "followup_kw": [[key[-1].lower()]]}])
        n_cl += 1
    data = {"version": 2, "seed": seed, "facts": facts + upd, "items": items,
            "groups": dict(collections.Counter(i[0] for i in items)), "note": "reserved names and phrasings only; facts stored oldest first, updates last"}
    OUT.mkdir(exist_ok=True)
    (OUT / "test_v2.json").write_text(json.dumps(data, indent=1))
    print(json.dumps({k: v for k, v in data.items() if k in ("groups", "version")}), len(items) + 20, "questions with follow-ups")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["train", "test"]); ap.add_argument("--n", type=int, default=100000)
    a = ap.parse_args()
    train(a.n) if a.what == "train" else test()
