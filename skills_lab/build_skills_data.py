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
import gen_skills_v4 as G4  # noqa: E402

OUT = ROOT / "data_skills"


def train(n: int, seed: int = 11, version: int = 3):
    """version 3: data_skills/train_v3.jsonl; version 4 (rewriting, 15 October): data_skills/train_v4.jsonl."""
    G3.setup(test=False)
    make = G4.make if version == 4 else G3.make
    rng = np.random.default_rng(seed)
    alp = E.m2_qa.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
    alpaca = lambda r: tuple(x.decode("utf-8", "replace") for x in alp[int(r.integers(len(alp)))])
    OUT.mkdir(exist_ok=True)
    kinds, words = collections.Counter(), 0
    with open(OUT / f"train_v{version}.jsonl", "w", encoding="utf-8") as f:
        for _ in range(n):
            w = make(rng, alpaca)
            if rng.random() < 0.10:
                w["q"] = E.m2_qa.add_typos(w["q"].encode(), rng).decode("utf-8", "replace")
            w["cands"] = [list(c) for c in w["cands"]]; w["turns"] = [list(t) for t in w["turns"]]
            kinds[w["kind"]] += 1
            words += sum(len(x.split()) for x in [w["q"], w["target"]] + [y for c in w["cands"] + w["turns"] for y in c])
            f.write(json.dumps(w) + "\n")
    man = {"version": version, "windows": n, "seed": seed, "kinds": dict(kinds), "words": words,
           "fields": {"person": sorted(G.P), "thing": sorted(G.TH)}, "names_used": len(G.NAMES), "names_reserved_for_test": len(G3.RESERVED)}
    (OUT / f"manifest_v{version}.json").write_text(json.dumps(man, indent=1))
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


def test4(seed: int = 41):
    """Held-out rewriting test (version 4): 15 short answers, 15 yes/no (8 true, 7 false claims), 10 lists, reserved names
    and phrasings only. Graded in eval_skills by the value and the form (extra["format"])."""
    G4.setup(test=True)
    rng = np.random.default_rng(seed)
    facts, items = [], []
    while len([i for i in items if i[0] == "short"]) < 15:
        w = G4.make(rng)
        if w["kind"] == "short":
            facts += [tuple(c) for c in w["cands"]]
            items.append(["short", w["q"], [[w["target"].rstrip(".").lower()]], {"format": "short", "value": w["target"].rstrip(".")}])
    n_yes = n_no = 0
    while n_yes + n_no < 15:
        w = G4.make(rng)
        if w["kind"] != "yesno":
            continue
        yes = w["target"].startswith("Yes")
        if (yes and n_yes >= 8) or (not yes and n_no >= 7):
            continue
        n_yes += yes; n_no += not yes
        facts += [tuple(c) for c in w["cands"]]
        value = w["target"].rstrip(".").split(" ")[-1]
        items.append(["yesno", w["q"], [[value.lower()]], {"format": "yes" if yes else "no"}])
    while len([i for i in items if i[0] == "list"]) < 10:
        w = G4.make(rng)
        if w["kind"] == "list":
            facts += [tuple(c) for c in w["cands"]]
            vals = [line.split(": ", 1)[1].lower() for line in w["target"].splitlines()]
            items.append(["list", w["q"], [[vals[0]], [vals[1]]], {"format": "list"}])
    data = {"version": 4, "seed": seed, "facts": facts, "items": items,
            "groups": dict(collections.Counter(i[0] for i in items)), "note": "rewriting test: reserved names and phrasings only"}
    (OUT / "test_v4_rewrite.json").write_text(json.dumps(data, indent=1))
    print(json.dumps(data["groups"]))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("what", choices=["train", "test", "train4", "test4"]); ap.add_argument("--n", type=int, default=100000)
    a = ap.parse_args()
    {"train": lambda: train(a.n), "test": test, "train4": lambda: train(a.n, version=4), "test4": test4}[a.what]()
