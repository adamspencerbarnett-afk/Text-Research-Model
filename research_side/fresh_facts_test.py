"""Two quick tests of the thing Adam does by hand (19 Oct): (1) /add one fact to the permanent store, then ask it, exact
and reworded, with nothing else in memory: 30 skill-style facts (held-out skills test) + 30 real short facts (held-out
real Q&A, memory-fact kind); (2) a two-step correction inside the window: "My <rel> is called <Name>." ... "<Name> <fact>."
... "Actually, <Name> <new fact>." then "<question about my <rel>>". Engine v2, each model's own store.

    python research_side/fresh_facts_test.py [--models v24a_s1,real_5m_s1,hyb_5m_s1] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler", ROOT):
    sys.path.insert(0, str(p))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
from chat22 import encoder as E  # noqa: E402
from chat22.engine import Engine  # noqa: E402
from chat22.tests import scoring as S  # noqa: E402
from chat22.windows_train import READING_RX, perturb_question  # noqa: E402

ACK = b"Got it. I'll remember that."


def undo_last_fact(e):
    """Take the last added fact back out of the permanent store (pool index, counts, vector, fact-table records)."""
    pool = e.memory.facts
    n = len(pool.items) - 1
    item = pool.items.pop(); k = pool.keys.pop()
    if pool.exact.get(k) == n:
        del pool.exact[k]
    for w in k:
        if w in pool.index and pool.index[w] and pool.index[w][-1] == n:
            pool.index[w].pop()
            if not pool.index[w]:
                del pool.index[w]
        pool.df[w] -= 1
    if pool.chunks:
        pool.chunks.pop()
    elif pool.joined is not None and len(pool.joined) > n:
        pool.joined = pool.joined[:n]
    if e.search is not None:
        t = e.search.uploaded
        for key in [key for key, r in t.records.items() if r.item is item]:
            del t.records[key]
        for key in [key for key, it in t.links.items() if it is item]:
            del t.links[key]


def fresh_facts(rng):
    G3.setup(test=True)
    t = json.loads((ROOT / "data_skills" / "test_v2.json").read_text())
    facts = {q: a for q, a in t["facts"]}
    cases = []
    for group, q, kw, extra in t["items"]:
        if group == "taught" and len(cases) < 30:
            # the stored fact whose answer holds the keywords
            src = next(((fq, fa) for fq, fa in facts.items() if all(any(k in fa.lower() for k in g) for g in kw)), None)
            if src:
                cases.append(("skill", src[0].encode(), src[1].encode(), q.encode(), kw))
    held = E.parse_exchanges((ROOT / "data_real" / "shared" / "eval" / "qa_heldout.txt").read_bytes())
    pool = [(q, a) for q, a in held if not READING_RX.search(q[:400]) and 3 < len(a) < 60 and not a.lower().startswith((b"yes", b"no"))]
    for i in rng.choice(len(pool), 30, replace=False):
        q, a = pool[int(i)]
        key = a.strip().rstrip(b".").decode("utf-8", "replace").lower()
        cases.append(("real", q, a, q, [[key]]))
    return cases


def two_step_corrections(rng, n=20):
    G3.setup(test=True)
    out = []
    while len(out) < n:
        f = G.Fact(rng, "p")
        if not f.can_update():
            continue
        rel, name = G.pick(rng, G.RELS), G.pick(rng, G.NAMES)
        fmt = {"e": name, "E": name, "ep": name + "'s", "Ep": name + "'s", "id": name}
        f2 = G.Fact(rng, "p", f.field, fmt)
        if not f2.can_update():
            continue
        st = f2.stored()[1]
        (_, new_s), _ = f2.update()
        old = f2.fmt; f2.fmt = dict(old, e=f"my {rel}", E=f"My {rel}", ep=f"my {rel}'s", Ep=f"My {rel}'s")
        q = f2._f(G.pick(rng, f2.spec[2])); f2.fmt = old
        turns = [(f"My {rel} is called {name}.", "Thanks, noted."), (st, "Got it."), ("Actually, " + new_s[0].lower() + new_s[1:], "Okay, I'll keep that in mind.")]
        out.append((turns, q, f2.new_value, f2.value))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="v24a_s1,real_5m_s1,real_5m_s2,hyb_5m_s1,hyb_5m_s2"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    cases = fresh_facts(rng); corr = two_step_corrections(np.random.default_rng(4))
    res = {}
    for m in a.models.split(","):
        e = Engine(str(ROOT / "runs" / "chat22" / f"{m}.pt"), a.device, log=lambda *x: None, search="v2")
        r = {"skill_exact": 0, "skill_reworded": 0, "real_exact": 0, "real_reworded": 0, "n_skill": 0, "n_real": 0, "twostep_corr": 0, "twostep_old": 0}
        for kind, fq, fa, q, kw in cases:
            r[f"n_{kind}"] += 1
            e.forget(); e.add_fact(fq, fa)
            ans = e.reply(q, 80)[0].decode("utf-8", "replace")
            r[f"{kind}_exact"] += S.right(ans, [[k for k in g] for g in kw], [])
            ans2 = e.reply(perturb_question(fq, rng, fq.split(b" ")), 80)[0].decode("utf-8", "replace")
            r[f"{kind}_reworded"] += S.right(ans2, [[k for k in g] for g in kw], [])
            undo_last_fact(e)
        for turns, q, new, old in corr:
            e.forget()
            for u, rr in turns:
                e.remember(u.encode(), rr.encode())
            ans = e.reply(q.encode(), 60)[0].decode("utf-8", "replace")
            r["twostep_corr"] += S.has(ans, new) and not S.has(ans, old); r["twostep_old"] += S.has(ans, old)
        res[m] = r
        print(m, json.dumps(r), flush=True)
    (ROOT / "research_side" / "fresh_facts_test.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
