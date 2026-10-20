"""Adam's case (19 Oct 16:50): a stored fact asked in natural words ("I forgot my password for the office wifi, can you
tell me what it is?") was refused although the fact was in the memory block. This test: 30 stored skill-style facts, each
asked (a) with its plain question, (b) with 2 natural rewordings (prefix / reshaped), (c) the plain question again AFTER a
refused rewording sits in the window (does one refusal pull the next answer along?). Engine v2, each model's own store.

    python research_side/natural_rewording_test.py [--models mix_5m_s1,mix_5m_s2,v24a_s1,real_5m_s1]
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
from chat22.engine import Engine  # noqa: E402
from chat22.tests import scoring as S  # noqa: E402
from research_side.fresh_facts_test import undo_last_fact  # noqa: E402

PREFIX = ["I forgot, ", "Remind me, ", "Do you know ", "Can you tell me ", "Quick question: ", "Sorry, "]
ADAM = [("What is the office wifi password?", "The office wifi password is maple42",
         ["I forgot my password for the office wifi, can you tell me what it is?", "What is the office password?"], ["maple42"])]


def reword(q: str, rng) -> list[str]:
    body = q[0].lower() + q[1:]
    a = G.pick(rng, PREFIX) + body
    b = "I can't remember: " + body if rng.random() < 0.5 else body.rstrip("?") + ", do you remember?"
    return [a, b]


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--models", default="mix_5m_s1,mix_5m_s2,v24a_s1,real_5m_s1"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    G3.setup(test=True); rng = np.random.default_rng(8)
    facts = []
    while len(facts) < 30:
        f = G.Fact(rng)
        q, st = f.ask(), f.stored()[1]
        facts.append((q, st, reword(q, rng), [str(f.value)]))
    res = {}
    for m in a.models.split(","):
        e = Engine(str(ROOT / "runs" / "chat22" / f"{m}.pt"), a.device, log=lambda *x: None, search="v2")
        r = {"plain": 0, "reworded": 0, "n_reworded": 0, "plain_after_refusal": 0, "n_after": 0, "adam": 0, "n_adam": 0, "found_but_refused": 0}
        for q, st, rws, want in facts + ADAM:
            e.forget(); e.add_fact(q.encode(), st.encode())
            ans = e.reply(q.encode(), 60)[0].decode("utf-8", "replace"); ok = S.right(ans, [[w] for w in want], [])
            if (q, st) == (ADAM[0][0], ADAM[0][1]):
                for rw in rws:
                    r["n_adam"] += 1; r["adam"] += S.right(e.reply(rw.encode(), 60)[0].decode("utf-8", "replace"), [[w] for w in want], [])
            else:
                r["plain"] += ok
                for rw in rws:
                    out, info = e.reply(rw.encode(), 60); t = out.decode("utf-8", "replace"); good = S.right(t, [[w] for w in want], [])
                    r["n_reworded"] += 1; r["reworded"] += good
                    if not good and info["items"]:
                        r["found_but_refused"] += S.abstained(t)
                        # the refusal stays in the window; does the plain question still get answered?
                        e.remember(rw.encode(), out)
                        t2 = e.reply(q.encode(), 60)[0].decode("utf-8", "replace")
                        r["n_after"] += 1; r["plain_after_refusal"] += S.right(t2, [[w] for w in want], [])
                        e.undo_last()
            undo_last_fact(e)
        res[m] = r; print(m, json.dumps(r), flush=True)
    (ROOT / "research_side" / "natural_rewording_test.json").write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
