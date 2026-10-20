"""Retrieval study (Adam, 16 Oct 2026): before v2.3 is trained on what the memory search returns, measure the search on
its own: does the right memory item reach the memory block, how much noise comes with it, and how fast is it?
No model answers anything here and nothing is trained; chat22 files are not changed (variants are built in this script).

Cases (held-out names and phrasings, G3.setup(test=True)):
  conversation  long_grid cases: one fact, two-step, distractor, never told, correction; 1,024 and 4,096 codes of real
                held-out conversation; every turn goes into conversation memory; the question is searched with no window
                (memory must find it: the any-length view)
  uploaded      the held-out skills facts (test_v2) stored among a 30,000-fact sample of the v2.3 uploaded pool; its
                questions (taught, near miss, update) searched
  speed         ms per search with the full 137,944-fact uploaded pool and 100 / 1,000 / 10,000 conversation entries
Variants: the key (word only / word + MiniLM / word + K2), name-following (on / off / fixed), the recent-turns search
(on / off), word stemming (off / on), the gate (chat22 default / step 1 strict).
    python research_side/retrieval_study.py [--n 40]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler", ROOT):
    sys.path.insert(0, str(p))
import long_grid as LG  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import chat22.memory as CM  # noqa: E402
from chat22 import encoder as E  # noqa: E402

NAME_STOP = {b"got", b"thanks", b"thank", b"okay", b"noted", b"yes", b"no", b"the", b"this", b"that", b"sure", b"actually",
             b"here", b"there", b"what", b"when", b"where", b"who", b"how", b"please", b"hello", b"hi", b"my", b"your", b"it",
             b"i", b"we", b"they", b"he", b"she", b"and", b"but", b"so", b"also", b"great", b"good", b"well", b"assistant", b"user"}
SUFFIX = (b"ing", b"ies", b"ied", b"es", b"ed", b"s")


def stem(w: bytes) -> bytes:
    for s in SUFFIX:
        if w.endswith(s) and len(w) - len(s) >= 3:
            return w[:-len(s)] + (b"y" if s in (b"ies", b"ied") else b"")
    return w


ORIG_KEY = CM.key_of


class StudyMemory(CM.Memory):
    """chat22's Memory.search with switches for the parts under study."""

    def __init__(self, dense, gate, gate_facts, names, recent):
        super().__init__(dense, gate=gate, gate_facts=gate_facts)
        self.names_mode, self.recent_on = names, recent

    def search(self, msg, recent=(), first_turn_in_window=None, k_facts=None, k_session=None):
        k_facts = self.k_facts if k_facts is None else k_facts
        k_session = self.k_session if k_session is None else k_session
        qk = CM.key_of(msg)
        qvec = self._vec([msg])[0] if self.dense is not None else None
        gf = self.gate_facts if self.gate_facts is not None else "default"
        facts = self._scores(self.facts, qk, qvec, False, gate=gf)[:k_facts]
        ex = self.facts.exact.get(qk) if len(qk) >= 2 else None
        if ex is not None:
            facts = [(9.0, 1.0, 1.0, ex)] + [f for f in facts if f[3] != ex][:k_facts - 1]
        sess = {i: s for s, _, _, i in self._scores(self.session, qk, qvec, True)}
        if recent and self.recent_on:
            rk = CM.key_of(b" ".join([msg] + list(recent)))
            for s, _, _, i in self._scores(self.session, rk, None, True):
                sess[i] = max(sess.get(i, 0.0), 0.6 * s)
        names = set()
        if self.names_mode != "off":
            for i in sorted(sess, key=sess.get, reverse=True)[:3]:
                it = self.session.items[i]
                found = {n.lower() for n in CM.NAME.findall(it.q + b" " + it.a)}
                if self.names_mode == "fixed":
                    found = {n for n in found if n not in NAME_STOP}
                names |= found - set(qk)
        if names:
            for s, _, _, i in self._scores(self.session, qk | frozenset(CM.key_of(b" ".join(names))), None, True):
                sess[i] = max(sess.get(i, 0.0), 0.9 * s)
        top_s = sorted(sess, key=sess.get, reverse=True)[:k_session]
        ranked = sorted([(s, "f", i) for s, _, _, i in facts] + [(sess[i], "s", i) for i in top_s], reverse=True)
        return [self.facts.items[i] if p == "f" else self.session.items[i] for _, p, i in ranked], {}


VARIANTS = {   # name: (key, names, recent, stemming, gate, gate_facts)
    "current (chat22 default)": ("minilm", "on", True, False, (0.15, 0.55), None),
    "word key only": ("word", "on", True, False, (0.15, 0.55), None),
    "word + K2 (ours)": ("k2", "on", True, False, (0.15, 0.55), None),
    "names off": ("minilm", "off", True, False, (0.15, 0.55), None),
    "names fixed": ("minilm", "fixed", True, False, (0.15, 0.55), None),
    "recent-turns search off": ("minilm", "fixed", False, False, (0.15, 0.55), None),
    "stemming on": ("minilm", "fixed", True, True, (0.15, 0.55), None),
    "step 1 strict gate": ("minilm", "fixed", True, False, (0.15, 0.55), (0.35, 0.7)),
    "fixed + stemming + strict": ("minilm", "fixed", True, True, (0.15, 0.55), (0.35, 0.7)),
}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--pool", type=int, default=30000)
    a = ap.parse_args()
    torch.set_num_threads(6)
    G3.setup(test=True)
    import dense_keys
    import m2_qa
    keys = {"word": None, "minilm": dense_keys.DenseKeys("cpu", weight=2.0),
            "k2": m2_qa.LearnedKeys(str(ROOT / "runs/k2_5m/key_head.pt"), str(ROOT / "data/cv0.exe"), "cpu")}
    core = torch.load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
    enc = E.load(ROOT / core["encoding"]["dictionary"])
    rng = np.random.default_rng(5)
    # the uploaded pool sample (balanced as in the full pool) + the held-out skills facts
    pool = [json.loads(l) for l in open(ROOT / "data_chat22/facts_pool.jsonl", encoding="utf-8")]
    V_all = np.load(ROOT / "data_chat22/facts_pool_minilm.npy")
    pick = np.sort(rng.choice(len(pool), a.pool, replace=False))
    test = json.loads((ROOT / "data_skills/test_v2.json").read_text())
    upl = [(pool[i]["q"].encode(), pool[i]["a"].encode(), pool[i]["key"].encode()) for i in pick] + \
          [(q.encode(), x.encode(), q.encode()) for q, x in test["facts"]]
    vec_cache = {"minilm": torch.from_numpy(np.concatenate([V_all[pick], keys["minilm"].embed([q.encode() for q, _ in test["facts"]]).numpy()]))}
    t0 = time.time(); vec_cache["k2"] = keys["k2"].embed([k for _, _, k in upl]); print(f"K2 vectors for {len(upl):,} facts: {time.time() - t0:.0f} s", flush=True)
    # conversation cases, built once
    class _E: pass
    _E.enc = enc
    grid = LG.Grid(_E, np.random.default_rng(11))
    cases = []
    for task in ("one", "two", "distractor", "absent", "correction"):
        for L in (1024, 4096):
            for j in range(a.n):
                turns, q, want, wrong, absent, stating = grid.case(task, L, (0.1, 0.5, 0.9)[j % 3])
                cases.append((task, L, turns, q, want, wrong, absent, stating))
    res = {}
    for vname, (key, names, recent, stemming, gate, gate_f) in VARIANTS.items():
        CM.key_of = (lambda t: frozenset(stem(w) for w in ORIG_KEY(t))) if stemming else ORIG_KEY
        dk = keys[key]
        m = StudyMemory(dk, gate, gate_f, names, recent)
        m.add_facts(upl, vecs=vec_cache.get(key) if dk is not None else None)
        conv = {}; t1 = time.time()
        for task, L, turns, q, want, wrong, absent, stating in cases:
            m.forget_session()
            for n_, (u, r) in enumerate(turns):
                m.add_turn(u, r, n_ + 1)
            items, _ = m.search(q, [t[0] for t in turns[-2:]], None)
            sess_items = [it for it in items if it.source == "turn"]
            txt = [(it.q + b" " + it.a).decode("utf-8", "replace").lower() for it in sess_items]
            gold_turns = {s + 1 for s in stating}
            got = {it.turn for it in sess_items} & gold_turns
            hit = (len(got) == len(gold_turns)) if gold_turns else None
            noise_same_field = sum(any(w.lower() in t for w in wrong) for t in txt)
            c = conv.setdefault(f"{task}", {"cases": 0, "hit": 0, "wrong_value_in_block": 0, "block": 0, "facts_in_block": 0})
            c["cases"] += 1; c["hit"] += bool(hit); c["wrong_value_in_block"] += noise_same_field > 0
            c["block"] += len(items); c["facts_in_block"] += sum(it.source == "fact" for it in items)
        uploaded = {"questions": 0, "hit": 0, "block": 0}
        for group, q, kw, extra in test["items"]:
            if group not in ("taught", "nearmiss", "update"):
                continue
            m.forget_session()
            items, _ = m.search(q.encode(), [], None)
            txt = [(it.q + b" " + it.a).decode("utf-8", "replace").lower() for it in items]
            uploaded["questions"] += 1; uploaded["block"] += len(items)
            uploaded["hit"] += any(all(any(k in t for k in g) for g in kw) for t in txt)
        for c in conv.values():
            c["block"] = round(c["block"] / c["cases"], 1); c["facts_in_block"] = round(c["facts_in_block"] / c["cases"], 1)
        uploaded["block"] = round(uploaded["block"] / max(1, uploaded["questions"]), 1)
        res[vname] = {"conversation": conv, "uploaded": uploaded, "seconds": round(time.time() - t1)}
        print(vname, json.dumps(res[vname]), flush=True)
    CM.key_of = ORIG_KEY
    # speed with the full uploaded pool
    speed = {}
    for key in ("word", "minilm", "k2"):
        if key == "k2":
            continue                                       # K2 needs its vectors for all 137,944 facts (~1 h on the CPU); the
        dk = keys[key]                                     # sample above measures its accuracy; its query time is in the note
        m = StudyMemory(dk, (0.15, 0.55), None, "fixed", True)
        allv = torch.from_numpy(V_all) if key == "minilm" else None
        m.add_facts([(r["q"].encode(), r["a"].encode(), r["key"].encode()) for r in pool], vecs=allv)
        filler = LG.Grid(_E, np.random.default_rng(3)).pairs
        for size in (100, 1000, 10000):
            m.forget_session()
            ts = filler[:size] if len(filler) >= size else (filler * (size // len(filler) + 1))[:size]
            if dk is not None:
                vv = dk.embed([u for u, _ in ts])
            for n_, (u, r) in enumerate(ts):
                m.session.add(CM.Item(u, r, "turn", n_ + 1), u, vv[n_:n_ + 1] if dk is not None else None)
            qs = [b"Where does my sister live?", b"What is the wifi password at the library?", b"When is the team meeting?"] * 10
            t1 = time.perf_counter()
            for q in qs:
                m.search(q, [], None)
            speed[f"{key} | {size} conversation entries + {len(pool):,} uploaded"] = round(1000 * (time.perf_counter() - t1) / len(qs), 1)
            print("speed", key, size, speed[f"{key} | {size} conversation entries + {len(pool):,} uploaded"], "ms", flush=True)
    out = {"date": time.strftime("%Y-%m-%d %H:%M"), "n_per_task_length": a.n, "uploaded_pool_sample": a.pool, "variants": res,
           "speed_ms_per_search": speed, "note": "K2 query time 15.3 ms (session 34, k2v2); accuracy measured on the sample"}
    (ROOT / "research_side" / "retrieval_study.json").write_text(json.dumps(out, indent=1))
    print("saved research_side/retrieval_study.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
