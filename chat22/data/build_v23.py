"""v2.3 datasets (16 October 2026; docs/MIX_STUDY_V23.md section 6). Writes data_chat22/ only.

  facts     the uploaded / stored facts pool (permanent memory), kept apart from conversation memory:
            Q&A exchanges (training split), stored skill facts (from the skills training records), and pieces of real
            documents from the long-document pool (as if a user uploaded them). One JSON row each:
            {"q", "a", "key", "source": "qa" | "skill" | "doc"}; key = the text the search indexes.
  longconv  any-length conversations: real dialogues stitched into sessions of 10-10,000 turns (log-uniform) with
            planted facts, corrections (some by name), two-step links, look-alikes, uploaded-fact questions and
            never-told questions. At each question the memory block is what the REAL chat22 search returns (word key +
            MiniLM, the conversation pool and the uploaded pool searched apart, a soft gate so look-alikes come back too;
            the trainer lays them out with the same window.build the engine uses). One JSON row per question.

    python -m chat22.data.build_v23 --only facts
    python -m chat22.data.build_v23 --only longconv --sessions 5000 [--shard 0 --shards 1]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

from .. import encoder as E
from .. import window as W
from ..memory import Memory, split_text

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_chat22"
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import gen_skills_v4 as G4  # noqa: E402

ACKS = ["Got it. I'll remember that.", "Thanks, noted.", "Okay, I'll keep that in mind.", "Thanks for telling me.", "Noted."]
# 19 Oct (Adam's hands-on case): questions about stored and told facts also arrive in natural, restructured words
VARY_PRE = ["I forgot, ", "Remind me, ", "Do you know ", "Can you tell me ", "Quick question: ", "Sorry, ", "I can't remember: ",
            "Hey, ", "One thing: ", "Do you remember "]
VARY_POST = [", do you remember?", " I forgot.", " Can you tell me?", " I can't recall.", ", if you have it?"]
VARY = False


def vary(q: str, rng) -> str:
    """A natural rewording 40% of the time: a prefix, a suffix, or both, around the plain question."""
    if not VARY or rng.random() >= 0.4:
        return q
    body = q[0].lower() + q[1:]
    r = rng.random()
    if r < 0.45:
        return G.pick(rng, VARY_PRE) + body
    if r < 0.8:
        return q.rstrip("?") + "?" + G.pick(rng, VARY_POST) if rng.random() < 0.5 else q.rstrip("?") + G.pick(rng, VARY_POST)
    return G.pick(rng, VARY_PRE) + body.rstrip("?") + G.pick(rng, VARY_POST)


def build_facts(n_skill: int = 20000, n_docs: int = 300, seed: int = 1) -> dict:
    rng = np.random.default_rng(seed)
    rows = []
    for q, a in E.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes()):
        if a.strip():
            rows.append({"q": q.decode("utf-8", "replace"), "a": a.decode("utf-8", "replace"), "key": q.decode("utf-8", "replace"), "source": "qa"})
    n_qa = len(rows)
    seen = set()
    sk = [json.loads(l) for l in open(ROOT / "data_skills" / "train_v4c.jsonl", encoding="utf-8")][:-2000]   # the held-out 2,000 stay out
    order = rng.permutation(len(sk))
    for j in order:
        for q, a in sk[j]["cands"]:
            if q in seen or not a.strip():
                continue
            seen.add(q); rows.append({"q": q, "a": a, "key": q, "source": "skill"})
        if len(seen) >= n_skill:
            break
    docs = [d for d in (OUT / "long_over8k.txt").read_bytes().split(b"\f") if d.strip()]
    for j in rng.choice(len(docs), min(n_docs, len(docs)), replace=False):
        d = docs[int(j)]
        title = d[:80].split(b"\n")[0].decode("utf-8", "replace").strip()[:60] or "an uploaded document"
        for p in split_text(d[:40_000]):
            t = p.decode("utf-8", "replace")
            if len(t) > 60:
                rows.append({"q": f"From the uploaded document \"{title}\"", "a": t, "key": t, "source": "doc"})
    with open(OUT / "facts_pool.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"rows": len(rows), "qa": n_qa, "skill": len(seen), "doc": len(rows) - n_qa - len(seen)}


def load_facts(path=None):
    return [json.loads(l) for l in open(path or (OUT / "facts_pool.jsonl"), encoding="utf-8")]


def question_like(r) -> bool:
    q = r["q"]
    a = r["a"]
    return (q.endswith("?") and len(q) < 200 and "\n" not in q and 20 < len(a) < 400
            and not a.startswith(("For ", "This ", "See ", "Not to be confused")))


# ---------------------------------------------------------------------- any-length sessions
class SessionMaker:
    def __init__(self, enc, ctx: int, rng, dense, shard: int, facts_path=None, dialogues=None):
        self.enc, self.ctx, self.rng, self.sep = enc, ctx, rng, enc.encode(E.SEP_TEXT)[0]
        G3.setup(test=False)                                   # training names and phrasings only
        self.facts = load_facts(facts_path)
        self.skill_facts = [r for r in self.facts if r["source"] == "skill"]
        self.qa_facts = [r for r in self.facts if r["source"] == "qa"]
        if facts_path is not None:                             # a real pool (data_real): stored-fact questions are real
            self.qa_facts = [r for r in self.facts if r["source"] == "qa" or (r["source"] in ("qa_real", "wiki") and question_like(r))]
            self.skill_facts = self.skill_facts or self.qa_facts
        # the uploaded pool, with its dense vectors computed once and cached on disk
        # the study's search fixes (names fixed, no recent-turns search); a soft gate on purpose, so look-alikes come back
        # into training blocks and the model learns to ignore them (the engine itself uses step 1's stricter setting)
        self.mem = Memory(dense, gate=(0.15, 0.55), k_facts=6, k_session=8, names_fixed=True, recent_search=False)
        vec_p = (Path(facts_path).with_name(Path(facts_path).stem + "_minilm.npy") if facts_path is not None else OUT / "facts_pool_minilm.npy")
        if dense is not None and vec_p.exists():
            import torch
            V = torch.from_numpy(np.load(vec_p)).to(dense.dev)
        elif dense is not None:
            import torch
            V = dense.embed([r["key"].encode("utf-8", "replace") for r in self.facts])
            np.save(vec_p, V.cpu().numpy())
        else:
            V = None
        self.mem.add_facts([(r["q"].encode("utf-8", "replace"), r["a"].encode("utf-8", "replace"), r["key"].encode("utf-8", "replace"))
                            for r in self.facts], vecs=V)
        # filler: real dialogue turns, their memory entries and vectors computed once
        pairs = []
        with open(dialogues or (OUT / "dialogues_train.jsonl"), encoding="utf-8") as f:
            for i, l in enumerate(f):
                if i % 7 != shard % 7:                         # different shards use different dialogues
                    continue
                t = json.loads(l)["turns"]
                pairs += [(t[j][1].encode("utf-8", "replace")[:600], t[j + 1][1].encode("utf-8", "replace")[:700])
                          for j in range(0, len(t) - 1, 2) if t[j][0] == "user" and t[j + 1][0] == "assistant"]
                if len(pairs) >= 30000:
                    break
        self.filler = pairs
        self.filler_entries = [Memory.turn_entries(u, r, 0) for u, r in pairs]
        if dense is not None:
            flat = [t for es in self.filler_entries for _, t in es]
            V = dense.embed(flat); self.filler_vecs, k = [], 0
            for es in self.filler_entries:
                self.filler_vecs.append(V[k:k + len(es)]); k += len(es)
        else:
            self.filler_vecs = [None] * len(pairs)

    def _turn_size(self, u, r):
        return len(self.enc.encode(E.render(u, r))[0])

    def session(self, n_turns: int, n_q: int) -> list:
        rng, mem = self.rng, self.mem
        mem.forget_session()
        name_fmt = lambda n: {"e": n, "E": n, "ep": n + "'s", "Ep": n + "'s"}
        # plan the statements: facts, look-alikes, corrections and two-step links, at random turns
        n_facts = int(rng.integers(3, 16))
        facts = [G.Fact(rng) for _ in range(n_facts)]
        events = {}                                            # turn -> (statement, ack)
        live = []                                              # (fact, current answer text, value, turn told, kind)
        def put(turn, text):
            while turn in events:
                turn += 1
            events[turn] = (text.encode(), G.pick(rng, ACKS).encode()); return turn
        for f in facts:
            t = put(int(rng.integers(0, max(1, n_turns - 1))), f.stored()[1])
            live.append([f, G4.your(f.stored()[1]), f.value, t, "fact"])
            if rng.random() < 0.6:                             # a look-alike: same field, someone else
                d = G.Fact(rng, f.kind, f.field)
                if d.ent["id"] not in f.ent["id"] and f.ent["id"] not in d.ent["id"] and d.value != f.value:
                    put(int(rng.integers(0, max(1, n_turns - 1))), d.stored()[1])
                    live.append([d, G4.your(d.stored()[1]), d.value, None, "lookalike"])
            if f.can_update() and rng.random() < 0.3:          # a correction later on (some by name)
                nm = f.ent["id"].split()[-1] if f.kind == "p" and f.ent["id"].split()[-1] in G.NAMES else None
                if nm and rng.random() < 0.5:
                    old = f.fmt; f.fmt = dict(old, **name_fmt(nm)); (_, new_s), _ = f.update(); f.fmt = old
                    new_ans = f._f(f.spec[1], f.new_value)
                else:
                    (_, new_s), _ = f.update(); new_ans = new_s
                tc = put(int(rng.integers(t + 1, max(t + 2, n_turns))), "Actually, " + new_s[0].lower() + new_s[1:])
                live[-1 if live[-1][0] is f else -2] = [f, G4.your(new_ans), f.new_value, tc, "corrected"]
        twostep = []
        if rng.random() < 0.5:
            rel, n = G.pick(rng, G.RELS), G.pick(rng, G.NAMES)
            f2 = G.Fact(rng, "p", None, dict(name_fmt(n), id=n))
            t1 = put(int(rng.integers(0, max(1, n_turns - 1))), G.pick(rng, [f"My {rel}'s name is {n}.", f"My {rel} is called {n}."]))
            t2 = put(int(rng.integers(0, max(1, n_turns - 1))), f2.stored()[1])
            old = f2.fmt; f2.fmt = dict(old, e=f"my {rel}", E=f"My {rel}", ep=f"my {rel}'s", Ep=f"My {rel}'s")
            q2 = f2._f(G.pick(rng, f2.spec[2])); f2.fmt = old
            old = f2.fmt; f2.fmt = dict(old, e=f"my {rel} {n}", E=f"My {rel} {n}", ep=f"my {rel} {n}'s", Ep=f"My {rel} {n}'s")
            a2 = G4.your(f2._f(f2.spec[1])); f2.fmt = old
            twostep.append((q2, a2, f2.value, max(t1, t2)))
        # question times: after the relevant statement, spread over the session
        qtimes = sorted(int(x) for x in rng.integers(max(2, n_turns // 10), n_turns + 1, size=n_q))
        out, turns, qi = [], [], 0
        last = max([n_turns] + list(events)) + 1
        for t in range(last + 1):
            while qi < len(qtimes) and qtimes[qi] == t:
                row = self._ask(t, turns, live, twostep)
                if row:
                    out.append(row)
                qi += 1
            if t >= last:
                break
            if t in events:
                u, r = events[t]; es = Memory.turn_entries(u, r, len(turns) + 1)
                mem.add_turn(u, r, len(turns) + 1, mem._vec([x for _, x in es]) if mem.dense is not None else None)
            else:
                j = int(rng.integers(len(self.filler))); u, r = self.filler[j]
                mem.add_turn(u, r, len(turns) + 1, self.filler_vecs[j])
            turns.append((u, r))
        return out

    def _ask(self, t, turns, live, twostep):
        rng = self.rng
        told = [x for x in live if x[3] is not None and x[3] < len(turns) and x[4] != "lookalike"]
        r = rng.random()
        absent = False
        if r < 0.45 and told:                                  # something said in this conversation (newest version)
            f, ans, val, _, kind = told[int(rng.integers(len(told)))]
            q = f.ask(); src = "conversation"
        elif r < 0.62:                                         # an uploaded skill fact
            x = self.skill_facts[int(rng.integers(len(self.skill_facts)))]
            q, ans, val, src = x["q"], (G4.your(x["a"]) if x.get("source") == "skill" else x["a"]), None, "uploaded"
        elif r < 0.70:                                         # a stored Q&A exchange, asked again
            x = self.qa_facts[int(rng.integers(len(self.qa_facts)))]
            q, ans, val, src = x["q"], x["a"], None, "uploaded"
        elif r < 0.75 and twostep and twostep[0][3] < len(turns):
            q, ans, val, _ = twostep[0]; src = "two-step"
        else:                                                  # never told (often with a look-alike told)
            f = G.Fact(rng); q, ans, val, src, absent = f.ask(), G.pick(rng, G.ABSTAIN), None, "never told", True
        q = vary(q, rng)
        qb = q.encode()
        # what the window holds without memory, then the real search for the rest
        kept = W.build(self.enc, self.sep, [], turns, qb, None, self.ctx, 1024, 300)[3]
        first_in = len(turns) - kept + (2 if kept else 1)
        items, found = self.mem.search(qb, [u for u, _ in turns[-2:]], first_in, k_facts=6, k_session=8)
        keep_turns = turns[-max(kept, 1) - 1:] if turns else []
        return {"q": q, "target": ans, "value": val, "source": src, "absent": absent, "n_turns": len(turns),
                "turns_offset": len(turns) - len(keep_turns),
                "turns": [[u.decode("utf-8", "replace"), r.decode("utf-8", "replace")] for u, r in keep_turns],
                "items": [[it.q.decode("utf-8", "replace"), it.a.decode("utf-8", "replace"), it.source, it.turn] for it in items]}


def build_longconv(n_sessions: int, shard: int, seed: int, facts_path=None, out_dir=None, dialogues=None) -> dict:
    import dense_keys
    import torch
    torch.set_num_threads(8)
    core = __import__("torch").load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
    enc = E.load(ROOT / core["encoding"]["dictionary"])
    rng = np.random.default_rng(seed * 100 + shard)
    t0 = time.time()
    maker = SessionMaker(enc, 2048, rng, dense_keys.DenseKeys("cpu", weight=2.0), shard, facts_path, dialogues)
    print(f"ready in {time.time() - t0:.0f} s: {len(maker.facts):,} uploaded facts, {len(maker.filler):,} filler turns", flush=True)
    out = (Path(out_dir) if out_dir else OUT) / f"longconv_s{seed}_{shard}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n_rows, n_turns_total = 0, 0
    with open(out, "w", encoding="utf-8") as f:
        for i in range(n_sessions):
            n = int(round(math.exp(rng.uniform(math.log(10), math.log(10000)))))
            rows = maker.session(n, n_q=int(rng.integers(8, 20)))
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_rows += len(rows); n_turns_total += n
            if (i + 1) % 50 == 0:
                print(f"{i + 1} sessions, {n_rows:,} question windows, {n_turns_total:,} turns, {time.time() - t0:.0f} s", flush=True)
    return {"file": out.name, "sessions": n_sessions, "windows": n_rows, "turns": n_turns_total, "seconds": round(time.time() - t0)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="facts"); ap.add_argument("--sessions", type=int, default=5000)
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--facts", default=None, help="a real pool, e.g. data_real/shared/look/facts_pool_real.jsonl")
    ap.add_argument("--out-dir", default=None); ap.add_argument("--dialogues", default=None)
    ap.add_argument("--vary", action="store_true", help="natural rewordings of 40% of the questions (19 Oct)")
    a = ap.parse_args()
    global VARY
    VARY = a.vary
    man_p = OUT / "manifest_v23.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    if a.only == "facts":
        man["facts"] = build_facts()
    else:
        res = build_longconv(a.sessions, a.shard, a.seed, a.facts and str(ROOT / a.facts), a.out_dir and str(ROOT / a.out_dir),
                             a.dialogues and str(ROOT / a.dialogues))
        if a.out_dir:                                       # a separate folder keeps its own manifest
            mp = ROOT / a.out_dir / "manifest_longconv.json"
            m2 = json.loads(mp.read_text()) if mp.exists() else {}
            m2[str(a.shard)] = res; mp.write_text(json.dumps(m2, indent=1))
            print(json.dumps(res)); return 0
        man.setdefault("longconv", {})[str(a.shard)] = res
    man_p.write_text(json.dumps(man, indent=1))
    print(json.dumps(man)[:800])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
