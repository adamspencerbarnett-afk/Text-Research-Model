"""The chat22 test suite: one runner, one scorer (scoring.py), the same cases for every model, results with 95% intervals.

    python -m chat22.tests.run_suite MODEL.pt [--tier quick|full] [--device cuda]

Works for chat22 models and for the earlier 5M chat models (arm A, v2.1) through the old engine, so they are compared
on identical cases. Sections:

  text      bits per byte of held-out clean wiki and books (first 400 KB each), the model's full output mixture
  skills    the held-out skills test v2 (200) and the rewriting test (40); facts are told as conversation turns in
            order, then each question is asked; clarify items are followed up and taken back
  probe     the 32 everyday questions; 'lookup' sections (stored, knowledge, arithmetic) reported apart from the rest
  convo     generated conversation probe (120, reserved names; correction/select with the wrong-value veto) and the
            27 real-phrased conversations, played live
  long      paired grid: 6 tasks x 512/1,024/1,900/4,096 codes, depth spread, the SAME case with memory on and off
  memory    any-length conversations: 100 / 1,000 / 10,000 real turns with 30 planted facts and 10 never told;
            recall by age, wrong value, abstain precision, lookup and reply time
  story     story worlds with reserved names: questions about earlier scenes from memory, and written scenes checked
            against the story notes (contradictions per scene)
  cache     (chat22 only) the same conversation with the history cache on and off must give identical replies
  latency   first-code time p50 / p95 over the 1,000-turn conversation

Writes results/suite_<model>.json; the guard (gate22.py) reads only these files."""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler", ROOT):
    sys.path.insert(0, str(p))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402

from chat22.tests import scoring as S  # noqa: E402


# ---------------------------------------------------------------------- one interface for old and new engines
class Chat22Adapter:
    def __init__(self, path, device, opts=None):
        from chat22.engine import Engine
        opts = dict(opts or {})
        search, judge, facts = opts.pop("search", "v1"), opts.pop("judge", None), opts.pop("facts", None)
        self.verify = int(opts.pop("verify", 0))
        self.e = Engine(path, device, log=lambda *x: None, memory_opts=opts, search=search, judge_path=judge, facts_path=facts)
        self.enc, self.ctx, self.kind = self.e.enc, self.e.ctx, "chat22"

    def reply(self, msg: str, n=80, mem=True):
        t, info = self.e.reply(msg.encode(), n, use_memory=mem, verify=self.verify)
        return t.decode("utf-8", "replace"), {"turns_in_window": info["turns_in_window"], "evidence": info["items"],
                                              "first_code_ms": info["first_code_ms"], "seconds": info["seconds"]}

    def remember(self, u: str, r: str):
        self.e.remember(u.encode(), r.encode())

    def undo(self):
        self.e.undo_last()

    def forget(self):
        self.e.forget()

    def add_fact(self, q, a):
        self.e.add_fact(q.encode(), a.encode())

    def nll_bits(self, x, y):
        m = self.e.model
        r = m.score(x, y, torch.zeros_like(x, dtype=torch.bool), loo=torch.zeros(x.shape[0], device=x.device))
        return r["nll"]


class OldAdapter:
    """The earlier chat engine (compiler/experiments/chat_engine.py) behind the same interface."""

    def __init__(self, path, device):
        import chat_engine as CE
        self.e = CE.ChatEngine(str(path), native=str(ROOT / "data" / "cv0.exe"), device=device, log=lambda *x: None)
        self.enc, self.ctx, self.kind = self.e.enc, self.e.a.ctx, "old"

    def reply(self, msg, n=80, mem=True):
        t, info = self.e.reply(msg.encode(), n, use_memory=mem)
        return t.decode("utf-8", "replace"), {"turns_in_window": info["turns_in_window"], "evidence": info["evidence"],
                                              "first_code_ms": info["first_code_ms"], "seconds": info["seconds"]}

    def remember(self, u, r):
        self.e.remember(u.encode(), r.encode())

    def undo(self):
        e = self.e
        if e.history:
            e.history.pop()
            if len(e.mem.exchanges) > e.base_n:          # the last session entry (one per turn in the old engine)
                e.mem.exchanges.pop(); k = e.mem.keys.pop()
                if e.mem.exact.get(k) == len(e.mem.exchanges):
                    del e.mem.exact[k]
                for w in k:
                    e.mem.index[w].pop(); e.mem.df[w] -= 1
                if e.mem.learned is not None:
                    e.mem.learned.vectors = e.mem.learned.vectors[:-1]

    def forget(self):
        self.e.forget()

    def add_fact(self, q, a):                             # only right after forget(): becomes stored knowledge
        self.e.add(q.encode(), a.encode()); self.e.base_n = len(self.e.mem.exchanges)

    def nll_bits(self, x, y):
        return self.e.model.nll(x, y, torch.zeros_like(x, dtype=torch.bool), torch.zeros(x.shape[0], device=x.device))


def adapter(path, device, opts=None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return Chat22Adapter(path, device, opts) if "config" in ck else OldAdapter(path, device)


# ---------------------------------------------------------------------- sections
@torch.no_grad()
def text_section(ad, device):
    out = {}
    for name in ("wiki", "books"):
        data = (ROOT / "data_chat22" / "eval_clean" / f"{name}.txt").read_bytes()[:400_000]
        codes, lens = ad.enc.encode(data)
        T = ad.ctx; tot_bits, tot_bytes = 0.0, 0
        dev = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        for s in range(0, len(codes) - 1, T):
            ch = codes[s:s + T + 1]
            if len(ch) < 64:
                break
            x = torch.from_numpy(ch[None, :-1].astype(np.int64)).to(dev); y = torch.from_numpy(ch[None, 1:].astype(np.int64)).to(dev)
            nll = ad.nll_bits(x, y).float()
            tot_bits += float(nll.sum()) / math.log(2); tot_bytes += int(lens[s + 1:s + len(ch)].sum())
        out[name] = round(tot_bits / max(1, tot_bytes), 4)
    return out


def skills_section(ad, limit=None, mode="told"):
    """mode 'stored': the facts are stored knowledge (as the earlier test did); 'told': the user said them earlier in
    this conversation, in order (as chat22 trains). Both are run for every model, so neither format is favoured."""
    res = {}
    for name, file in (("heldout_v2", "test_v2.json"), ("rewrite", "test_v4_rewrite.json")):
        t = json.loads((ROOT / "data_skills" / file).read_text())
        ad.forget()
        for q, a in t["facts"]:
            if mode == "stored":
                ad.add_fact(q, a)
            else:
                ad.remember(q, a)
        groups, wrong_ref = {}, 0
        items = t["items"][:limit] if limit else t["items"]
        for group, q, kw, extra in items:
            ans, _ = ad.reply(q, 80)
            if group == "unknown":
                ok = S.abstained(ans)
            elif group == "clarify":
                ok = S.asks(ans) and S.right(ans, kw)
                ad.remember(q, ans)
                fa, _ = ad.reply(extra["followup"], 80)
                ad.undo()
                g = groups.setdefault("followup", [0, 0]); g[0] += S.right(fa, extra["followup_kw"]); g[1] += 1
            else:
                ok = S.right(ans, kw) and not S.abstained(ans)
                fm = extra.get("format")
                if fm == "short":
                    ok = ok and len(ans.split()) <= len(extra["value"].split()) + 3
                elif fm in ("yes", "no"):
                    ok = ok and S.norm(ans).strip().startswith(fm)
                elif fm == "list":
                    ok = ok and (len(ans.splitlines()) >= 2 or ans.count("- ") >= 2 or ans.count(",") >= 1)
                wrong_ref += S.abstained(ans)
            g = groups.setdefault(group, [0, 0]); g[0] += bool(ok); g[1] += 1
        k = sum(v[0] for v in groups.values() if v is not groups.get("followup")); n = sum(v[1] for v in groups.values() if v is not groups.get("followup"))
        res[name] = {"total": S.rate(k, n), "groups": {g: S.rate(*v) for g, v in groups.items()}, "wrong_refusals": wrong_ref}
        print("skills", mode, name, res[name]["total"], flush=True)
    ad.forget()
    return res


def probe_section(ad):
    import chat_probe as CP
    ad.forget()
    for q, a in CP.TEACH:
        ad.add_fact(q, a)
    lookup, other = [0, 0], [0, 0]
    by = {}
    for sec, q, kw in CP.PROBES:
        ans, _ = ad.reply(q, 100)
        ok = S.right(ans, kw)
        bucket = lookup if sec in ("stored", "knowledge", "arithmetic") else other
        bucket[0] += ok; bucket[1] += 1
        b = by.setdefault(sec, [0, 0]); b[0] += ok; b[1] += 1
    ad.forget()
    return {"lookup": S.rate(*lookup), "not_lookup": S.rate(*other), "by_section": {k: S.rate(*v) for k, v in by.items()}}


def convo_section(ad):
    import mt_probe as MP
    import mt_probe2 as MP2
    out = {}
    convs = MP2.build(31)
    groups = {}
    for group, turns, ask, want, wrong in convs:
        ad.forget()
        for u in turns:
            r, _ = ad.reply(u, 60); ad.remember(u, r or "Okay.")
        ans, _ = ad.reply(ask, 60)
        g = groups.setdefault(group, [0, 0]); g[0] += S.right(ans, want, wrong); g[1] += 1
    out["generated"] = {"total": S.rate(sum(v[0] for v in groups.values()), sum(v[1] for v in groups.values())),
                        "groups": {g: S.rate(*v) for g, v in groups.items()}}
    groups = {}
    for group, turns, ask, want in MP.CONVERSATIONS:
        ad.forget()
        for u in turns:
            r, _ = ad.reply(u, 60); ad.remember(u, r or "Okay.")
        ans, _ = ad.reply(ask, 60)
        g = groups.setdefault(group, [0, 0]); g[0] += S.right(ans, want); g[1] += 1
    out["real_phrasing"] = {"total": S.rate(sum(v[0] for v in groups.values()), sum(v[1] for v in groups.values())),
                            "groups": {g: S.rate(*v) for g, v in groups.items()}}
    ad.forget()
    print("convo", out["generated"]["total"], out["real_phrasing"]["total"], flush=True)
    return out


def long_section(ad, n_per, lengths=(512, 1024, 1900, 4096)):
    """Paired: each case is run with memory on and off on the same conversation."""
    import long_grid as LG

    class _E:                                               # what long_grid.Grid needs: .enc
        enc = ad.enc
    G3.setup(test=True)
    grid = LG.Grid(_E, np.random.default_rng(11))
    rows = {}
    ground = {"right": 0, "copied_wrong": 0, "refused": 0, "invented": 0}   # memory-on answers (the grounding score)
    for task in LG.TASKS:
        for L in lengths:
            for mem in (False, True):
                rows[(task, L, mem)] = [0, 0, 0]          # right, wrong evidence, n
            for j in range(n_per):
                depth = (0.1, 0.5, 0.9)[j % 3]
                turns, q, want, wrong, absent, stating = grid.case(task, L, depth)
                for mem in (False, True):
                    ad.forget()
                    for u, r in turns:
                        ad.remember(u.decode("utf-8", "replace"), r.decode("utf-8", "replace"))
                    ans, info = ad.reply(q.decode(), 60, mem)
                    ok = (S.abstained(ans) and not any(S.has(ans, w) for w in wrong)) if absent else S.right(ans, [[w] for w in want], wrong)
                    rr = rows[(task, L, mem)]; rr[0] += ok; rr[1] += any(S.has(ans, w) for w in wrong); rr[2] += 1
                    if mem:
                        ground[grounding(ans, ok, absent, wrong)] += 1
        print("long", task, {f"{L}{'m' if m else ''}": f"{v[0]}/{v[2]}" for (t, L, m), v in rows.items() if t == task}, flush=True)
    ad.forget()
    out = {f"{t}|{L}|{'memory' if m else 'window'}": {**S.rate(v[0], v[2]), "wrong_evidence": v[1]} for (t, L, m), v in rows.items()}
    summ = {}
    for mem in (False, True):
        for t in LG.TASKS:
            k = sum(v[0] for (tt, _, m), v in rows.items() if tt == t and m == mem); n = sum(v[2] for (tt, _, m), v in rows.items() if tt == t and m == mem)
            summ[f"{t}|{'memory' if mem else 'window'}"] = S.rate(k, n)
    return {"cells": out, "by_task": summ, "grounding": ground}


def grounding(ans, ok, absent, wrong) -> str:
    """One memory-on answer: right; copied a wrong memory (a look-alike or an outdated value); refused although the fact
    was told; or invented (an answer that is neither right nor a wrong memory). The facts are invented, so a right answer
    can only have come from the window or memory."""
    if ok:
        return "right"
    if any(S.has(ans, w) for w in wrong if w):
        return "copied_wrong"
    if S.abstained(ans):
        return "refused" if not absent else "right"
    return "invented"


def memory_section(ad, sizes=(100, 1000, 10000), n_facts=30, n_absent=10):
    import chat_tests as CT
    G3.setup(test=True)
    rng = np.random.default_rng(7)
    pairs = CT.heldout_pairs()
    res, lat = {}, []
    for N in sizes:
        ad.forget()
        facts = [G.Fact(rng) for _ in range(n_facts)]
        at = sorted(int(t) for t in rng.choice(N, n_facts, replace=False))
        fi = 0; t0 = time.perf_counter()
        for t in range(N):
            if fi < n_facts and t == at[fi]:
                ad.remember(facts[fi].stored()[1], "Got it. I'll remember that."); fi += 1
            else:
                u, r = pairs[int(rng.integers(len(pairs)))]
                ad.remember(u.decode("utf-8", "replace")[:600], r.decode("utf-8", "replace")[:700])
        store_s = time.perf_counter() - t0
        ages = {"recent (<10% back)": [0, 0], "middle": [0, 0], "old (>90% back)": [0, 0]}
        ok_all, wrong, secs = 0, 0, []
        ground = {"right": 0, "copied_wrong": 0, "refused": 0, "invented": 0}
        for f, t in zip(facts, at):
            ans, info = ad.reply(f.ask(), 60)
            secs.append(info["seconds"]); lat.append(info["first_code_ms"]) if N == 1000 else None
            ok = S.right(ans, [[f.value.lower()]]) and not S.abstained(ans)
            age = (N - t) / N
            b = ages["recent (<10% back)" if age < 0.1 else ("old (>90% back)" if age > 0.9 else "middle")]
            b[0] += ok; b[1] += 1; ok_all += ok; wrong += not ok and not S.abstained(ans)
            ground[grounding(ans, ok, False, [])] += 1
        abst_ok = 0
        for _ in range(n_absent):
            g = G.Fact(rng)
            ans, _ = ad.reply(g.ask(), 60)
            abst_ok += S.abstained(ans)
            ground[grounding(ans, S.abstained(ans), True, [])] += 1
        res[str(N)] = {"recall": S.rate(ok_all, n_facts), "by_age": {k: S.rate(*v) for k, v in ages.items()},
                       "wrong_value": wrong, "never_told_abstained": S.rate(abst_ok, n_absent), "grounding": ground,
                       "store_seconds": round(store_s, 1), "reply_seconds_p50": round(float(np.median(secs)), 3)}
        print("memory", N, res[str(N)]["recall"], res[str(N)]["never_told_abstained"], flush=True)
    ad.forget()
    lat = [x for x in lat if x is not None]
    return res, ({"first_code_ms_p50": round(float(np.percentile(lat, 50))), "first_code_ms_p95": round(float(np.percentile(lat, 95)))} if lat else {})


def story_section(ad, n_worlds=25):
    from chat22.data import story_world as SW
    G3.setup(test=True)
    rng = np.random.default_rng(13)
    q_ok, q_n, contra, scenes, kept, kept_n = 0, 0, 0, 0, 0, 0
    for _ in range(n_worlds):
        w = SW.World(rng)
        ad.forget()
        for c in range(len(w.chars)):
            for f in w.bible_facts(c):
                ad.remember("Story notes.", f)
        for _ in range(int(rng.integers(4, 9))):
            text, _ = w.next_scene()
            ad.remember("Continue the story.", text)
        for j in rng.choice(len(w.scenes), min(3, len(w.scenes)), replace=False):
            q, a = w.question(int(j))
            ans, _ = ad.reply(q, 60)
            key = [x for x in a.rstrip(".").split() if x[:1].isupper() or len(x) > 3][-1].lower()
            q_ok += S.has(ans, key); q_n += 1
        ans, _ = ad.reply("Continue the story.", 160)
        scenes += 1
        # a written scene that names a character must not change its facts. 19 Oct: judged per sentence, for the one
        # character named in that sentence, and an item that belongs to another character of the story is not a
        # contradiction (the old count charged a character with the other character's city, pet or object)
        sents = [x for x in re.split(r"(?<=[.!?])\s+", ans) if x.strip()]
        for ch in w.chars:
            if ch["name"].lower() in S.norm(ans):
                kept_n += 1; kept += S.has(ans, ch["trait"]) or S.has(ans, ch["role"])
        for sent in sents:
            named = [ch for ch in w.chars if ch["name"].lower() in S.norm(sent)]
            if len(named) != 1:
                continue
            ch = named[0]
            for field, others in (("home", G.CITIES), ("thing", SW.THINGS), ("pet", SW.PETS)):
                theirs = {c[field] for c in w.chars if c is not ch}
                key = lambda o: o.split()[-1] if field != "home" else o
                said = [o for o in others if o != ch[field] and o not in theirs and S.has(sent, key(o))]
                if said and not S.has(sent, key(ch[field])):
                    contra += 1
    ad.forget()
    return {"questions": S.rate(q_ok, q_n), "contradictions_per_scene": round(contra / max(1, scenes), 3),
            "character_facts_used": S.rate(kept, kept_n)}


def cache_section(ad):
    if ad.kind != "chat22":
        return None
    from chat22.tests.test_engine import conversation
    ad.e.forget(); ad.e.reuse = True; on = conversation(ad.e)
    ad.e.forget(); ad.e.reuse = False; off = conversation(ad.e)
    ad.e.reuse = True; ad.e.forget()
    same = sum(1 for x, y in zip(on, off) if x[0] == "reused" or x[:3] == y[:3])
    return {"identical_turns": same, "turns": len(on)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--tier", default="full", choices=["quick", "full"]); ap.add_argument("--device", default="cuda")
    ap.add_argument("--engine-opts", default="", help='chat22 memory options as JSON, e.g. {"gate_facts": [0.35, 0.7], "k_facts": 2}')
    ap.add_argument("--sections", default="", help="comma list to run only some sections (text,skills,skills_stored,probe,convo,long,memory,story,cache)")
    ap.add_argument("--tag", default="", help="suffix of the result file")
    a = ap.parse_args()
    path = ROOT / a.model if not Path(a.model).is_absolute() else Path(a.model)
    t0 = time.time()
    ad = adapter(path, a.device, json.loads(a.engine_opts) if a.engine_opts else None)
    want = set(a.sections.split(",")) if a.sections else None
    run = lambda name: want is None or name in want
    quick = a.tier == "quick"
    res = {"model": a.model, "engine": ad.kind, "tier": a.tier, "date": time.strftime("%Y-%m-%d %H:%M"),
           "retrieval": "word key + MiniLM dense key (outside reference)"}
    res["engine_opts"] = a.engine_opts or None
    res["clean_store_after_stored_skills"] = True
    if run("text"):
        res["text"] = text_section(ad, a.device); print("text", res["text"], flush=True)
    if run("skills"):
        res["skills"] = skills_section(ad, 40 if quick else None, "told")
    if run("skills_stored"):
        res["skills_stored"] = skills_section(ad, 40 if quick else None, "stored")
        # 18 Oct: the stored-skills test adds ~180 facts to the permanent store and forget() keeps them, so every later
        # section was tested with them; a fresh engine restores the model's own store
        ad = adapter(path, a.device, json.loads(a.engine_opts) if a.engine_opts else None)
    if run("probe"):
        res["probe"] = probe_section(ad); print("probe", res["probe"]["lookup"], res["probe"]["not_lookup"], flush=True)
    if run("convo"):
        res["convo"] = convo_section(ad)
    if run("long"):
        res["long"] = long_section(ad, 6 if quick else 51, (512, 1900) if quick else (512, 1024, 1900, 4096))
    if run("memory"):
        res["memory"], res["latency"] = memory_section(ad, (100,) if quick else (100, 1000, 10000))
    g = {"right": 0, "copied_wrong": 0, "refused": 0, "invented": 0}
    for part in ([res["long"]["grounding"]] if "long" in res else []) + [m.get("grounding", {}) for m in (res.get("memory") or {}).values()]:
        for k, v in part.items():
            g[k] += v
    if sum(g.values()):
        tot = sum(g.values())
        res["grounding"] = {**g, "n": tot, "right_share": round(g["right"] / tot, 3), "invented_share": round(g["invented"] / tot, 3),
                            "copied_wrong_share": round(g["copied_wrong"] / tot, 3)}
        print("grounding", res["grounding"], flush=True)
    if run("story"):
        res["story"] = story_section(ad, 6 if quick else 25); print("story", res["story"], flush=True)
    if run("cache"):
        res["cache"] = cache_section(ad); print("cache", res["cache"], flush=True)
    res["seconds"] = round(time.time() - t0)
    out = ROOT / "results" / f"suite_{path.stem}{'_quick' if quick else ''}{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print("saved", out, f"({res['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
