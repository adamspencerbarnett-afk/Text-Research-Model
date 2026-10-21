"""Tests added 20 October 2026 for the 5M loop (recipes/LOOP.json): the chat abilities the suite did not measure.
Run as run_suite sections (only when named): --sections afterno,phrasing,probe2,reason,twomem,open --tag _ev2new.

  afterno   every answerable question directly after a correct refusal of a never-told question: arithmetic, a told
            fact, a stored real fact, a told fact after two refusals; control: the same questions with no refusal.
  phrasing  200 told-and-asked facts in natural chat wording (lower case, no question mark, filler words, chat turns
            between), with wraps that no training generator uses; test-reserved names and ways of asking.
  probe2    the 32-question chat probe with its stored and reworded questions rewritten for the real-facts store
            (facts checked to be in data_real/shared/knowledge/knowledge_real.jsonl); other sections unchanged.
  reason    over the conversation: counting what was told (1-4 items), the first thing told, a summary of 3-4 facts.
  twomem    a told personal fact plus a stored real fact ("My favorite song is X." -> "Who wrote my favorite song?",
            answered from the store's "Who wrote the song X?"). Rows come from the store's test split (md5 % 10 == 0),
            which twomem training never uses. Control: the store question asked directly.
  multi     messages with several parts: three facts told at once, two asked at once, one known + one unknown, a
            statement with a question, two stored real facts, a quoted note with two questions (all parts must be right)
  open      open-ended requests from held-out Alpaca (explain, describe, how, why, suggest...) against the reference
            answer: content-word F1, a complete last sentence, repetition, length; 'good' needs all four."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np

from chat22.tests import scoring as S

ROOT = Path(__file__).resolve().parents[2]
ACK = "Got it. I'll remember that."
NUMW = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _turn(ad, msg, n=80):
    ans, _ = ad.reply(msg, n)
    ad.remember(msg, ans or "Okay.")
    return ans


def _fillers(ad, rng, pairs, k):
    for _ in range(k):
        u, r = pairs[int(rng.integers(len(pairs)))]
        ad.remember(u.decode("utf-8", "replace")[:600], r.decode("utf-8", "replace")[:700])


def _arith(rng):
    a, b = int(rng.integers(11, 60)), int(rng.integers(2, 40))
    op = rng.integers(3)
    return [(f"What is {a} plus {b}?", a + b), (f"What is {a} minus {b}?", a - b), (f"What is {a} times {b % 12 + 2}?", a * (b % 12 + 2))][op]


# ---------------------------------------------------------------------- afterno
def afterno_section(ad, n=40):
    """Every answerable question comes DIRECTLY after a correct refusal (Adam's case, 19 Oct: "What is my brother's name?"
    -> refused, then "What is 12 plus 30?" was refused too): arithmetic, a told fact, a stored real fact, and a told fact
    after two refusals in a row. Control: the same four questions in a fresh conversation with no refusal."""
    import chat_tests as CT
    import gen_skills as G
    import gen_skills_v3 as G3
    G3.setup(test=True)
    rng = np.random.default_rng(21)
    pairs = CT.heldout_pairs()
    store = twomem_cases("test")
    types = ("arithmetic", "told fact", "stored fact", "after two refusals")
    after = {t: [0, 0] for t in types}; control = {t: [0, 0] for t in types}; refused = [0, 0]
    for _ in range(n):
        f1 = G.Fact(rng); f2 = G._other(rng, {f1.ent["id"]})
        nt = [G._other(rng, {f1.ent["id"], f2.ent["id"]}) for _ in range(4)]
        aq, av = _arith(rng)
        _, _, sq, sa = store[int(rng.integers(len(store)))]
        sq = sq[0].upper() + sq[1:]
        asks = {"arithmetic": (aq, str(av)), "told fact": (f1.ask(), f1.value), "stored fact": (sq, sa), "after two refusals": (f2.ask(), f2.value)}
        k = int(rng.integers(0, 3))
        ad.forget(); ad.remember(f1.stored()[1], ACK); ad.remember(f2.stored()[1], ACK); _fillers(ad, rng, pairs, k)
        for t in types:
            q, v = asks[t]
            control[t][0] += S.right(_turn(ad, q).replace("\xa0", " "), [[v]]); control[t][1] += 1
        ad.forget(); ad.remember(f1.stored()[1], ACK); ad.remember(f2.stored()[1], ACK); _fillers(ad, rng, pairs, k)
        order = [(nt[0], "arithmetic"), (nt[1], "told fact"), (nt[2], "stored fact"), (nt[3], None), (None, "after two refusals")]
        for ntf, t in order:
            if ntf is not None:
                refused[0] += S.abstained(_turn(ad, ntf.ask())); refused[1] += 1
                if t is None:
                    continue
            q, v = asks[t]
            after[t][0] += S.right(_turn(ad, q).replace("\xa0", " "), [[v]]); after[t][1] += 1
    ad.forget()
    tot = lambda d: S.rate(sum(x[0] for x in d.values()), sum(x[1] for x in d.values()))
    out = {"answered_after_refusal": tot(after), "control_no_refusal": tot(control), "never_told_refused": S.rate(*refused),
           "after_by_type": {t: S.rate(*v) for t, v in after.items()}, "control_by_type": {t: S.rate(*v) for t, v in control.items()}}
    print("afterno", out["answered_after_refusal"], "control", out["control_no_refusal"], "by type", {t: v["k"] for t, v in out["after_by_type"].items()}, flush=True)
    return out


# ---------------------------------------------------------------------- phrasing
TELL = ["oh btw, {sl}", "Just so you know, {sl}", "{s} Can you keep that in mind?", "fyi {sl}", "Quick note: {s}",
        "Something about me: {s}", "I should mention that {sl}", "Heads up, {sl}", "{sn} haha", "ok so {sl}",
        "{s} Just wanted you to know.", "Random fact: {s}"]
ASK = ["hmm, {ql}", "wait, {ql}", "{q} I totally blanked on it.", "ok so {ql}", "um, {ql}", "{qn} again?", "real quick, {ql}",
       "any idea {qln}?", "{q} You should know this one.", "Help me out here: {ql}", "{qn}, what was it?", "so {ql}"]


def _low(s):
    return s[0].lower() + s[1:] if s else s


def phrasing_section(ad, n=200):
    import chat_tests as CT
    import gen_skills as G
    import gen_skills_v3 as G3
    G3.setup(test=True)
    rng = np.random.default_rng(22)
    pairs = CT.heldout_pairs()
    ok, wrong_refusals, groups = 0, 0, {"plain_case": [0, 0], "lower_case": [0, 0], "with_chat_between": [0, 0], "no_chat_between": [0, 0]}
    for _ in range(n):
        f = G.Fact(rng)
        s = f.stored()[1]
        tell = G.pick(rng, TELL).format(s=s, sl=_low(s), sn=s.rstrip("."))
        q = f.ask()
        ask = G.pick(rng, ASK).format(q=q, ql=_low(q), qn=q.rstrip("?"), qln=_low(q).rstrip("?"))
        lower = rng.random() < 0.3
        if lower:
            tell, ask = tell.lower(), ask.lower()
        if rng.random() < 0.2:
            ask = ask.rstrip("?")
        ad.forget()
        r0, _ = ad.reply(tell, 40); ad.remember(tell, r0 or "Okay.")
        k = int(rng.integers(0, 7)); _fillers(ad, rng, pairs, k)
        ans, _ = ad.reply(ask, 80)
        good = S.right(ans, [[f.value]])
        ok += good; wrong_refusals += (not good) and S.abstained(ans)
        for g in (("lower_case" if lower else "plain_case"), ("with_chat_between" if k else "no_chat_between")):
            groups[g][0] += good; groups[g][1] += 1
    ad.forget()
    out = {"total": S.rate(ok, n), "wrong_refusals": int(wrong_refusals), "groups": {g: S.rate(*v) for g, v in groups.items()}}
    print("phrasing", out["total"], "wrong refusals", wrong_refusals, flush=True)
    return out


# ---------------------------------------------------------------------- probe2
PROBES2_NEW = [
    ("stored", "What is the name of the biggest city in nigeria?", [["lagos"]]),
    ("stored", "Who holds the record for most premier league goals?", [["shearer"]]),
    ("stored", "Who has won the most wimbledon men's singles titles?", [["federer"]]),
    ("stored", "Who wrote the song we shall behold him?", [["rambo"]]),
    ("stored", "What is the capital of Australia?", [["canberra"]]),
    ("reworded", "Which city in Nigeria is the largest?", [["lagos"]]),
    ("reworded", "Who has scored more Premier League goals than anyone else?", [["shearer"]]),
    ("reworded", "Which man has won the Wimbledon singles title the most times?", [["federer"]]),
    ("reworded", "Who is the songwriter of We Shall Behold Him?", [["rambo"]]),
    ("reworded", "Which city is Australia's capital?", [["canberra"]]),
]


def probe2_section(ad):
    import chat_probe as CP
    probes = PROBES2_NEW + [p for p in CP.PROBES if p[0] not in ("stored", "reworded")]
    ad.forget()
    for q, a in CP.TEACH:
        ad.add_fact(q, a)
    lookup, other, by = [0, 0], [0, 0], {}
    for sec, q, kw in probes:
        ans, _ = ad.reply(q, 100)
        ok = S.right(ans, kw)
        bucket = lookup if sec in ("stored", "knowledge", "arithmetic") else other
        bucket[0] += ok; bucket[1] += 1
        b = by.setdefault(sec, [0, 0]); b[0] += ok; b[1] += 1
    ad.forget()
    out = {"lookup": S.rate(*lookup), "not_lookup": S.rate(*other), "by_section": {k: S.rate(*v) for k, v in by.items()}}
    print("probe2", out["lookup"], out["not_lookup"], flush=True)
    return out


# ---------------------------------------------------------------------- reason
def _count_case(rng, G):
    cat = rng.integers(3); k = int(rng.integers(1, 5))
    if cat == 0:
        kinds = list(rng.choice(G.PETS, k, replace=False)); names = list(rng.choice(G.PETNAMES, k, replace=False))
        stm = [f"I have {G._art(p)} {p} named {nm}." for p, nm in zip(kinds, names)]
        ask = G.pick(rng, ["How many pets do I have?", "How many pets did I tell you about?"])
    elif cat == 1:
        cars = [f"{c} {m}" for c, m in zip(rng.choice(G.COLOURS, k, replace=False), rng.choice(G.CARS, k, replace=False))]
        stm = [f"I own {G._art(c)} {c}." for c in cars]
        ask = G.pick(rng, ["How many cars do I own?", "How many cars did I say I have?"])
    else:
        names = list(rng.choice(G.NAMES, k, replace=False))
        stm = [f"My {G.pick(rng, ['son', 'daughter'])} {nm} is {int(rng.integers(2, 17))} years old." for nm in names]
        ask = G.pick(rng, ["How many children do I have?", "How many kids did I tell you about?"])
    return stm, ask, [[str(k), NUMW[k]]]


def reason_section(ad, n_count=30, n_order=30, n_sum=20):
    import chat_tests as CT
    import gen_skills as G
    import gen_skills_v3 as G3
    G3.setup(test=True)
    rng = np.random.default_rng(23)
    pairs = CT.heldout_pairs()
    cnt, order, summ, cover = [0, 0], [0, 0], [0, 0], []
    for _ in range(n_count):
        stm, ask, want = _count_case(rng, G)
        ad.forget()
        for s in stm:
            ad.remember(s, ACK); _fillers(ad, rng, pairs, int(rng.integers(0, 2)))
        cnt[0] += S.right(_turn(ad, ask), want); cnt[1] += 1
    for _ in range(n_order):
        a = G.Fact(rng); b = G._other(rng, {a.ent["id"]}, (a.field,))
        ad.forget(); ad.remember(a.stored()[1], ACK); _fillers(ad, rng, pairs, int(rng.integers(0, 2))); ad.remember(b.stored()[1], ACK)
        ans = S.norm(_turn(ad, G.pick(rng, ["What was the first thing I told you?", "What did I tell you first?"])))
        ia, ib = ans.find(a.value.lower()), ans.find(b.value.lower())
        order[0] += ia >= 0 and (ib < 0 or ia < ib); order[1] += 1
    for _ in range(n_sum):
        k = int(rng.integers(3, 5)); fs = [G.Fact(rng)]
        while len(fs) < k:
            fs.append(G._other(rng, {f.ent["id"] for f in fs}))
        ad.forget()
        for f in fs:
            ad.remember(f.stored()[1], ACK)
        ans = _turn(ad, G.pick(rng, ["Can you sum up what I've told you so far?", "Summarize what I told you.",
                                       "What have I told you about so far?"]), 200)
        got = sum(S.has(ans, f.value) for f in fs); cover.append(got / k)
        summ[0] += got >= k - 1; summ[1] += 1
    ad.forget()
    out = {"count": S.rate(*cnt), "first_told": S.rate(*order), "summary": S.rate(*summ),
           "summary_mean_coverage": round(float(np.mean(cover)), 3) if cover else None}
    print("reason", out, flush=True)
    return out


# ---------------------------------------------------------------------- twomem
TWOMEM = [
    (r"^who wrote the song (.+?)\??$", 'My favorite song is "{x}".', "Who wrote my favorite song?"),
    (r"^who sings the song (.+?)\??$", "I can't stop listening to \"{x}\".", "Who sings the song I can't stop listening to?"),
    (r"^where was the movie (.+?) filmed\??$", "My favorite movie is {X}.", "Where was my favorite movie filmed?"),
    (r"^when did the movie (.+?) come out\??$", "Last night I watched {X}.", "When did the movie I watched last night come out?"),
    (r"^what is the meaning of the name (.+?)\??$", "My daughter's name is {X}.", "What does my daughter's name mean?"),
    (r"^where does the last name (.+?) come from\??$", "My last name is {X}.", "Where does my last name come from?"),
    (r"^who is the current leader of (.+?)\??$", "I'm moving to {X} next year.", "Who is the current leader of the country I'm moving to?"),
    (r"^what is the name of the biggest city in (.+?)\??$", "My cousin lives in {X}.", "What is the name of the biggest city in the country my cousin lives in?"),
]


def store_split(q: str) -> str:
    """'test' for md5(question) % 10 == 0, else 'train' (twomem training reads only 'train' rows)."""
    return "test" if int(hashlib.md5(q.strip().lower().encode("utf-8")).hexdigest(), 16) % 10 == 0 else "train"


def twomem_cases(split: str, per_pattern: int | None = None, store=None) -> list:
    """(statement, chained question, direct store question, answer key) from the store rows of one split."""
    store = store or ROOT / "data_real" / "shared" / "knowledge" / "knowledge_real.jsonl"
    rx = [(re.compile(p, re.I), t, a) for p, t, a in TWOMEM]
    out, n = [], [0] * len(rx)
    for line in open(store, encoding="utf-8"):
        r = json.loads(line)
        if r.get("source") != "qa_memory":
            continue
        q, a = r["q"].strip(), r["a"].replace(" ", " ").strip().rstrip(".")
        if not a or len(a.split()) > 3 or store_split(q) != split:
            continue
        for j, (p, t, ask) in enumerate(rx):
            m = p.match(q)
            if m and (per_pattern is None or n[j] < per_pattern):
                x = m.group(1).strip()
                out.append((t.format(x=x, X=x.title() if x.islower() else x), ask, q, a)); n[j] += 1
                break
    return out


def twomem_section(ad, per_pattern=8):
    import chat_tests as CT
    rng = np.random.default_rng(24)
    pairs = CT.heldout_pairs()
    cases = twomem_cases("test", per_pattern)
    comb, direct = [0, 0], [0, 0]
    for stm, ask, q, a in cases:
        ad.forget(); ad.remember(stm, ACK); _fillers(ad, rng, pairs, int(rng.integers(0, 3)))
        comb[0] += S.right(_turn(ad, ask).replace(" ", " "), [[a]]); comb[1] += 1
        ad.forget()
        direct[0] += S.right(_turn(ad, q[0].upper() + q[1:]).replace(" ", " "), [[a]]); direct[1] += 1
    ad.forget()
    out = {"combined": S.rate(*comb), "direct_store_question": S.rate(*direct), "patterns": len(TWOMEM)}
    print("twomem", out, flush=True)
    return out


# ---------------------------------------------------------------------- open
OPEN_START = ("Explain", "Describe", "How do", "How can", "How to", "Why ", "Suggest", "Give me some", "Give some", "What are the benefits",
              "What are some ways", "Provide tips", "Give tips", "Discuss")
WORD = re.compile(r"[a-z0-9']+")
STOP = set("the a an is are was were of to in and or for on at by with as it its this that be can will your you i we they he she "
           "their our from but not have has had do does so if than then there these those which who what how why when".split())


def _cw(t):
    return [w for w in WORD.findall(t.lower()) if w not in STOP and len(w) > 2]


def open_section(ad, n=40):
    import sys
    sys.path.insert(0, str(ROOT))
    from chat22 import encoder as E
    ex = E.parse_exchanges((ROOT / "data_qa" / "heldout" / "qa_heldout.txt").read_bytes())
    pick = []
    for q, a in ex:
        qs, as_ = q.decode("utf-8", "replace"), a.decode("utf-8", "replace")
        if qs.startswith(OPEN_START) and "\n" not in qs and len(qs) < 200 and len(as_.split()) >= 25:
            pick.append((qs, as_))
    pick.sort(key=lambda t: hashlib.md5(t[0].encode()).hexdigest())
    pick = pick[:n]
    good, f1s, comp, reps, lens = 0, [], 0, [], []
    ad.forget()
    for q, ref in pick:
        ans, _ = ad.reply(q, 300); ad.forget()
        a_w, r_w = _cw(ans), _cw(ref)
        common = sum(min(a_w.count(w), r_w.count(w)) for w in set(a_w))
        f1 = 0.0 if not a_w or not r_w or not common else 2 * common / (len(a_w) + len(r_w))
        toks = WORD.findall(ans.lower())
        tri = [tuple(toks[i:i + 3]) for i in range(len(toks) - 2)]
        rep = 1 - len(set(tri)) / len(tri) if tri else 0.0
        complete = ans.strip()[-1:] in (".", "!", "?") if ans.strip() else False
        f1s.append(f1); reps.append(rep); lens.append(len(toks)); comp += complete
        good += f1 >= 0.15 and complete and rep < 0.15 and len(toks) >= 12
    out = {"good": S.rate(good, len(pick)), "mean_f1": round(float(np.mean(f1s)), 3), "complete": S.rate(comp, len(pick)),
           "mean_repetition": round(float(np.mean(reps)), 3), "mean_words": round(float(np.mean(lens)), 1)}
    print("open", out, flush=True)
    return out


# ---------------------------------------------------------------------- multi
T_JOIN3 = ["Here's some stuff about me. {A} {B} {C}", "{A} Plus, {b} And {c}"]
T_ASK2 = ["I have two questions. {Q1} {Q2}", "{Q1n}? Oh, and {q2}", "Quick ones: {Q1} {Q2}"]
T_STORE2 = ["I'm curious about two things. {Q1} {Q2}", "{Q1} Oh, and one more: {q2}"]
T_NOTE = ['My {who} just texted me: "{A} {B}" {Q1n}, and {q2}']
T_WITHQ = ["{S} Unrelated, but {q}", "{S} Also, quick question: {q}"]


def _lcw(s):
    w = s.split(" ", 1)[0]
    return s[0].lower() + s[1:] if w in {"My", "The", "Our", "Your", "Parking", "What", "When", "Where", "Which", "Who", "How", "Can", "Is", "Do", "Does"} else s


def multi_section(ad, n=20):
    """Messages with several parts, wordings no training generator uses: three facts told in one message then one asked
    (tell3); two told facts asked in one message (ask2); one told and one never told asked together (partial: the told
    value AND an "I don't have that"); a new statement plus a question (with_statement); two real stored facts asked
    together (store2, store test split); a quoted note with two facts and two questions (note2)."""
    import chat_tests as CT
    import gen_skills as G
    import gen_skills_v3 as G3
    G3.setup(test=True)
    rng = np.random.default_rng(25)
    pairs = CT.heldout_pairs()
    store = [(q[0].upper() + q[1:], a) for _, _, q, a in twomem_cases("test")]
    res = {t: [0, 0] for t in ("tell3", "ask2", "partial", "with_statement", "store2", "note2")}
    for i in range(n):
        fs = [G.Fact(rng)]
        while len(fs) < 3:
            fs.append(G._other(rng, {f.ent["id"] for f in fs}))
        st = [f.stored()[1] for f in fs]
        # tell3
        ad.forget()
        msg = G.pick(rng, T_JOIN3).format(A=st[0], B=st[1], C=st[2], b=_lcw(st[1]), c=_lcw(st[2]))
        _turn(ad, msg, 40); _fillers(ad, rng, pairs, int(rng.integers(0, 3)))
        f = fs[int(rng.integers(3))]
        res["tell3"][0] += S.right(_turn(ad, f.ask()), [[f.value]]); res["tell3"][1] += 1
        # ask2
        ad.forget(); ad.remember(st[0], ACK); ad.remember(st[1], ACK); _fillers(ad, rng, pairs, int(rng.integers(0, 2)))
        q1, q2 = fs[0].ask(), fs[1].ask()
        q = G.pick(rng, T_ASK2).format(Q1=q1, Q2=q2, q2=_lcw(q2), Q1n=q1.rstrip("?"))
        res["ask2"][0] += S.right(_turn(ad, q), [[fs[0].value], [fs[1].value]]); res["ask2"][1] += 1
        # partial
        ad.forget(); ad.remember(st[0], ACK)
        q1, q2 = (fs[0].ask(), fs[2].ask()) if i % 2 == 0 else (fs[2].ask(), fs[0].ask())
        ans = _turn(ad, G.pick(rng, T_ASK2).format(Q1=q1, Q2=q2, q2=_lcw(q2), Q1n=q1.rstrip("?")))
        says_unknown = S.abstained(ans) or re.search(r"don'?t have [^.?!]{0,60} in my memory", S.norm(ans)) is not None
        res["partial"][0] += S.right(ans, [[fs[0].value]]) and says_unknown; res["partial"][1] += 1   # 20 Oct: "I don't have X's birthday in my memory" counts
        # with_statement
        ad.forget(); ad.remember(st[0], ACK)
        ans = _turn(ad, G.pick(rng, T_WITHQ).format(S=st[1], q=_lcw(fs[0].ask())))
        res["with_statement"][0] += S.right(ans, [[fs[0].value]]); res["with_statement"][1] += 1
        # store2
        ad.forget()
        (sq1, sa1), (sq2, sa2) = [store[int(j)] for j in rng.choice(len(store), 2, replace=False)]
        ans = _turn(ad, G.pick(rng, T_STORE2).format(Q1=sq1, Q2=sq2, q2=_lcw(sq2)), 100).replace("\xa0", " ")
        res["store2"][0] += S.right(ans, [[sa1], [sa2]]); res["store2"][1] += 1
        # note2
        ad.forget()
        ev = [G.Fact(rng, "t"), G.Fact(rng, "t")]
        while ev[1].ent["id"] == ev[0].ent["id"]:
            ev[1] = G.Fact(rng, "t")
        q1, q2 = ev[0].ask(), ev[1].ask()
        msg = G.pick(rng, T_NOTE).format(who=G.pick(rng, ["sister", "boss", "landlord", "coach"]), A=ev[0].stored()[1], B=ev[1].stored()[1],
                                         Q1n=q1.rstrip("?"), q2=_lcw(q2))
        res["note2"][0] += S.right(_turn(ad, msg, 100), [[ev[0].value], [ev[1].value]]); res["note2"][1] += 1
    ad.forget()
    out = {"all_parts_right": S.rate(sum(v[0] for v in res.values()), sum(v[1] for v in res.values())),
           "by_type": {t: S.rate(*v) for t, v in res.items()}}
    print("multi", out["all_parts_right"], {t: v["k"] for t, v in out["by_type"].items()}, flush=True)
    return out


SECTIONS = {"afterno": afterno_section, "phrasing": phrasing_section, "probe2": probe2_section, "reason": reason_section,
            "twomem": twomem_section, "open": open_section, "multi": multi_section}
