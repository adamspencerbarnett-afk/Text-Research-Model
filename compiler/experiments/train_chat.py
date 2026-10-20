"""Train the 5M chat model v2 (15 October 2026, Adam: "a new 5M model based on the best of everything we have learned";
adapted the same day after the reasoning-research review: intermediate targets, verification, tools, self-training).

From the best 5M core (runs/core_mt5m: modern block, Muon + WSD, P100 mix, clean Wikipedia 1.260 bits/byte), one mixed
stage (session 33's verdict; --stage a / b remain for the sequential structure) of five window kinds, all in the chat
layout of chat_engine.build_window, at a 2,048-code window:

  qa        the adopted Q&A recipe on Alpaca: own exchange 50% (30% of those reworded), else the nearest by MiniLM; half
            show 2-3 candidates; 20% answered beside unrelated evidence                   [sessions 2, 21, 23; skills lab]
  skills    data_skills/train_v4c.jsonl (130k, audited: duplicates and test-identical questions removed): near misses,
            updates, 'I don't know', clarify + follow-up, combined facts, rewriting, 'your' not 'my'    [33 -> 65/100]
  convo     real multi-turn conversations (deduplicated, held-out overlap removed, replies over 1,200 codes not used as
            targets), as many earlier turns as fit; 30% with an unrelated stored exchange    [session 31: history gain]
  recall    long generated conversations: facts stated early (some corrected later), real filler turns, then a question;
            a fact whose turn scrolled out of the window is in the memory slot, as the session memory brings it back
  rehearse  core text from the P100 mix at the full window

The copy head may point into the stored facts AND the earlier turns (a name said three turns ago is copied, not guessed).

--aux W (arm B, process supervision): an extra loss at every reply code that can be copied from the right evidence: the
copy head's attention should fall on the right stored fact or the right earlier turn (the gold span), not on a near miss.
Gold spans are known for the own exchange in Q&A windows, the answering fact(s) in skills windows, and the stating turn
(or its memory-slot copy) in recall windows. No reasoning text is generated; the target is "which fact, which words".

    python compiler/experiments/train_chat.py --out runs/chat5m_v2 --name chat5m_v2_A --steps 30000 --amp [--aux 0.5]
    python compiler/experiments/train_chat.py --audit 3000        # build windows of every kind and check them, no training
"""
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
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import chat_engine as CE  # noqa: E402
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import gen_skills_v4 as G4  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import m2_qa  # noqa: E402
import train_lm  # noqa: E402

SHARES = {  # window kind -> share of batches
    "mixed": {"qa": 0.30, "skills": 0.25, "convo": 0.15, "recall": 0.10, "rehearse": 0.20},
    "a": {"qa": 0.45, "skills": 0.35, "rehearse": 0.20},                                   # sequential, stage 1
    "b": {"qa": 0.10, "skills": 0.10, "convo": 0.35, "recall": 0.25, "rehearse": 0.20},    # sequential, stage 2
    # v2.1 (15 October, long-context review): buried evidence, recall v2, chit-chat in convo; recall uses w_recall2
    "v21": {"qa": 0.25, "skills": 0.20, "convo": 0.15, "recall": 0.10, "buried": 0.15, "rehearse": 0.15},
}
GREET = ["Hi", "Hello!", "Hey there", "Good morning", "Good evening!", "How are you today?", "Hi, how's it going?", "Hello, how are you?",
         "Hey, nice to meet you.", "Thanks!", "Thank you so much.", "Bye for now.", "Good night."]
GREET_R = {"thank": ["You're welcome!", "Happy to help.", "Any time."], "bye": ["Goodbye! Talk soon.", "Bye for now!"],
           "night": ["Good night!", "Sleep well."], "how": ["I'm doing well, thanks for asking. How can I help you today?",
           "I'm good, thank you. What can I do for you?"], "": ["Hello! How can I help you today?", "Hi! What can I do for you?",
           "Hello! What would you like to talk about?"]}
DOC_INTRO = ["Here is a text:", "Read this:", "Some notes for you:", "Here's a passage from something I'm reading.", "Please keep this in mind:",
             "I found this:"]
ACKS = ["Got it. I'll remember that.", "Thanks, noted.", "Okay, I'll keep that in mind.", "Thanks for telling me.", "Noted."]
ANSWERABLE = {"single", "nearmiss", "update", "update_earlier", "combo", "followup", "short", "yesno", "list"}
WORD = re.compile(r"[a-z0-9][a-z0-9@.\-#']*")
STOPW = set(m2_qa.STOP) | {"your", "yes", "no", "is", "are", "now"}


def words(s) -> set:
    s = s.decode("utf-8", "replace") if isinstance(s, bytes) else s
    return {w.strip(".'") for w in WORD.findall(s.lower()) if w.strip(".'") not in STOPW and len(w) > 1}


def nll_and_aux(model, x, y, src, gold, want_aux: bool, point=None):
    """V2LM.nll with the copy head's attention kept: per-code nll, and (want_aux) the intermediate-target loss
    -log(attention mass on gold positions whose follower is the target code) at codes where such a position exists.
    With ``point`` (v2.1): a second target, -log(attention mass on the point positions) at every code without a gold
    position: the separator when nothing in the window answers ("nothing here"), both facts when the reply must ask.
    Returns (nll, aux, has) or, with point, (nll, aux, has, aux_point, has_point)."""
    h = model.hidden(x)
    logits = h @ model.emb.weight.T
    p_parts, confs = [F.log_softmax(logits, -1).gather(-1, y[..., None])[..., 0].exp()], []
    loo = 1.0 if model.training else 0.0
    for table in model.ngrams:
        ids, counts, totals = table.features(x)
        c_target = (counts * (ids == y[..., None])).sum(-1)
        kept = counts.sum(-1)
        c_target = (c_target - loo * (c_target > 0)).clamp_min(0.0)
        kept = (kept - loo * (totals > 0)).clamp_min(0.0)
        p_parts.append(torch.where(kept > 0, c_target / kept.clamp_min(1.0), torch.zeros_like(kept)))
        confs.extend(train_lm.table_features(counts, totals, loo, ids, y))
    B, T, D = h.shape
    follow, x_next = model._span(x, src)
    causal = torch.ones(T, T, dtype=torch.bool, device=x.device).tril(-1)
    allowed = follow[:, None, :] & causal[None]
    scores = (model.copy_q(h) @ model.copy_k(h).transpose(1, 2)) / math.sqrt(D)
    attn = torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1).nan_to_num(0.0)
    match = x_next[:, None, :] == y[..., None]
    p_parts.append((attn * match).sum(-1))
    confs += [attn.max(-1).values, -(attn * torch.log2(attn.clamp_min(1e-9))).sum(-1)]
    mix = F.softmax(model.mix_head(torch.cat([h] + [c[..., None] for c in confs], -1)), -1)
    nll = -torch.log((mix * torch.stack(p_parts, -1)).sum(-1).clamp_min(1e-9))
    if not want_aux:
        return (nll, None, None) if point is None else (nll, None, None, None, None)
    hit = allowed & match & gold[:, None, :]
    has = hit.any(-1)
    aux = -torch.log((attn * hit).sum(-1).float().clamp_min(1e-6))
    if point is None:
        return nll, aux, has
    pt = allowed & point[:, None, :]
    has_p = pt.any(-1) & ~has
    aux_p = -torch.log((attn * pt).sum(-1).float().clamp_min(1e-6))
    return nll, aux, has, aux_p, has_p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="runs/core_mt5m/scale_custom_v0_8k.pt"); ap.add_argument("--init", default="")
    ap.add_argument("--out", default="runs/chat5m_v2"); ap.add_argument("--name", default="chat5m_v2")
    ap.add_argument("--stage", default="mixed", choices=list(SHARES)); ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--aux", type=float, default=0.0, help="weight of the intermediate-target (which fact, which words) loss; 0 = arm A")
    ap.add_argument("--skills-data", default="data_skills/train_v4c.jsonl"); ap.add_argument("--mt-data", default="data_mt")
    ap.add_argument("--extra-data", default="", help="a jsonl of extra windows (self-training: selftrain_chat.py), mixed in as kind 'self'")
    ap.add_argument("--extra-p", type=float, default=0.0)
    ap.add_argument("--point", type=float, default=0.0, help="v2.1: weight of the 'nothing here' / 'ask which' pointer target; 0 = off")
    ap.add_argument("--layout", default="v2", choices=["v2", "v22"], help="v22: turns, then the memory block, then the message (16 October)")
    ap.add_argument("--mem-codes", type=int, default=0, help="v22: memory block budget in codes (default: a quarter of the window)")
    ap.add_argument("--shares", default="", help='JSON overriding window shares, e.g. \'{"buried": 0}\' (ablations)')
    ap.add_argument("--device", default="cuda"); ap.add_argument("--amp", action="store_true"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="a few steps on small data: checks the script")
    ap.add_argument("--audit", type=int, default=0, help="build this many windows of every kind, report on them, and stop")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    G3.setup(test=False)          # invented facts never use the names and phrasings reserved for the held-out tests
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    native = str(ROOT / "data" / "cv0.exe")
    core_ck = torch.load(ROOT / a.core, map_location="cpu", weights_only=False)
    dict_path = str(ROOT / core_ck["encoding"]["dictionary"]).replace("\\", "/")
    enc = LE.V0Dict(dict_path, native)
    sep = enc.encode(CE.SEP_TEXT)[0]
    shape = core_ck["model"]
    ctx, nb = a.ctx, CE.EV_CODES
    small = a.smoke or a.audit

    # ---- Q&A data, its count tables (as the engine rebuilds them), dense neighbours
    train_file = ROOT / "data_qa" / "train" / "qa_train.txt"
    train_x = m2_qa.parse_exchanges(train_file.read_bytes())
    all_codes = enc.encode_file(str(train_file), str(out))[0]
    if a.smoke:
        train_x = train_x[:3000]
    ngrams = [train_lm.NgramTable(all_codes, o, 64) for o in (2, 3)] if not a.audit else []
    usable = [i for i, (_, x) in enumerate(train_x) if x.strip()]          # 28 Alpaca exchanges have no answer
    nb_file = out / "dense_top4.npy"
    if nb_file.exists() and not a.smoke:
        nbrs = np.load(nb_file)
    else:
        import dense_keys
        dk = dense_keys.DenseKeys(a.device, weight=2.0)
        V = dk.embed([q for q, _ in train_x])
        nbrs = np.zeros((len(train_x), 4), dtype=np.int64)
        for s in range(0, len(train_x), 2048):
            sims = V[s:s + 2048] @ V.T
            sims[torch.arange(sims.shape[0]), torch.arange(s, s + sims.shape[0])] = -2
            nbrs[s:s + 2048] = torch.topk(sims, 4).indices.cpu().numpy()
        if not a.smoke:
            np.save(nb_file, nbrs)
        del dk, V; torch.cuda.empty_cache()
    say(f"{len(usable):,} usable Q&A exchanges, dense neighbours ready")

    # ---- skills, conversations (audited), rehearsal text, self-training windows
    shares = dict(SHARES[a.stage])
    if a.shares:
        shares.update(json.loads(a.shares))
        shares = {k: v for k, v in shares.items() if v > 0}
    if a.extra_data and a.extra_p:
        shares["self"] = a.extra_p
    skills = []
    if "skills" in shares or "buried" in shares:
        with open(ROOT / a.skills_data, encoding="utf-8") as f:
            for i, l in enumerate(f):
                if small and i >= 20000:
                    break
                skills.append(json.loads(l))
        rng.shuffle(skills)
    extra = []
    if "self" in shares:
        extra = [json.loads(l) for l in open(ROOT / a.extra_data, encoding="utf-8")]
    dialogues = []
    if "convo" in shares or "recall" in shares or "buried" in shares:
        held = {json.loads(l)["turns"][0][1][:200] for l in open(ROOT / a.mt_data / "heldout" / "dialogues.jsonl", encoding="utf-8")}
        seen = set()
        with open(ROOT / a.mt_data / "train" / "dialogues.jsonl", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if small and i >= 20000:
                    break
                t = json.loads(l)["turns"]
                first = t[0][1][:200]
                if first in held or first in seen:      # held-out overlap and duplicates (audit, 15 October)
                    continue
                seen.add(first); dialogues.append(t)
    v21 = a.stage == "v21"
    stories = [j for j, t in enumerate(dialogues) if t and t[0][0] == "user" and "story" in t[0][1][:300].lower()] if v21 else []
    say(f"{len(skills):,} skill windows, {len(dialogues):,} conversations, {len(extra):,} self-training windows")
    sample = out / "rehearse_sample.txt"
    if not sample.exists():
        parts = []
        for name, mb in (("wiki", 70), ("fineweb", 90), ("books", 40), ("ultrachat", 60), ("evidence", 40)):
            with open(ROOT / "data_core" / "mix_P100" / "train" / f"{name}.txt", "rb") as f:
                parts.append(f.read((3 if small else mb) * 1_000_000))
        sample.write_bytes(b"\f".join(parts))
    rehearse = enc.encode_file(str(sample), str(out))[0]
    say(f"rehearsal text: {len(rehearse):,} codes")

    B = lambda x: x if isinstance(x, bytes) else x.encode("utf-8", "replace")
    MAX_REPLY = 1200          # conversation replies longer than this (codes) are not used as targets: they would be cut

    def pack(seq, copy_end, pl, gold_spans=(), point_spans=()):
        full = len(seq)
        seq = seq[:ctx + 1]
        mask = np.zeros(len(seq), np.float32); mask[pl:] = 1.0
        src = np.zeros(len(seq), bool); src[:copy_end] = True
        gold = np.zeros(len(seq), bool)
        for sp_ in gold_spans:
            if sp_ is not None:
                gold[sp_[0]:sp_[1]] = True
        point = np.zeros(len(seq), bool)
        for sp_ in point_spans:
            if sp_ is not None:
                point[sp_[0]:sp_[1]] = True
        return seq, mask, src, gold, full, point

    v22 = a.layout == "v22"
    mem_codes = a.mem_codes or ctx // 4

    def window(cands, turns, q, ans, labels=None):
        sp = {}
        if v22:            # memory items carry their source: a stored fact, or the turn of this conversation
            items = [(c[0], c[1], (labels[j] if labels and labels[j] else b"[fact] ")) for j, c in enumerate(cands)]
            codes_, copy_end, pl, kept = CE.build_window_v22(enc, sep, items, turns, q, ans, ctx, mem_codes, spans=sp)
        else:
            codes_, copy_end, pl, kept = CE.build_window(enc, sep, cands, turns, q, ans, ctx, nb, spans=sp)
        return codes_, copy_end, pl, kept, sp

    def sink_of(sp):
        """'Nothing here' positions. v2: the last stored-fact code (its follower is the separator) and the separator
        itself (its follower is the first turn); v22: the memory heading's last code. The loss keeps only positions the
        copy head may point at."""
        if "sink" in sp:
            return [sp["sink"]]
        ev_len = sp["cands"][-1][1] if sp.get("cands") else 0
        return [(max(0, ev_len - 1), ev_len + 1)]

    def w_qa():
        i = usable[int(rng.integers(len(usable)))]; q, ans = train_x[i]
        if rng.random() < 0.2:          # general question beside unrelated evidence: still answered
            cands = [train_x[usable[int(rng.integers(len(usable)))]] for _ in range(int(rng.integers(1, 4)))]
            c, ce, pl, _, _ = window(cands, [], q, ans)
            return pack(c, ce, pl)
        self_ = rng.random() < 0.5
        first = (q, ans)
        if self_ and rng.random() < 0.3:
            first = (m2_qa.perturb_question(q, rng, train_x[int(rng.integers(len(train_x)))][0].split(b" ")), ans)
        if not self_:
            first = train_x[int(nbrs[i, 0])]
        cands = [first]
        if rng.random() < 0.5:
            pool = [(i if self_ else int(nbrs[i, 0]), first)] + [(int(j), train_x[int(j)]) for j in nbrs[i, (0 if self_ else 1):][:int(rng.integers(1, 3))]]
            cands = [c for _, c in sorted(pool, key=lambda t: t[0])]
        c, ce, pl, _, sp = window(cands, [], q, ans)
        gold = [sp["cands"][j] for j, cc in enumerate(cands) if self_ and cc is first]
        return pack(c, ce, pl, gold)

    sk = [0]

    def w_skills(src_list=None):
        lst = src_list if src_list is not None else skills
        w = lst[sk[0] % len(lst)] if src_list is None else lst[int(rng.integers(len(lst)))]
        if src_list is None:
            sk[0] += 1
        cands = [(B(c[0]), B(c[1])) for c in w["cands"]]
        c, ce, pl, _, sp = window(cands, [(B(t[0]), B(t[1])) for t in w["turns"]], B(w["q"]), B(w["target"]))
        gold, point = [], []
        if w["kind"] in ANSWERABLE and cands:
            tw = words(w["target"])
            ov = [len(words(cc[1]) & tw) for cc in cands]
            if max(ov) > 0:
                pick = [j for j, o in enumerate(ov) if (o > 0 if w["kind"] == "list" else o == max(ov))]
                gold = [sp["cands"][j] for j in pick]
        elif w["kind"] == "unknown":
            point = sink_of(sp)
        elif w["kind"] == "clarify" and cands:
            tw = words(w["target"])
            point = [sp["cands"][j] for j, cc in enumerate(cands) if words(cc[1]) & tw]
        return pack(c, ce, pl, gold, point)

    def turn_pairs(t):
        return [(t[j][1].encode("utf-8", "replace"), t[j + 1][1].encode("utf-8", "replace")) for j in range(0, len(t) - 1, 2)
                if t[j][0] == "user" and t[j + 1][0] == "assistant"]

    def w_greet():
        """v2.1 chit-chat (live chat refused greetings and thanks): a greeting answered as one, often beside an unrelated
        stored exchange, since the memory always returns something."""
        g = G.pick(rng, GREET)
        key = next((k for k in ("thank", "bye", "night", "how") if k in g.lower()), "")
        turns = filler(int(rng.integers(0, 3))) if key in ("thank", "bye", "night") else []
        cands = [train_x[usable[int(rng.integers(len(usable)))]] for _ in range(int(rng.integers(1, 3)))] if rng.random() < 0.6 else []
        c, ce, pl, _, _ = window(cands, turns, g.encode(), G.pick(rng, GREET_R[key]).encode())
        return pack(c, ce, pl)

    def w_convo():
        if v21 and rng.random() < 0.10:
            return w_greet()
        story = v21 and stories and rng.random() < 0.10       # story requests (live chat refused named ones)
        for _ in range(50):
            pairs = turn_pairs(dialogues[stories[int(rng.integers(len(stories)))] if story else int(rng.integers(len(dialogues)))])
            if len(pairs) < (1 if story else 2):
                continue
            k = 0 if story else int(rng.integers(1, len(pairs)))
            if len(enc.encode(pairs[k][1])[0]) <= MAX_REPLY:
                break
        q, ans = pairs[k][0][-1200:], pairs[k][1]
        cands = [train_x[usable[int(rng.integers(len(usable)))]]] if rng.random() < (0.6 if story else 0.3) else []
        c, ce, pl, _, _ = window(cands, pairs[:k], q, ans)
        return pack(c, ce, pl)

    def filler(n):
        out_ = []
        while len(out_) < n:
            p = turn_pairs(dialogues[int(rng.integers(len(dialogues)))])
            if p:
                u, r = p[int(rng.integers(len(p)))]
                out_.append((u[:600], r[:700]))
        return out_

    def w_recall():
        facts = [G.Fact(rng) for _ in range(int(rng.integers(1, 4)))]
        turns = filler(int(rng.integers(1, 14)))
        stmts = {}
        for f in facts:
            st_ = f.stored()[1].encode(); stmts[id(f)] = st_
            turns.insert(int(rng.integers(0, len(turns) + 1)), (st_, G.pick(rng, ACKS).encode()))
        target_f = facts[int(rng.integers(len(facts)))]
        q, ans = target_f.ask(), G4.your(target_f.stored()[1])
        answer_turn = stmts[id(target_f)]                          # the user turn that holds the answer
        if target_f.can_update() and rng.random() < 0.3:          # a correction later in the conversation
            (_, new_ans), _ = target_f.update()
            own_pos = next(j for j, t in enumerate(turns) if t[0] == answer_turn)
            answer_turn = ("Actually, " + new_ans[0].lower() + new_ans[1:]).encode()
            turns.insert(int(rng.integers(own_pos + 1, len(turns) + 1)), (answer_turn, G.pick(rng, ACKS).encode()))
            ans = G4.your(new_ans)
        q, ans = q.encode(), ans.encode()
        stating = next(j for j, t in enumerate(turns) if t[0] == answer_turn)
        c, ce, pl, kept, sp = window([], turns, q, ans)
        gold = [sp["turns"][stating]] if stating is not None and stating in sp["turns"] else []
        if stating is not None and stating < len(turns) - kept:     # it scrolled out: the session memory brings it back
            c, ce, pl, kept, sp = window([turns[stating]], turns, q, ans)
            gold = [sp["cands"][0]]
        return pack(c, ce, pl, gold)

    def name_fmt(n):
        return {"e": n, "E": n, "ep": n + "'s", "Ep": n + "'s"}

    def as_fmt(f, s, v=None, **kw):
        old = f.fmt; f.fmt = dict(old, **kw)
        try:
            return f._f(s, v)
        finally:
            f.fmt = old

    def name_of(f):
        last = f.ent["id"].split()[-1] if f.kind == "p" else ""
        return last if last in G.NAMES else None

    def clash(d, t):
        return d.ent["id"] in t.ent["id"] or t.ent["id"] in d.ent["id"] or d.value == t.value

    def w_recall2():
        """Recall v2 (long-context review, gap 4): statements spread over 1-20 real turns (any depth); same-field
        distractors ('my brother lives in ...' beside 'my sister lives in ...'); corrections and questions by name;
        two-step questions (a name in one turn, a fact about that name in another); 25% never told -> 'I don't have that'
        with the 'nothing here' target; 10% end on a new statement -> an acknowledgement. Scrolled-out statements come
        back in the memory slot, as the session memory brings them; otherwise the slot often holds another session fact."""
        tgt = G.Fact(rng); nm = name_of(tgt)
        r = rng.random()
        absent, twostep, stmt_end = r < 0.25, 0.25 <= r < 0.40, 0.40 <= r < 0.50
        others = [G._other(rng, {tgt.ent["id"]}) for _ in range(int(rng.integers(0, 3)))]
        distract = []
        if absent or rng.random() < 0.6:
            for _ in range(int(rng.integers(1, 3))):
                d = G.Fact(rng, tgt.kind, tgt.field)
                if not clash(d, tgt):
                    distract.append(d)
        stmts = [(o.stored()[1], False) for o in others + distract]
        corr = None
        if twostep:
            rel, n = G.pick(rng, G.RELS), G.pick(rng, G.NAMES)
            f2 = G.Fact(rng, "p", None, dict(name_fmt(n), id=n))
            relf = {"e": f"my {rel}", "E": f"My {rel}", "ep": f"my {rel}'s", "Ep": f"My {rel}'s"}
            stmts += [(G.pick(rng, [f"My {rel}'s name is {n}.", f"My {rel} is called {n}."]), True), (f2.stored()[1], True)]
            q = as_fmt(f2, G.pick(rng, f2.spec[2]), **relf)
            ans = G4.your(as_fmt(f2, f2.spec[1], e=f"my {rel} {n}", E=f"My {rel} {n}", ep=f"my {rel} {n}'s", Ep=f"My {rel} {n}'s"))
        elif stmt_end:
            q, ans = tgt.stored()[1], G.pick(rng, ACKS)
        else:
            q = as_fmt(tgt, G.pick(rng, tgt.spec[2]), **name_fmt(nm)) if nm and rng.random() < 0.3 else tgt.ask()
            ans = G.pick(rng, G.ABSTAIN) if absent else G4.your(tgt.stored()[1])
            if not absent:
                if tgt.can_update() and rng.random() < 0.35:
                    if nm and rng.random() < 0.5:                       # the correction names the person
                        old = tgt.fmt; tgt.fmt = dict(old, **name_fmt(nm))
                        (_, new_s), _ = tgt.update(); tgt.fmt = old
                        new_ans = tgt._f(tgt.spec[1], tgt.new_value)
                    else:
                        (_, new_s), _ = tgt.update(); new_ans = new_s
                    corr = "Actually, " + new_s[0].lower() + new_s[1:]
                    ans = G4.your(new_ans)
                stmts.append((tgt.stored()[1], corr is None))
        turns = filler(int(rng.integers(1, 21)))
        gold_txt = set()
        for s_, g_ in stmts:
            st_ = s_.encode(); turns.insert(int(rng.integers(0, len(turns) + 1)), (st_, G.pick(rng, ACKS).encode()))
            if g_:
                gold_txt.add(st_)
        if corr:                                                         # after the statement it corrects
            own_pos = next(j for j, t in enumerate(turns) if t[0] == tgt.stored()[1].encode())
            turns.insert(int(rng.integers(own_pos + 1, len(turns) + 1)), (corr.encode(), G.pick(rng, ACKS).encode()))
            gold_txt.add(corr.encode())
        q, ans = q.encode(), ans.encode()
        gold_idx = [j for j, t in enumerate(turns) if t[0] in gold_txt]
        slot = []
        if absent and distract and rng.random() < 0.7:
            slot = [next(t for t in turns if t[0] == distract[0].stored()[1].encode())]
        elif not absent and rng.random() < 0.5 and others:
            slot = [next(t for t in turns if t[0] == others[0].stored()[1].encode())]
        tlab = lambda t_: f"[turn {turns.index(t_) + 1}] ".encode()          # v22: where in the conversation it was said
        c, ce, pl, kept, sp = window(slot, turns, q, ans, [tlab(t_) for t_ in slot])
        out_ = [j for j in gold_idx if j < len(turns) - kept]
        if out_:                                                         # scrolled out: the session memory returns them
            slot = (slot + [turns[j] for j in out_])[-3:]
            c, ce, pl, kept, sp = window(slot, turns, q, ans, [tlab(t_) for t_ in slot])
        gold = [sp["turns"][j] for j in gold_idx if j in sp["turns"]]
        gold += [sp["cands"][k] for k, s_ in enumerate(slot) if s_[0] in gold_txt]
        point = sink_of(sp) if absent else []
        return pack(c, ce, pl, gold, point)

    def enc_b(b):
        return enc.encode(b)[0].astype(np.int64)

    def w_buried():
        """Buried evidence (long-context review, gap 1; our form of the proxy-context method): the evidence of a Q&A
        or skills window is placed at a random depth (10-90%) inside a long pasted document (600-1,900 codes of rehearsal
        text) in an earlier user turn, beside distractors (the nearest Q&A answers, or the window's other facts). The
        gold span follows the evidence; 'I don't know' windows point at the separator, 'which one?' windows at both facts."""
        T = int(rng.integers(600, 1900)); depth = rng.uniform(0.1, 0.9)
        slot = [train_x[usable[int(rng.integers(len(usable)))]]] if rng.random() < 0.5 else []
        post, point_kind = [], None
        if rng.random() < 0.6 or not skills:
            i = usable[int(rng.integers(len(usable)))]; q, target = train_x[i]
            if rng.random() < 0.3:
                q = m2_qa.perturb_question(q, rng, train_x[int(rng.integers(len(train_x)))][0].split(b" "))
            facts = [] if rng.random() < 0.15 else [(target, True)]          # 15%: not in the document (answered anyway)
            facts += [(train_x[int(j)][1], False) for j in nbrs[i, :int(rng.integers(1, 3))]]
        else:
            w = skills[int(rng.integers(len(skills)))]
            q, target = B(w["q"]), B(w["target"])
            tw = words(w["target"]); ov = [len(words(cc[1]) & tw) for cc in w["cands"]] or [0]
            good = (lambda j: ov[j] > 0) if w["kind"] in ("list", "clarify") else (lambda j: ov[j] == max(ov) > 0)
            facts = [(B(cc[1]), w["kind"] in ANSWERABLE | {"clarify"} and good(j)) for j, cc in enumerate(w["cands"])]
            post = [(B(t[0]), B(t[1])) for t in w["turns"]]
            point_kind = w["kind"] if w["kind"] in ("unknown", "clarify") else None
        if v22:
            ev = np.concatenate([enc_b(CE.MEM_HEAD)] + ([enc_b(b"[fact] " + CE.render(*slot[0]))[:CE.ITEM_CAP]] if slot else []))
        else:
            ev = enc_b(CE.render(*slot[0]))[-nb:] if slot else np.array([], np.int64)
        head = enc_b(b"User: " + G.pick(rng, DOC_INTRO).encode() + b"\n")
        tail = enc_b(b"\nAssistant: " + G.pick(rng, ["Okay.", "Got it.", "Thanks, I've read it."]).encode() + b"\n\n")
        post_c = [enc_b(CE.render(*t)) for t in post]
        own = np.append(enc_b(CE.render(q, target)), m2_qa.EOS); qc = enc_b(CE.render(q, None))
        pieces = [(enc_b(b" " + f + b" ")[:300], g) for f, g in facts]
        fixed = len(ev) + len(sep) + len(head) + len(tail) + sum(map(len, post_c))
        doc_n = max(64, min(T, ctx + 1 - len(own)) - fixed - sum(len(p) for p, _ in pieces))
        j0 = int(rng.integers(0, len(rehearse) - doc_n - 1))
        doc = rehearse[j0:j0 + doc_n].astype(np.int64)
        # offsets in the document: the first gold piece at the chosen depth of the whole prompt, the others anywhere
        first_gold = next((k for k, (_, g) in enumerate(pieces) if g), None)
        offs = [int(rng.integers(0, doc_n + 1)) for _ in pieces]
        if first_gold is not None:
            offs[first_gold] = int(np.clip(depth * (fixed + doc_n) - len(ev) - len(sep) - len(head), 0, doc_n))
        order = sorted(range(len(pieces)), key=lambda k: offs[k])
        base = len(head) if v22 else len(ev) + len(sep) + len(head)
        parts, spans, prev, pos = [], {}, 0, base
        for k in order:
            parts.append(doc[prev:offs[k]]); pos += offs[k] - prev; prev = offs[k]
            parts.append(pieces[k][0]); spans[k] = (pos, pos + len(pieces[k][0])); pos += len(pieces[k][0])
        parts.append(doc[prev:])
        if v22:            # turns first, then the memory block, then the message
            seq = np.concatenate([head] + parts + [tail] + post_c + [ev, sep, own]).astype(np.int64)
            hl = len(head) + sum(map(len, parts)) + len(tail) + sum(map(len, post_c))
            sink_sp = [(hl + len(enc_b(CE.MEM_HEAD)) - 1, hl + len(enc_b(CE.MEM_HEAD)))]
        else:
            seq = np.concatenate([ev, sep, head] + parts + [tail] + post_c + [own]).astype(np.int64)
            sink_sp = [(max(0, len(ev) - 1), len(ev) + 1)]
        copy_end = len(seq) - len(own); pl = copy_end + len(qc)
        gold = [spans[k] for k, (_, g) in enumerate(pieces) if g] if point_kind is None else []
        point = sink_sp if point_kind == "unknown" else ([spans[k] for k, (_, g) in enumerate(pieces) if g] if point_kind == "clarify" else [])
        return pack(seq, copy_end, pl, gold, point)

    def w_rehearse():
        j = int(rng.integers(0, len(rehearse) - ctx - 1))
        seq = rehearse[j:j + ctx + 1].astype(np.int64)
        return seq, np.ones(len(seq), np.float32), np.zeros(len(seq), bool), np.zeros(len(seq), bool), len(seq), np.zeros(len(seq), bool)

    makers = {"qa": w_qa, "skills": w_skills, "convo": w_convo, "recall": w_recall2 if v21 else w_recall, "rehearse": w_rehearse,
              "buried": w_buried, "self": lambda: w_skills(extra)}
    kinds, probs = list(shares), np.array(list(shares.values())); probs = probs / probs.sum()

    # ---------------------------------------------------------------- audit mode: check every window kind, no training
    if a.audit:
        rep = {}
        for kind in kinds:
            lens, cut, gold_n, gold_ok, ans_codes, point_n = [], 0, 0, 0, [], 0
            for n_ in range(a.audit):
                seq, mask, src, gold, full, point = makers[kind]()
                point_n += bool(point.any())
                if n_ < 2 and kind != "rehearse":      # show two windows of each kind, decoded
                    dec = lambda sel: " / ".join(enc.decode(seq[sel].tolist()).decode("utf-8", "replace").splitlines())
                    ce = int(src.sum())
                    print(f"--- {kind} #{n_ + 1}: {len(seq)} codes, copyable {ce}, gold {int(gold.sum())}")
                    print("  COPYABLE:", dec(src)[:260], "..." if ce > 80 else "")
                    print("  GOLD:    ", dec(gold)[:200] if gold.any() else "(none)", "| POINT:", dec(point)[:120] if point.any() else "(none)")
                    print("  REPLY:   ", dec(mask.astype(bool))[:200])
                lens.append(full); cut += full > ctx + 1; ans_codes.append(int(mask.sum()))
                if gold.any():
                    gold_n += 1
                    # the gold span must contain the reply's content words that are not in the question
                    reply = enc.decode(seq[mask.astype(bool)].tolist())
                    gtxt = enc.decode(seq[gold].tolist())
                    need = words(reply) - {"remember", "got", "noted"}
                    gold_ok += bool(need) and len(need & words(gtxt)) >= max(1, len(need) // 2)
            rep[kind] = {"codes_p50_p95_max": [int(np.percentile(lens, 50)), int(np.percentile(lens, 95)), int(max(lens))],
                         "windows_cut": cut, "reply_codes_p50": int(np.percentile(ans_codes, 50)), "no_reply_codes": sum(x == 0 for x in ans_codes),
                         "with_gold": gold_n, "gold_holds_the_answer": gold_ok, "with_point": point_n}
            say(f"audit {kind}: {rep[kind]}")
        (ROOT / "results" / f"chat5m_window_audit{'_' + a.stage if a.stage != 'mixed' else ''}{'_' + a.layout if a.layout != 'v2' else ''}.json").write_text(json.dumps(rep, indent=1))
        return 0

    # ---- model: the core, a fresh copy head and trust head; or a stage-1 / arm model (--init)
    model = m2_qa.V2LM(enc, shape["width"], shape["layers"], shape["heads"], ctx, ngrams, True, False, False, shape.get("block", "modern")).to(dev)
    init = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)["state_dict"] if a.init else core_ck["state_dict"]
    own = model.state_dict()
    sd = {k: v for k, v in init.items() if k in own and own[k].shape == v.shape and not k.startswith("table")}
    model.load_state_dict(sd, strict=False)
    say(f"model: {model.param_counts()['total']:,} learned parameters, window {ctx}; {len(sd)} tensors from {a.init or a.core}; aux {a.aux}")

    heads = {n for n, _ in model.named_parameters() if n.split(".")[0] in ("copy_q", "copy_k", "mix_head")}
    groups = []
    for is_head in (False, True):
        ps = [(n, p) for n, p in model.named_parameters() if (n in heads) == is_head]
        lr = a.head_lr if is_head else a.lr
        groups += [{"params": [p for n, p in ps if p.dim() >= 2], "weight_decay": 0.1, "peak": lr},
                   {"params": [p for n, p in ps if p.dim() < 2], "weight_decay": 0.0, "peak": lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, 0.95))
    params = [p for g in groups for p in g["params"]]
    steps = 30 if a.smoke else a.steps
    stats = {k: [0.0, 0, 0.0, 0, 0.0, 0] for k in kinds}; log = []; t0 = time.time(); model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = g["peak"] * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        items = [makers[kind]() for _ in range(a.batch)]
        L = max(len(it[0]) for it in items)
        X = np.zeros((a.batch, L), np.int64); M = np.zeros((a.batch, L), np.float32); S = np.zeros((a.batch, L), bool); Gd = np.zeros((a.batch, L), bool)
        Pt = np.zeros((a.batch, L), bool)
        for r, (s_, m_, sr, gd, _, pt_) in enumerate(items):
            X[r, :len(s_)] = s_; M[r, :len(m_)] = m_; S[r, :len(sr)] = sr; Gd[r, :len(gd)] = gd; Pt[r, :len(pt_)] = pt_
        x = torch.from_numpy(X).to(dev); m = torch.from_numpy(M[:, 1:]).to(dev); s = torch.from_numpy(S[:, :-1]).to(dev)
        gold = torch.from_numpy(Gd[:, :-1]).to(dev)
        use_pt = a.point > 0 and bool(Pt.any())
        want = (a.aux > 0 and bool(Gd.any())) or use_pt
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
            if use_pt:
                nll, aux, has, aux_p, has_p = nll_and_aux(model, x[:, :-1], x[:, 1:], s, gold, True, torch.from_numpy(Pt[:, :-1]).to(dev))
            else:
                nll, aux, has = nll_and_aux(model, x[:, :-1], x[:, 1:], s, gold, want)
        if a.smoke and step == 0:          # the copied loss must equal the model's own (checked once)
            with torch.no_grad():
                ref = model.nll(x[:, :-1], x[:, 1:], s, torch.zeros(a.batch, device=dev))
            say(f"check: nll_and_aux vs V2LM.nll max difference {float((ref - nll.detach()).abs().max()):.2e}")
        loss = (nll.float() * m).sum() / m.sum().clamp_min(1.0)
        aux_v, pt_v = 0.0, 0.0
        if want and a.aux > 0:
            am = has.float() * m
            if am.sum() > 0:
                aux_l = (aux * am).sum() / am.sum()
                loss = loss + a.aux * aux_l; aux_v = float(aux_l)
        if use_pt:
            pm = has_p.float() * m
            if pm.sum() > 0:
                pt_l = (aux_p * pm).sum() / pm.sum()
                loss = loss + a.point * pt_l; pt_v = float(pt_l)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        st = stats[kind]; st[0] += float(loss.detach()) - a.aux * aux_v - a.point * pt_v; st[1] += 1; st[2] += aux_v; st[3] += aux_v > 0
        st[4] += pt_v; st[5] += pt_v > 0
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {"step": step + 1, "s": round(time.time() - t0)}
            for k, v in stats.items():
                if v[1]:
                    row[k] = round(v[0] / v[1] / math.log(2), 3)
                    if v[3]:
                        row[k + "_aux"] = round(v[2] / v[3], 3)
                    if v[5]:
                        row[k + "_point"] = round(v[4] / v[5], 3)
            log.append(row); say(f"step {step + 1}/{steps} ({(time.time() - t0) / (step + 1):.2f} s/step) " + ", ".join(f"{k} {v}" for k, v in row.items() if k not in ("step", "s")))
            stats = {k: [0.0, 0, 0.0, 0, 0.0, 0] for k in kinds}
    args = {"data": "data_qa", "work": a.out, "dict": dict_path.replace(str(ROOT).replace("\\", "/") + "/", ""), "width": shape["width"],
            "layers": shape["layers"], "heads": shape["heads"], "ctx": ctx, "block": shape.get("block", "modern"), "copy_head": True,
            "evidence": False, "sim_feature": False, "eos": True, "topk": 64, "neighbor_codes": nb, "history_codes": ctx, "dense": 2.0,
            "dense_train": True, "max_train": 0, "fold_keys": False, "calculator": False, "copy_question": False, "rerank_core": 0,
            "greedy": True, "chat_v2": {"ev_codes": nb, "copy_history": True, "stage": a.stage, "shares": shares, "aux": a.aux, "point": a.point, "layout": a.layout, "mem_codes": mem_codes if v22 else nb, "steps": steps,
                                        "core": a.core, "init": a.init, "skills_data": a.skills_data, "extra_data": a.extra_data,
                                        "train_s": round(time.time() - t0), "log": log}}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "args": args}, out / f"{a.name}.pt")
    say(f"saved {out / (a.name + '.pt')} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
