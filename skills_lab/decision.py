"""Decision head (14 October 2026, side quest): after retrieval and the model's reply, choose to ANSWER, ask a
CLARIFYING question, or ABSTAIN ("I don't have that in my memory."). A small classifier on cheap signals the system
already has; it never sees the skills test. Adopted only if it clearly helps (Adam).

Signals (24): best combined retrieval score, its literal part, margin to the second, candidates within 0.25 of the best,
exact match, share of the question's content words found in the best / second stored question, word overlap of the two
best stored questions, whether the question names a word that tells them apart, how much of the reply comes from the
shown answers (best, any), the model's mean / lowest / first-codes confidence while writing, personal question.

Training data: invented-fact scenarios from gen_skills (no test name or value), stored in the engine's memory beside
the 51k Alpaca exchanges, each asked live through the same engine and mode the head will serve:
  answer   a stored fact asked in other words, a near miss asked by its own name, the newest of an update,
           and held-out Alpaca questions (general knowledge: the network must still answer)
  clarify  two facts that fit an under-specified question equally
  abstain  a person or thing that is not stored, or stored with other fields only

    .venv/Scripts/python.exe -B skills_lab/decision.py MODEL.pt --name NAME --mode k1|k3close [--key-head K2.pt | --dense 2]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402
import gen_skills as G  # noqa: E402

m2_qa = E.m2_qa
LABELS = ["answer", "clarify", "abstain"]
ABSTAIN_TEXT = "I don't have that in my memory."
PERSONAL = {b"my", b"i", b"me", b"our", b"we", b"mine"}
NAMES = ["best_comb", "best_literal", "margin", "n_close", "exact", "cover_best", "cover_second", "overlap_12",
         "names_option", "support_best", "support_any", "conf_mean", "conf_min", "conf_first", "personal",
         "names_in_question", "names_found_best", "rest_cover_best", "rest_cover_best_qa",
         "nll_own_answer", "nll_abstain", "nll_clarify_start", "abstain_minus_answer", "clarify_minus_answer"]


def qnames(q: bytes) -> set[bytes]:
    """Who or what the question is about: capitalised words after the first, and the relation after 'my'."""
    toks = re.findall(rb"[A-Za-z][A-Za-z\-]*(?:'s)?", q)
    out = {t.lower()[:-2] if t.lower().endswith(b"'s") else t.lower() for t in toks[1:] if t[:1].isupper() and t != b"I"}
    out |= {m.lower() for m in re.findall(rb"(?:^|\s)[Mm]y ([a-z]+)", q)}
    return out


def cwords(b: bytes) -> list[bytes]:
    """Content words, lower case, possessive 's dropped (Sam's -> sam)."""
    out = []
    for w in re.findall(rb"[A-Za-z0-9][A-Za-z0-9#@.\-]*", b):
        w = w.lower().rstrip(b".")
        w = w[:-2] if w.endswith(b"'s") else w
        if w.decode("latin-1") not in m2_qa.STOP and len(w) > 1:
            out.append(w)
    return out


def overlap(a: bytes, b: bytes) -> float:
    A, B = set(cwords(a)), set(cwords(b))
    return len(A & B) / max(1, len(A))


def features(eng, q: bytes, out: bytes, info: dict) -> list[float]:
    comb = info.get("comb") or []
    c1 = comb[0][0] if comb else 0.0
    c2 = comb[1][0] if len(comb) > 1 else 0.0
    l1 = comb[0][1] if comb else 0.0
    exch = lambda j: eng.mem.exchanges[comb[j][2]] if len(comb) > j else (b"", b"")
    (q1, a1), (q2, a2) = exch(0), exch(1)
    qw = set(cwords(q)); w1, w2 = set(cwords(q1)), set(cwords(q2))
    diff = (w1 - w2) | (w2 - w1)
    shown = [eng.mem.exchanges[i][1] for i in info.get("idx", [])] or [a1]
    p = info.get("probs") or [0.0]
    nm = qnames(q)
    idx = info.get("idx", [])
    n_own = eng.reply_nll(q, idx, out) if out else 5.0
    n_abs = eng.reply_nll(q, idx, ABSTAIN_TEXT.encode())
    n_clar = eng.reply_nll(q, idx, b"Do you mean")
    return [c1 if c1 < 9 else 3.0, l1, (c1 - c2) if c1 < 9 else 3.0, float(sum(c[0] >= c1 - 0.25 for c in comb)), float(info.get("exact", False)),
            len(qw & w1) / max(1, len(qw)), len(qw & w2) / max(1, len(qw)), len(w1 & w2) / max(1, len(w1 | w2)),
            float(bool(qw & diff)), overlap(out, a1), max(overlap(out, a) for a in shown),
            float(np.mean(p)), float(np.min(p)), float(np.mean(p[:3])), float(bool(set(re.findall(rb"[a-z]+", q.lower())) & PERSONAL)),
            float(len(nm)), len(nm & (w1 | set(cwords(a1)))) / max(1, len(nm)),
            len((qw - nm) & w1) / max(1, len(qw - nm)), len((qw - nm) & (w1 | set(cwords(a1)))) / max(1, len(qw - nm)),
            n_own, n_abs, n_clar, n_abs - n_own, n_clar - n_own]


def clarify_text(eng, info: dict) -> str:
    """'Do you mean X or Y?' from the words that tell the two best stored questions apart."""
    comb = info.get("comb") or []
    if len(comb) < 2:
        return ABSTAIN_TEXT + " Could you tell me more?"
    q1, q2 = eng.mem.exchanges[comb[0][2]][0], eng.mem.exchanges[comb[1][2]][0]
    order = sorted([comb[0][2], comb[1][2]])
    qa, qb = eng.mem.exchanges[order[0]][0], eng.mem.exchanges[order[1]][0]

    def distinct(x, y):
        yw = set(cwords(y))
        raw = [w for w in re.findall(rb"[A-Za-z0-9][A-Za-z0-9#@.\-']*", x)]
        cand = [w.rstrip(b".") for w in raw if cwords(w) and cwords(w)[0] not in yw]
        cap = [w for w in cand if w[:1].isupper()]
        w = (cap or cand or [b"the first one"])[0]
        return (w[:-2] if w.endswith(b"'s") else w).decode("utf-8", "replace")
    a, b = distinct(qa, qb), distinct(qb, qa)
    return f"Do you mean {a} or {b}?" if a.lower() != b.lower() else ABSTAIN_TEXT + " Could you tell me more?"


class Head(torch.nn.Module):
    def __init__(self, n_in, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32)); self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, 32), torch.nn.GELU(), torch.nn.Linear(32, 3))

    def forward(self, x):
        return self.net((x - self.mu) / self.sd)


def load_head(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    h = Head(len(ck["names"]), ck["mu"], ck["sd"]); h.load_state_dict(ck["state_dict"]); h.eval()
    return h, ck


def decide(head, feats) -> str:
    with torch.no_grad():
        return LABELS[int(head(torch.tensor([feats], dtype=torch.float32)).argmax(-1))]


def scenarios(rng, n: int, heldout):
    """(facts to store, oldest first; list of (question, label)) for n questions of each invented kind."""
    facts, upd, qs = [], [], []
    for _ in range(n):                                   # answer: a stored fact asked in other words
        f = G.Fact(rng); facts.append(f.stored()); qs.append((f.ask(), "answer"))
    for _ in range(n // 2):                              # answer: near misses asked by their own name
        f = G.Fact(rng, "p")
        g = G.Fact(rng, "p", f.field, G.person(rng), extra={"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]})
        facts += [f.stored(), g.stored()]; qs += [(f.ask(), "answer"), (g.ask(), "answer")]
    for _ in range(n):                                   # answer: the newest value of an update
        f = G.Fact(rng)
        while not f.can_update():
            f = G.Fact(rng)
        facts.append(f.stored()); ask = f.ask(); new, _ = f.update(); upd.append(new); qs.append((ask, "answer"))
    for _ in range(n):                                   # clarify: two facts fit the question equally
        w = None
        while w is None or w["kind"] != "clarify":
            w = G.make(rng)
        facts += w["cands"]; qs.append((w["q"], "clarify"))
    for _ in range(n):                                   # abstain: not stored (sometimes the person is, with other fields)
        f = G.Fact(rng)
        if f.kind == "p" and rng.random() < 0.5:
            other = G.pick(rng, [k for k in G.P if k != f.field])
            facts.append(G.Fact(rng, "p", other, dict(f.ent)).stored())
        qs.append((f.ask(), "abstain"))
    qs += [(q.decode("utf-8", "replace"), "answer") for q in heldout[:n]]   # general questions: the network answers
    return facts + upd, qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--name", required=True); ap.add_argument("--mode", default="k1", choices=["k1", "k3close"])
    ap.add_argument("--n", type=int, default=250, help="questions per invented kind"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dense", type=float, default=2.0); ap.add_argument("--key-head", default=""); ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    t0 = time.time()
    eng = E.SkillEngine(str(ROOT / a.model), native=str(ROOT / "data" / "cv0.exe"), device=a.device,
                        dense=0.0 if a.key_head else a.dense, log=lambda *x: None)
    if a.key_head or a.dense == 0:
        eng.mem.learned = None
    if a.key_head:
        eng.mem.attach_learned(m2_qa.LearnedKeys(str(ROOT / a.key_head), str(ROOT / "data" / "cv0.exe"), a.device))
    held = [q for q, _ in m2_qa.parse_exchanges((ROOT / "data_qa" / "heldout" / "qa_heldout.txt").read_bytes())]
    rng.shuffle(held)
    facts, qs = scenarios(rng, a.n, held)
    for q, ans in facts:
        eng.add(q.encode() if isinstance(q, str) else q, ans.encode() if isinstance(ans, str) else ans)
    X, Y = [], []
    k = 1 if a.mode == "k1" else 3
    for q, lab in qs:
        out, info = eng.ask_k(q.encode(), k=k, policy="close")
        X.append(features(eng, q.encode(), out, info)); Y.append(LABELS.index(lab))
    X, Y = np.array(X, dtype=np.float32), np.array(Y)
    print(f"{len(Y)} scenarios ({np.bincount(Y, minlength=3).tolist()} answer/clarify/abstain), {time.time() - t0:.0f}s", flush=True)
    perm = rng.permutation(len(Y)); cut = int(0.8 * len(Y)); tr, va = perm[:cut], perm[cut:]
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    head = Head(X.shape[1], mu, sd)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-3, weight_decay=1e-3)
    w = torch.tensor(len(tr) / (3 * np.bincount(Y[tr], minlength=3).clip(1)), dtype=torch.float32)
    xt, yt = torch.tensor(X[tr]), torch.tensor(Y[tr])
    for _ in range(1500):
        loss = torch.nn.functional.cross_entropy(head(xt), yt, weight=w)
        opt.zero_grad(); loss.backward(); opt.step()
    head.eval()
    with torch.no_grad():
        pv = head(torch.tensor(X[va])).argmax(-1).numpy()
    conf = np.zeros((3, 3), dtype=int)
    for t, p in zip(Y[va], pv):
        conf[t, p] += 1
    res = {"model": a.model, "mode": a.mode, "retrieval": a.key_head or (f"MiniLM x{a.dense}" if a.dense else "literal only"),
           "scenarios": len(Y), "val_accuracy": round(float((pv == Y[va]).mean()), 3),
           "val_confusion_rows_true_answer_clarify_abstain": conf.tolist(), "seconds": round(time.time() - t0)}
    torch.save({"state_dict": head.state_dict(), "mu": mu, "sd": sd, "names": NAMES, **res}, LAB / "models" / f"decision_{a.name}.pt")
    (LAB / "results" / f"decision_{a.name}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))


if __name__ == "__main__":
    main()
