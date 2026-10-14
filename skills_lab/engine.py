"""Skills lab engine: the research QAEngine plus (1) several retrieved candidates in the evidence slot, oldest first,
(2) a stricter exact match, (3) a clarifying turn kept in the window. The window layout is build_window(), shared
with the training script so both see the same format.

  one candidate:   [its exchange, last 250 codes] SEP [earlier turns] User: question Assistant:   (the research layout)
  several:         [exchange 1][exchange 2][exchange 3] SEP ...   each trimmed from its end to an equal share,
                   in the order they were stored (oldest first), so a later fact sits nearer the question
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB / "lib"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import m2_qa  # noqa: E402

assert Path(m2_qa.__file__).resolve().parent == LAB / "lib"
NB = 250   # evidence codes, as in every research Q&A model


def build_window(enc, sep, cands, turns, q: bytes, answer: bytes | None, nb: int = NB, codes_of=None):
    """Codes of [evidence] SEP [turns] User: q Assistant: [answer EOS]; returns (codes, evidence length, prompt length)."""
    enc_x = codes_of or (lambda c: enc.encode(m2_qa.render(*c))[0])
    if len(cands) == 1:
        ev = enc_x(cands[0])[-nb:]
    elif cands:
        share = nb // len(cands)
        ev = np.concatenate([enc_x(c)[:share] for c in cands])
    else:
        ev = np.array([], dtype=np.int64)
    hist = b"".join(m2_qa.render(tq, ta) for tq, ta in turns)
    hc = enc.encode(hist)[0] if hist else np.array([], dtype=np.int64)
    qc = enc.encode(m2_qa.render(q, None))[0]
    prompt = np.concatenate([ev, sep, hc, qc]).astype(np.int64)
    if answer is None:
        return prompt, len(ev), len(prompt)
    # as in the research training: the whole exchange encoded at once, the loss from the question's length on
    own = np.append(enc.encode(m2_qa.render(q, answer))[0], m2_qa.EOS)
    return np.concatenate([ev, sep, hc, own]).astype(np.int64), len(ev), len(prompt)


def norm(q: bytes) -> bytes:
    return re.sub(rb"[^a-z0-9 ]", b"", q.lower()).strip()


class SkillEngine(m2_qa.QAEngine):
    def exact(self, q: bytes):
        """Exact match only when the content-word key has at least two words or the whole question is the same
        (the research key reduced 'How are you today?' and 'Generate a to-do-list for today.' both to {today})."""
        i = self.mem.exact_hit(q)
        if i is None:
            return None
        if len(self.mem.key_of(q)) >= 2 or norm(self.mem.exchanges[i][0]) == norm(q):
            return i
        return None

    def candidates(self, q: bytes, k: int = 3, policy: str = "close", delta: float = 0.25):
        """Up to k stored exchanges for the question, ranked by literal + dense similarity; with policy 'close',
        only those within delta of the best combined score (the user's idea: show several when they are close)."""
        r = self.mem.retrieve(q, 10)
        if not r:
            return [], 0.0
        L = self.mem.learned
        cos = (L.vectors @ L.embed([q])[0]).cpu() if L is not None else None
        # the memory's own ranking formula (MiniLM: lit_std = key_std = 1; K1/K2 heads carry their own scales)
        comb = [(lit / getattr(L, 'lit_std', 1.0) + (L.weight * float(cos[i]) / getattr(L, 'key_std', 1.0)) if L is not None else lit, lit, i)
                for lit, i in r]
        comb.sort(reverse=True)
        ex = self.exact(q)
        if ex is not None:
            comb = [(9.0, 1.0, ex)] + [c for c in comb if c[2] != ex]
        if policy == "close":
            keep = [c for c in comb if c[0] >= comb[0][0] - delta][:k]
        else:
            keep = comb[:k]
        if k == 1:
            keep = comb[:1]
        top_lit = keep[0][1]
        idx = sorted(c[2] for c in keep)      # oldest first: a later fact sits nearer the question
        return idx, top_lit

    @torch.no_grad()
    def ask_k(self, q: bytes, k: int = 3, policy: str = "close", history=None, max_codes: int = 100, no_repeat: int = 3):
        a, enc, model, dev = self.a, self.enc, self.model, self.dev
        query = (history[-1][0] + b" " + q) if history else q
        idx, sim = self.candidates(query, k, policy)
        cands = [self.mem.exchanges[i] for i in idx]
        prompt, ev_len, _ = build_window(enc, self.sep, cands, history or [], q, None)
        seq = [int(t) for t in prompt[-a.ctx:]]
        off = len(prompt) - len(seq)
        ev_len = max(0, ev_len - off)
        start = len(seq)
        if getattr(self, "_never_dev", None) is None:
            self._never_dev = self.never_seen.to(dev)
        seq_dev = torch.tensor(seq, device=dev)
        sim_t = torch.tensor([sim], device=dev)
        cache, keys, done = None, None, 0
        while len(seq) - start < max_codes and len(seq) < a.ctx:
            x = seq_dev[None]
            span = torch.arange(len(seq), device=dev) < ev_len
            h_new, cache = model.hidden_cached(seq_dev[done:][None], done, cache)
            keys = h_new if keys is None else torch.cat([keys, h_new], 1)
            done = len(seq); model._keys = keys
            logits = model.mixed_logits(keys[:, -1], x, span[None], sim_t).masked_fill(self._never_dev, float("-inf"))
            banned = m2_qa.repeat_banned(seq[start:], no_repeat)
            if banned:
                logits[0, banned] = float("-inf")
            nxt = int(logits.argmax(-1).item())
            if nxt == m2_qa.EOS:
                break
            seq.append(nxt); seq_dev = torch.cat([seq_dev, torch.tensor([nxt], device=dev)])
            text = enc.decode(seq[max(start, len(seq) - 17):])
            if b"\nUser" in text or b"\nAssistant:" in text:
                break
        out = enc.decode(seq[start:]).split(b"\nUser")[0].split(b"\nAssistant:")[0].strip()
        return out, {"candidates": [c[0].decode("utf-8", "replace")[:80] for c in cands], "similarity": round(sim, 3)}
