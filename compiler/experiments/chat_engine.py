"""Conversation engine for the 5M chat model v2 (15 October 2026). One window layout, shared by training
(train_chat.py) and chat (run_model.py), so the model is used exactly as it was trained:

    [memory: 1-3 stored exchanges] SEP [recent turns, oldest first] User: message Assistant: reply

  memory      the closest stored exchanges for the message (word key + MiniLM; several only when their scores are close,
              stored order = oldest first, so an update sits nearer the question). Up to EV_CODES codes.
  turns       as many earlier turns as fit, newest kept first; older turns fall out of the window
  session     every turn of the conversation is also stored in memory (user message -> model reply), so a detail that
              fell out of the window can still be retrieved until the chat is closed
  reply room  codes reserved for the reply (stories need room); the window is never overrun

The engine streams the reply as it is written (a callback per piece of text) and reports how the window was used.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import m2_qa  # noqa: E402

EV_CODES = 400          # memory evidence budget (codes)
SEP_TEXT = b"\n\n"


def render(q: bytes, a: bytes | None) -> bytes:
    return m2_qa.render(q, a)


def build_window(enc, sep, cands, turns, q: bytes, answer: bytes | None, ctx: int, ev_codes: int = EV_CODES,
                 reply_room: int = 0, codes_of=None):
    """Codes of [evidence] SEP [turns] User: q Assistant: [answer EOS], trimmed to fit ctx (+1 when an answer is given).
    Returns (codes, evidence length, prompt length, turns kept)."""
    enc_x = codes_of or (lambda c: enc.encode(render(*c))[0])
    if len(cands) == 1:
        ev = enc_x(cands[0])[-ev_codes:]
    elif cands:
        share = ev_codes // len(cands)
        ev = np.concatenate([enc_x(c)[:share] for c in cands])
    else:
        ev = np.array([], dtype=np.int64)
    qc = enc.encode(render(q, None))[0]
    own = np.append(enc.encode(render(q, answer))[0], m2_qa.EOS) if answer is not None else qc
    limit = (ctx + 1) if answer is not None else ctx
    room = limit - len(ev) - len(sep) - len(own) - (0 if answer is not None else reply_room)
    if room < 0:                      # a very long message or reply: keep its end (the request) and drop the turns
        own = own[-(limit - len(ev) - len(sep) - (0 if answer is not None else reply_room)):]
        qc = qc[-len(own):] if answer is None else qc
        room = 0
    kept, parts = 0, []
    for tq, ta in reversed(turns):
        tc = enc.encode(render(tq, ta))[0]
        if len(tc) <= room:
            parts.append(tc); room -= len(tc); kept += 1
        else:
            if room > 32:             # the oldest turn that still fits partly: keep its end
                parts.append(tc[-room:]); kept += 1; room = 0
            break
    hist = np.concatenate(parts[::-1]) if parts else np.array([], dtype=np.int64)
    codes = np.concatenate([ev, sep, hist, own]).astype(np.int64)
    prompt_len = len(ev) + len(sep) + len(hist) + min(len(qc), len(own))
    return codes, len(ev), prompt_len, kept


class ChatEngine(m2_qa.QAEngine):
    """The research Q&A engine plus conversation: history in the window, session memory, several candidates, streaming."""

    def __init__(self, checkpoint: str, *args, **kw):
        super().__init__(checkpoint, *args, **kw)
        cfg = getattr(self.a, "chat_v2", None) or {}
        self.ev_codes = cfg.get("ev_codes", EV_CODES)
        self.base_n = len(self.mem.exchanges)        # entries below this are the stored knowledge; above: this session
        self.history: list[tuple[bytes, bytes]] = []
        norm = lambda b: re.sub(rb"[^a-z0-9 ]", b"", b.lower()).strip()
        loose = self.mem.exact_hit

        def strict(q):                              # one-word keys ('today') are never an exact match
            i = loose(q)
            return i if i is not None and (len(self.mem.key_of(q)) >= 2 or norm(self.mem.exchanges[i][0]) == norm(q)) else None
        self.mem.exact_hit = strict

    # ---------------------------------------------------------------- memory
    def candidates(self, q: bytes, k: int = 3, delta: float = 0.25, skip=()):
        r = [x for x in self.mem.retrieve(q, 12) if x[1] not in skip]
        if not r:
            return [], 0.0
        L = self.mem.learned
        cos = (L.vectors @ L.embed([q])[0]).float().cpu() if L is not None else None
        comb = sorted(((lit / getattr(L, "lit_std", 1.0) + (L.weight * float(cos[i]) / getattr(L, "key_std", 1.0) if cos is not None else 0.0), lit, i)
                       for lit, i in r), reverse=True)
        ex = self.mem.exact_hit(q)
        if ex is not None and ex not in skip:
            comb = [(9.0, 1.0, ex)] + [c for c in comb if c[2] != ex]
        keep = [c for c in comb if c[0] >= comb[0][0] - delta][:k]
        return sorted(c[2] for c in keep), keep[0][1]

    def remember(self, user: bytes, reply: bytes) -> None:
        """Every turn goes into the session memory as well as the history."""
        self.history.append((user, reply))
        self.add(user, reply)

    def forget(self) -> None:
        """Drop this session: the history and every session entry in memory."""
        self.history = []
        while len(self.mem.exchanges) > self.base_n:
            self.mem.exchanges.pop(); k = self.mem.keys.pop()
            if self.mem.exact.get(k) == len(self.mem.exchanges):
                del self.mem.exact[k]
            for w in k:
                self.mem.index[w].pop(); self.mem.df[w] -= 1
            if self.mem.learned is not None:
                self.mem.learned.vectors = self.mem.learned.vectors[:-1]

    # ---------------------------------------------------------------- answering
    @torch.no_grad()
    def reply(self, msg: bytes, max_codes: int = 300, stream=None, no_repeat: int = 3, k: int = 3, use_memory: bool = True,
              history: list | None = None):
        a, enc, model, dev = self.a, self.enc, self.model, self.dev
        hist = self.history if history is None else history
        t0 = time.perf_counter()
        # session entries still inside the window are not retrieved again (they are already there)
        in_window = set(range(self.base_n + max(0, len(self.history) - 12), len(self.mem.exchanges))) if history is None else set()
        idx, sim = self.candidates(msg, k=k, skip=in_window) if use_memory else ([], 0.0)
        cands = [self.mem.exchanges[i] for i in idx]
        reply_room = min(max_codes, a.ctx // 3)
        prompt, ev_len, _, kept = build_window(enc, self.sep, cands, hist, msg, None, a.ctx, self.ev_codes, reply_room)
        seq = [int(t) for t in prompt]
        start = len(seq)
        if getattr(self, "_never_dev", None) is None:
            self._never_dev = self.never_seen.to(dev)
        seq_dev = torch.tensor(seq, device=dev)
        sim_t = torch.tensor([sim], device=dev)
        cache, keys, done, shown, first = None, None, 0, b"", None
        stop = None
        while len(seq) - start < max_codes and len(seq) < a.ctx:
            span = torch.arange(len(seq), device=dev) < ev_len
            h_new, cache = model.hidden_cached(seq_dev[done:][None], done, cache)
            keys = h_new if keys is None else torch.cat([keys, h_new], 1)
            done = len(seq); model._keys = keys
            logits = model.mixed_logits(keys[:, -1], seq_dev[None], span[None], sim_t).masked_fill(self._never_dev, float("-inf"))
            banned = m2_qa.repeat_banned(seq[start:], no_repeat)
            if banned:
                logits[0, banned] = float("-inf")
            nxt = int(logits.argmax(-1).item())
            if first is None:
                first = time.perf_counter() - t0
            if nxt == m2_qa.EOS:
                stop = "end"; break
            seq.append(nxt); seq_dev = torch.cat([seq_dev, torch.tensor([nxt], device=dev)])
            text = enc.decode(seq[start:])
            cut = min([p for p in (text.find(b"\nUser"), text.find(b"\nAssistant:")) if p >= 0], default=-1)
            if cut >= 0:
                text = text[:cut]; stop = "turn"
            # stream all but the last 12 bytes until the reply ends, so a stop marker is never shown
            safe = text if stop else text[:max(0, len(text) - 12)]
            if stream is not None and len(safe) > len(shown):
                stream(safe[len(shown):]); shown = safe
            if stop:
                break
        full = enc.decode(seq[start:])
        cut = min([p for p in (full.find(b"\nUser"), full.find(b"\nAssistant")) if p >= 0], default=len(full))
        if stream is not None and cut > len(shown):
            stream(full[len(shown):cut])            # the held-back end of the reply
        out = full[:cut].strip()
        if stop is None:
            stop = "window" if len(seq) >= a.ctx else "length"
        dt = time.perf_counter() - t0
        info = {"memory": [self.mem.exchanges[i][0].decode("utf-8", "replace")[:70] for i in idx if i < self.base_n],
                "session_recalled": sum(i >= self.base_n for i in idx), "similarity": round(sim, 3),
                "window_used": len(seq), "window": a.ctx, "turns_in_window": kept, "turns_total": len(hist),
                "codes": len(seq) - start, "seconds": round(dt, 2), "first_code_ms": round(1000 * (first or dt)),
                "stopped": stop}
        return out, info
