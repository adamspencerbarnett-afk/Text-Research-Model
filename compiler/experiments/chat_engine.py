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
                 reply_room: int = 0, codes_of=None, spans: dict | None = None):
    """Codes of [evidence] SEP [turns] User: q Assistant: [answer EOS], trimmed to fit ctx (+1 when an answer is given).
    Returns (codes, copyable length, prompt length, turns kept). The copy head may point anywhere in the copyable part:
    the stored facts AND the earlier turns (a name the user said three turns ago is copied, not re-guessed).
    With ``spans`` (a dict), fills spans["cands"] = [(start, end)] per candidate and spans["turns"] = {turn index: (start, end)}
    for the kept turns: the positions the training's intermediate targets point at."""
    enc_x = codes_of or (lambda c: enc.encode(render(*c))[0])
    cand_spans = []
    if len(cands) == 1:
        ev = enc_x(cands[0])[-ev_codes:]; cand_spans = [(0, len(ev))]
    elif cands:
        share = ev_codes // len(cands)
        pieces = [enc_x(c)[:share] for c in cands]
        pos = 0
        for p in pieces:
            cand_spans.append((pos, pos + len(p))); pos += len(p)
        ev = np.concatenate(pieces)
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
    parts = parts[::-1]
    hist = np.concatenate(parts) if parts else np.array([], dtype=np.int64)
    codes = np.concatenate([ev, sep, hist, own]).astype(np.int64)
    prompt_len = len(ev) + len(sep) + len(hist) + min(len(qc), len(own))
    if spans is not None:
        spans["cands"] = cand_spans
        pos, first = len(ev) + len(sep), len(turns) - kept
        spans["turns"] = {}
        for j, p in enumerate(parts):
            spans["turns"][first + j] = (pos, pos + len(p)); pos += len(p)
    return codes, len(ev) + len(sep) + len(hist), prompt_len, kept


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

    # ---------------------------------------------------------------- answer-time tools
    def tools(self, msg: bytes):
        """Deterministic tools whose result goes into the memory slot, where the model reads and copies it (the network
        decides how to answer, the tool computes): simple arithmetic, and the day or month after / before another.
        Returns a stored-exchange-shaped (question, result) or None. No training data is involved (math stays paused)."""
        text = msg.decode("utf-8", "replace").lower()
        r = arithmetic(text) or calendar(text)
        return (msg, r.encode()) if r else None

    # ---------------------------------------------------------------- answering
    @torch.no_grad()
    def _generate(self, prompt, copy_end, sim, max_codes, no_repeat, stream=None, temperature=0.0, top_k=20, gen=None):
        """One reply from a built window: KV cache, greedy or sampled; returns (text, codes, mean log-prob, stop, first-code s)."""
        a, enc, model, dev = self.a, self.enc, self.model, self.dev
        t0 = time.perf_counter()
        seq = [int(t) for t in prompt]; start = len(seq)
        if getattr(self, "_never_dev", None) is None:
            self._never_dev = self.never_seen.to(dev)
        seq_dev = torch.tensor(seq, device=dev); sim_t = torch.tensor([sim], device=dev)
        cache, keys, done, shown, first, stop, lp = None, None, 0, b"", None, None, []
        while len(seq) - start < max_codes and len(seq) < a.ctx:
            span = torch.arange(len(seq), device=dev) < copy_end
            h_new, cache = model.hidden_cached(seq_dev[done:][None], done, cache)
            keys = h_new if keys is None else torch.cat([keys, h_new], 1)
            done = len(seq); model._keys = keys
            logits = model.mixed_logits(keys[:, -1], seq_dev[None], span[None], sim_t).masked_fill(self._never_dev, float("-inf"))
            banned = m2_qa.repeat_banned(seq[start:], no_repeat)
            if banned:
                logits[0, banned] = float("-inf")
            logp = torch.log_softmax(logits[0].float(), -1)
            if temperature > 0:
                lg = logits[0].float() / temperature
                kth = torch.topk(lg, top_k).values[-1]
                nxt = int(torch.multinomial(torch.softmax(lg.masked_fill(lg < kth, float("-inf")), -1).cpu(), 1, generator=gen).item())
            else:
                nxt = int(logits.argmax(-1).item())
            lp.append(float(logp[nxt]))
            if first is None:
                first = time.perf_counter() - t0
            if nxt == m2_qa.EOS:
                stop = "end"; break
            seq.append(nxt); seq_dev = torch.cat([seq_dev, torch.tensor([nxt], device=dev)])
            text = enc.decode(seq[start:])
            cut = min([p for p in (text.find(b"\nUser"), text.find(b"\nAssistant")) if p >= 0], default=-1)
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
        if stop is None:
            stop = "window" if len(seq) >= a.ctx else "length"
        return full[:cut].strip(), len(seq) - start, (sum(lp) / max(1, len(lp))), stop, (first or 0.0), len(seq)

    @torch.no_grad()
    def reply(self, msg: bytes, max_codes: int = 300, stream=None, no_repeat: int = 3, k: int = 3, use_memory: bool = True,
              history: list | None = None, verify: int = 0, use_tools: bool = True):
        """Answer one message. verify=N>1: N candidate replies (greedy + sampled) are checked against the evidence and the
        best supported one is shown (only when memory or a tool supplied evidence; otherwise the reply streams as written)."""
        a, enc = self.a, self.enc
        hist = self.history if history is None else history
        t0 = time.perf_counter()
        # session entries still inside the window are not retrieved again (they are already there)
        in_window = set(range(self.base_n + max(0, len(self.history) - 12), len(self.mem.exchanges))) if history is None else set()
        idx, sim = self.candidates(msg, k=k, skip=in_window) if use_memory else ([], 0.0)
        cands = [self.mem.exchanges[i] for i in idx]
        tool = self.tools(msg) if use_tools else None
        if tool:
            cands = (cands + [tool])[-3:]            # the tool's result sits last, nearest the question
        reply_room = min(max_codes, a.ctx // 3)
        prompt, copy_end, _, kept = build_window(enc, self.sep, cands, hist, msg, None, a.ctx, self.ev_codes, reply_room)
        chosen, n_cand, supports = None, 1, []
        pool = set().union(*[content_words(q + b" " + r) for q, r in cands]) if cands else set()
        qw = content_words(msg)
        relevant = bool(qw) and len(qw & pool) >= 0.5 * len(qw)       # the evidence is about this question
        if verify > 1 and cands and (tool or relevant):
            pool |= qw
            for t in hist[-kept:] if kept else []:
                pool |= content_words(t[0] + b" " + t[1])
            gen = torch.Generator().manual_seed(len(msg) + len(hist))
            outs = []
            for j in range(verify):
                o = self._generate(prompt, copy_end, sim, max_codes, no_repeat, None, 0.0 if j == 0 else 0.7, 20, gen)
                aw = content_words(o[0])
                sup = sum(w in pool for w in aw) / max(1, len(aw)) if aw else 0.0
                if any(p in o[0].lower() for p in ABSTAIN_B):     # saying it doesn't know makes no claim: kept, but a
                    sup = 0.8                                     # fully supported answer still wins
                outs.append((sup, o[2], o)); supports.append(round(sup, 2))
                if j == 0 and any(p in o[0].lower() for p in ABSTAIN_B):
                    break       # the trained 'I don't know' is kept: the check fixes wrong copies, not honest refusals
            best = max(outs, key=lambda t: (round(t[0], 2), t[1]))
            chosen, n_cand = best[2], len(outs)
            if stream is not None:                   # show the chosen reply word by word
                for piece in re.findall(rb"\S+\s*", chosen[0]):
                    stream(piece)
        else:
            chosen = self._generate(prompt, copy_end, sim, max_codes, no_repeat, stream)
        out, codes, mean_lp, stop, first, used = chosen
        dt = time.perf_counter() - t0
        info = {"memory": [self.mem.exchanges[i][0].decode("utf-8", "replace")[:70] for i in idx if i < self.base_n],
                "session_recalled": sum(i >= self.base_n for i in idx), "tool": tool[1].decode() if tool else None,
                "similarity": round(sim, 3), "window_used": used, "window": a.ctx, "turns_in_window": kept, "turns_total": len(hist),
                "codes": codes, "seconds": round(dt, 2), "first_code_ms": round(1000 * (first if n_cand == 1 else dt)),
                "stopped": stop, "candidates_checked": n_cand, "supports": supports, "mean_logprob": round(mean_lp, 3),
                "evidence": [(q + b" " + r).decode("utf-8", "replace")[:400] for q, r in cands]}
        return out, info


ABSTAIN_B = (b"don't have", b"do not have", b"don't know", b"not in my memory", b"could you tell me", b"can you tell me")
WORD = re.compile(rb"[a-z0-9][a-z0-9@.\-#']*")
STOPWORDS = {w.encode() for w in m2_qa.STOP} | {b"your", b"yes", b"no", b"is", b"are", b"it", b"was", b"mean", b"one"}


def content_words(b: bytes) -> set:
    return {w.strip(b".'") for w in WORD.findall(b.lower()) if w.strip(b".'") not in STOPWORDS and len(w) > 1}


NUMS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(NUMS) + r")"
OPS = {"plus": "+", "+": "+", "add": "+", "minus": "-", "-": "-", "times": "*", "x": "*", "*": "*", "multiplied by": "*",
       "divided by": "/", "/": "/", "over": "/"}
TAKE = r"(?:eat|ate|eats|lose|lost|give away|gave away|give|gave|use|used|spend|spent|sell|sold|throw away|break|broke)"
GET = r"(?:buy|bought|get|got|find|found|receive|received|add|added|pick|picked)"


def _n(s: str) -> float:
    return float(NUMS[s]) if s in NUMS else float(s)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"


def arithmetic(t: str) -> str | None:
    """'What is 12 plus 30?', '7 x 8', 'If I have three apples and eat one, how many are left?' -> a short worked result."""
    # word operators, or a symbol with spaces on both sides: phone numbers (555-0142), codes and times never match
    m = (re.search(r"\b" + NUM + r"\s+(multiplied by|divided by|plus|minus|times|over)\s+" + NUM + r"\b", t)
         or re.search(r"(?<![\d:.\-])" + NUM + r"\s+([-+*/x])\s+" + NUM + r"(?![\d:.\-])", t))
    if m:
        x, op, y = _n(m.group(1)), OPS[m.group(2)], _n(m.group(3))
        if op == "/" and y == 0:
            return None
        v = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else 0}[op]
        word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
        return f"{_fmt(x)} {word} {_fmt(y)} is {_fmt(v)}."
    m = re.search(r"\bhave\s+" + NUM + r"\s+([a-z]+).*?\b" + TAKE + r"\s+" + NUM, t)
    if m:
        x, thing, y = _n(m.group(1)), m.group(2), _n(m.group(3))
        return f"{_fmt(x)} {thing} minus {_fmt(y)} leaves {_fmt(x - y)} {thing}."
    m = re.search(r"\bhave\s+" + NUM + r"\s+([a-z]+).*?\b" + GET + r"\s+" + NUM, t)
    if m:
        x, thing, y = _n(m.group(1)), m.group(2), _n(m.group(3))
        return f"{_fmt(x)} {thing} plus {_fmt(y)} makes {_fmt(x + y)} {thing}."
    return None


DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def calendar(t: str) -> str | None:
    """'What comes after Tuesday?', 'Which month is before March?' -> the neighbouring day or month."""
    m = re.search(r"\b(after|before|follows|precedes)\s+(" + "|".join(DAYS + MONTHS) + r")\b", t)
    if not m:
        return None
    seq = DAYS if m.group(2) in DAYS else MONTHS
    i = seq.index(m.group(2)); step = 1 if m.group(1) in ("after", "follows") else -1
    return f"The {'day' if seq is DAYS else 'month'} {'after' if step > 0 else 'before'} {m.group(2).title()} is {seq[(i + step) % len(seq)].title()}."
