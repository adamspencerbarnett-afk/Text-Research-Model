"""The one window layout of chat22, used by training and answering alike (so they cannot drift apart):

    [turns, oldest first]  Memory:\\n [items]  SEP  User: message\\nAssistant: [reply EOS]

* The turns come first so they stay in the attention cache between replies; the memory block and the message are the
  only parts read again each turn.
* Items are chosen in RELEVANCE order (best first) until the memory budget is spent (each at most ITEM_CAP codes), then
  PLACED in a fixed order: stored facts first, then conversation items by turn, so the newest is nearest the message.
  Each item carries its source label: "[fact] " or "[turn N] ". Search selects; it never orders. (16 October design
  decision 1: the old v22 code put the best item nearest the message while training lists put the OLD value there.)
* reply_room: the room kept for the reply. Answering keeps it for a reply not written yet; training keeps the same
  room around the target reply, so the same number of turns fits in both (decision 3).
* Spans for the training targets: spans["items"][j] = (start, end) of item j as given (None if it did not fit),
  spans["turns"][i] = (start, end) of turn i if kept, spans["sink"] = the 'nothing here' position (the memory heading's
  last code, whose follower is the first item or SEP)."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .encoder import EOS, MEM_HEAD, render

ITEM_CAP = 160


@dataclass
class Item:
    q: bytes
    a: bytes
    source: str = "fact"          # "fact" | "turn"
    turn: int = 0                 # 1-based turn number for source "turn"

    def label(self) -> bytes:
        return b"[fact] " if self.source == "fact" else f"[turn {self.turn}] ".encode()

    def text(self) -> bytes:
        return self.label() + render(self.q, self.a)


def place_order(items: list[Item], chosen: list[int]) -> list[int]:
    """Placement of the chosen items: facts first (in relevance order), then turns oldest to newest."""
    facts = [j for j in chosen if items[j].source == "fact"]
    turns = sorted((j for j in chosen if items[j].source != "fact"), key=lambda j: items[j].turn)
    return facts + turns


def build(enc, sep, items: list[Item], turns: list[tuple[bytes, bytes]], q: bytes, answer: bytes | None, ctx: int,
          mem_codes: int, reply_room: int, spans: dict | None = None, encode_item=None):
    """Returns (codes, copy_end, prompt_len, turns_kept). With an answer, codes has up to ctx + 1 codes (inputs and
    targets); without, up to ctx. copy_end: the copyable prefix (turns + memory block + SEP)."""
    enc_item = encode_item or (lambda it: enc.encode(it.text())[0])
    chosen, codes_of, used = [], {}, 0
    for j, it in enumerate(items):                       # relevance order
        c = enc_item(it)[:ITEM_CAP]
        if used + len(c) <= mem_codes:
            chosen.append(j); codes_of[j] = c; used += len(c)
    order = place_order(items, chosen)
    head = enc.encode(MEM_HEAD)[0]
    qc = enc.encode(render(q, None))[0]
    own = np.append(enc.encode(render(q, answer))[0], EOS) if answer is not None else qc
    limit = ctx + 1 if answer is not None else ctx
    mem_len = len(head) + used + len(sep)
    tail = max(len(own), len(qc) + reply_room)            # the message plus the room for its reply
    room = limit - mem_len - tail
    if room < 0:                                          # a very long message: keep its end, no turns
        keep = max(1, limit - mem_len - (tail - len(own)))
        own = own[-keep:]; qc = qc[-min(len(qc), keep):]
        room = 0
    kept, parts = 0, []
    for tq, ta in reversed(turns):
        tc = enc.encode(render(tq, ta))[0]
        if len(tc) <= room:
            parts.append(tc); room -= len(tc); kept += 1
        else:
            if room > 32:                                 # the oldest turn that fits partly: its end
                parts.append(tc[-room:]); kept += 1; room = 0
            break
    parts.reverse()
    hist = np.concatenate(parts) if parts else np.zeros(0, np.int64)
    block = [codes_of[j] for j in order]
    mem = np.concatenate([head] + block) if block else head
    codes = np.concatenate([hist, mem, sep, own]).astype(np.int64)
    copy_end = len(hist) + len(mem) + len(sep)
    prompt_len = copy_end + min(len(qc), len(own))
    if spans is not None:
        pos, first = 0, len(turns) - kept
        spans["turns"] = {}
        for i, p in enumerate(parts):
            spans["turns"][first + i] = (pos, pos + len(p)); pos += len(p)
        spans["sink"] = (len(hist) + len(head) - 1, len(hist) + len(head))
        spans["items"] = [None] * len(items)
        pos = len(hist) + len(head)
        for j in order:
            spans["items"][j] = (pos, pos + len(codes_of[j])); pos += len(codes_of[j])
    return codes, copy_end, prompt_len, kept
