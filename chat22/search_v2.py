"""Search engine v2, part 1 (docs/SEARCH_ENGINE_V2.md): the fact table, question understanding and the decide step, on
top of chat22's Memory (which stays the candidate finder). No training. Part 2 adds the judge.

Write path (remember / add_facts):
  W4  statements become records in a FACT TABLE: (person or thing, attribute) -> value, turn, the memory item; a
      correction ("Actually, Kim moved to Bergen.") replaces the value and keeps the old one as history; "My sister is
      called Kim" links "my sister" and "kim" as one person. Conversation records and uploaded records are kept apart.
Read path (search):
  R0  understand the question: personal / event fact (who, attribute), or anything else
  R1  fast lane: the newest conversation record, else the uploaded record; a person known but without that attribute,
      or no record at all for a fact question, means "never told" -> an empty block
  R2  anything else: chat22's search (stricter gate, fixed name-following, no recent-turns search) ...
  R4  ... then decide: drop items that state the asked attribute for SOMEONE ELSE (look-alikes) and items whose value a
      newer record replaced (outdated), keep a few
The patterns come from the fact generator's own sentence and question templates, so they are exact for every phrasing it
produces; anything else falls back to the candidate path."""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

from .memory import Memory, key_of
from .window import Item

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402

STRICTER = {"gate": (0.3, 0.6), "gate_facts": (0.5, 0.8), "k_facts": 2, "k_session": 4, "names_fixed": True, "recent_search": False}


# ---------------------------------------------------------------------- patterns from the generator's templates
def _rx(template: str, kind: str):
    """A template ("{E} lives in {v}.", "What is {ep} job?") as a regex with named groups who / v."""
    t = re.escape(template.rstrip(".?!"))
    who = r"(?P<who>[A-Za-z][\w' ]*?)"
    subs = {r"\{Ep\}": who + r"'s", r"\{ep\}": who + r"'s", r"\{E\}": who, r"\{e\}": who, r"\{t\}": r"(?P<who>[\w' ]+?)",
            r"\{v\}": r"(?P<v>.+?)", r"\{art\}": r"(?:a|an)", r"\{fav\}": r"(?:favourite|favorite)", r"\{col\}": r"(?:colour|color)",
            r"\{pet\}": "(?:" + "|".join(G.PETS) + ")", r"\{d\}": "(?P<d>" + "|".join(G.DAYS) + ")"}
    for k, v in subs.items():
        t = t.replace(k, v, 1)
        t = t.replace(k, v.replace("?P<who>", "").replace("?P<v>", "").replace("?P<d>", ""))   # later repeats: no names
    return re.compile(r"^\s*" + t + r"\s*[.?!]*\s*$", re.I)


def _specs():
    """(statement patterns, question patterns), each as [(kind, field, regex)], most specific first (the longest fixed
    text first, so "X's food is now Y" is read by the correction pattern, not as the value "now Y")."""
    G3.setup(test=False)
    stm, ask = [], []
    for kind, table in (("p", G.P), ("t", G.TH)):
        for fld, spec in table.items():
            s = spec if kind == "p" else spec[1:]
            for x in [s[1]] + list(s[4] or []):
                stm.append((len(re.sub(r"\{\w+\}", "", x)), kind, fld, _rx(x, kind)))
            for x in [s[0]] + list(G3.BANK.get(fld, s[2])) + list(s[3] or []):
                ask.append((len(re.sub(r"\{\w+\}", "", x)), kind, fld, _rx(x, kind)))
    G3.setup(test=False)
    key = lambda t: -t[0]
    return [t[1:] for t in sorted(stm, key=key)], [t[1:] for t in sorted(ask, key=key)]


SPECS = None
LINK_RX = [re.compile(r"^\s*my (?P<rel>[a-z]+)(?:'s name is| is called| is named) (?P<name>[A-Z][a-z]+)\s*[.!]*\s*$", re.I)]
CORR_RX = re.compile(r"^\s*actually,?\s*", re.I)
PRONOUNS = {"she", "he", "they", "her", "him", "them"}
CITY_SET = {c.lower() for c in G.CITIES}
NOT_PLACE = {"school", "academy", "college", "university", "institute", "high", "primary", "secondary", "elementary",
             "kindergarten", "nursery", "middle", "prep", "grammar", "lot", "garage", "street", "road", "avenue",
             "office", "building", "hall", "park", "centre", "center", "room", "floor", "class", "team", "department", "the"}


def is_city(v: str) -> bool:
    """A known city, or a place-like name (1-3 capitalised words, no institution word): 'Lima', 'Santa Cruz'."""
    w = v.split()
    return v.lower() in CITY_SET or (0 < len(w) <= 3 and all(x[:1].isupper() for x in w) and not {x.lower() for x in w} & NOT_PLACE)


def norm_who(who: str) -> tuple[str, set]:
    """'my sister Kim' -> ('my sister kim', {'my sister kim', 'my sister', 'kim'})."""
    w = re.sub(r"\s+", " ", who.strip().lower())
    w = re.sub(r"^(your|my) ", "my ", w)
    al = {w}
    m = re.match(r"^my ([a-z]+) ([a-z]+)$", w)
    if m:
        al |= {f"my {m.group(1)}", m.group(2)}
    return w, al


@dataclass
class Record:
    who: str
    attribute: str
    value: str
    turn: int
    item: Item
    history: list = field(default_factory=list)       # [(value, turn, item)] replaced by corrections


class FactTable:
    def __init__(self):
        self.entities: list[set] = []                   # alias sets; one per person / thing
        self.records: dict = {}                         # (entity id, attribute) -> Record
        self.links: dict = {}                           # entity id -> the memory item that linked a relation to a name

    def entity(self, aliases: set, create: bool = True):
        hits = [i for i, a in enumerate(self.entities) if a & aliases]
        named = [i for i in hits if any(" " not in x and not x.startswith("my ") for x in self.entities[i] & aliases)]
        if len(named) == 1:
            i = named[0]
        elif len(hits) == 1:
            i = hits[0]
        elif not hits and create:
            self.entities.append(set(aliases)); return len(self.entities) - 1
        elif hits and create:
            # several people share a relation word ("my sister" for Kim and Eli): a new entity unless a name matches
            exact = [i for i in hits if aliases <= self.entities[i]]
            if len(exact) == 1:
                return exact[0]
            self.entities.append(set(aliases)); return len(self.entities) - 1
        else:
            return None
        self.entities[i] |= aliases
        return i

    def lookup(self, aliases: set) -> list:
        return [i for i, a in enumerate(self.entities) if a & aliases]

    def link(self, rel: str, name: str, item=None):
        a, b = self.lookup({f"my {rel}"}), self.lookup({name.lower()})
        al = {f"my {rel}", name.lower(), f"my {rel} {name.lower()}"}
        if a and b and a[0] != b[0]:                    # merge the two entities
            keep, drop = a[0], b[0]
            self.entities[keep] |= self.entities[drop] | al
            for (e, att), r in list(self.records.items()):
                if e == drop:
                    self.records[(keep, att)] = r; del self.records[(e, att)]
            self.entities[drop] = set()
            if item is not None:
                self.links[keep] = item
        else:
            e = self.entity(al)
            if item is not None:
                self.links[e] = item

    def put(self, who: str, attribute: str, value: str, turn: int, item: Item, correction: bool):
        _, al = norm_who(who)
        e = self.entity(al)
        key = (e, attribute)
        old = self.records.get(key)
        if old is not None:
            old.history.append((old.value, old.turn, old.item))
            old.value, old.turn, old.item = value, turn, item
        else:
            self.records[key] = Record(who, attribute, value, turn, item)


def parse_statement(text: str):
    """(kind, attribute, who, value, correction) or ('link', rel, name) or None."""
    global SPECS
    SPECS = SPECS or _specs()
    t = text.strip()
    for rx in LINK_RX:
        m = rx.match(t)
        if m:
            return ("link", m.group("rel").lower(), m.group("name"))
    corr = bool(CORR_RX.match(t))
    t = CORR_RX.sub("", t)
    t = re.sub(r"\s+now\s*([.!]*)\s*$", r"\1", t, flags=re.I)  # "Kim lives in Lima now."
    t = t[:1].upper() + t[1:]
    event = t.lower().startswith(("the ", "parking for "))
    # events ("The vet appointment moved to ...") try the event patterns first, everything else the personal ones first
    order = sorted(SPECS[0], key=lambda x: (x[0] != ("t" if event else "p")))
    for kind, fld, rx in order:
        m = rx.match(t)
        if m and m.groupdict().get("who"):
            who = re.sub(r"\s+now$", "", m.group("who").strip(), flags=re.I)
            v = (m.groupdict().get("v") or m.groupdict().get("d") or "").strip()
            v = re.sub(r"^now\s+", "", v, flags=re.I)
            if fld == "city" and not is_city(v):
                continue                                # "moved to" is also a school or a parking lot: not a city
            if kind == "p" and fld == "school" and " moved to " in t.lower() and is_city(v):
                fld = "city"                            # "Kim moved to Lima": a city, not a school
            return (kind, fld, who, v, corr)
    return None


def parse_question(text: str):
    """(kind, attribute, who) for a personal / event fact question, else None."""
    global SPECS
    SPECS = SPECS or _specs()
    t = text.strip()
    for kind, fld, rx in SPECS[1]:
        m = rx.match(t)
        if m and m.groupdict().get("who"):
            return (kind, fld, re.sub(r"\s+now$", "", m.group("who").strip(), flags=re.I))
    return None


class SearchV2:
    """Wraps a chat22 Memory: remember -> memory + fact table; search -> fast lane, else candidates + decide."""

    def __init__(self, memory: Memory, judge=None):
        self.mem = memory
        for k, v in STRICTER.items():
            setattr(self.mem, k if k != "gate" else "gate", v)
        self.conv = FactTable()
        self.uploaded = FactTable()
        self.last_who = None
        self.judge = judge                              # part 2: scores (question, item); None = rules only
        self.split_statements = False                   # 20 Oct (engine split=True): each sentence of a message is parsed
        self.index_existing_facts()

    def index_existing_facts(self):
        """Records for stored facts already in memory (stored skill-style facts; documents and Q&A are skipped)."""
        for it in self.mem.facts.items:
            self._index(it)

    def _index(self, it):
        """A stored fact gets a record only if its question reads as a personal / event fact question and its answer as
        a statement (article and document pieces, Q&A and encyclopedia leads are found by the search instead)."""
        if it.q.startswith(b"From the ") or parse_question(it.q.decode("utf-8", "replace")) is None:
            return
        p = parse_statement(it.a.decode("utf-8", "replace"))
        if p and p[0] != "link":
            self.uploaded.put(p[2], p[1], p[3], 0, it, False)

    def _judged(self, q: bytes, items: list) -> list:
        if self.judge is None or not items:
            return items
        keep = self.judge.keep(q.decode("utf-8", "replace"), [(it.q + b" " + it.a).decode("utf-8", "replace") for it in items])
        return [it for it, k in zip(items, keep) if k]

    # ---- write path
    def add_facts(self, pairs, vecs=None):
        start = len(self.mem.facts.items)
        self.mem.add_facts(pairs, vecs)
        for i in range(start, len(self.mem.facts.items)):
            it = self.mem.facts.items[i]
            self._index(it)

    def add_turn(self, user: bytes, reply: bytes, turn: int, vecs=None):
        n0 = len(self.mem.session.items)
        self.mem.add_turn(user, reply, turn, vecs)
        item = self.mem.session.items[n0] if len(self.mem.session.items) > n0 else Item(user, reply, "turn", turn)
        text = user.decode("utf-8", "replace")
        parts = [text]
        if self.split_statements:                       # "My sister Kim is a nurse. She lives in Oslo." -> one record per sentence
            parts = [x for x in re.split(r"(?<=[.!])\s+", text.strip()) if x] or [text]
        for part in parts:
            p = parse_statement(part)
            if p is None:
                continue
            if p[0] == "link":
                self.conv.link(p[1], p[2], item)
            else:
                self.conv.put(p[2], p[1], p[3], turn, item, p[4])
                self.last_who = norm_who(p[2])[1]

    def forget_session(self):
        self.mem.forget_session(); self.conv = FactTable(); self.last_who = None

    # ---- read path
    def search(self, msg: bytes, recent=(), first_turn_in_window=None):
        q = msg.decode("utf-8", "replace")
        pq = parse_question(q)
        info = {"path": "candidates"}
        if pq:
            kind, att, who = pq
            _, al = norm_who(who)
            if al & PRONOUNS and self.last_who:
                al = self.last_who
            for table, name in ((self.conv, "conversation"), (self.uploaded, "uploaded")):
                ents = table.lookup(al)
                if name == "uploaded" and len(ents) > 1:
                    continue                                            # stored facts from many sources: "my sister" is not one person
                hits = [(e, table.records[(e, att)]) for e in ents if (e, att) in table.records]
                if hits:                                                # R1 fast lane: the newest value per person
                    items = [r.item for _, r in hits]
                    if name == "uploaded":                              # a stored fact may be about someone else
                        items = self._judged(msg, items)
                        if not items:
                            continue
                    # a two-step question ("my sister" -> Kim): the statement that links them comes too
                    items += [table.links[e] for e, _ in hits if e in table.links and table.links[e] not in items]
                    if first_turn_in_window is not None:
                        items = [it for it in items if it.source != "turn" or it.turn < first_turn_in_window] or []
                    info = {"path": f"fast lane ({name})", "people": len(hits)}
                    return items, info
            known = self.conv.lookup(al) or len(self.uploaded.lookup(al)) == 1
            if known:                                                   # the person is known, this attribute never told
                return [], {"path": "never told (person known)"}
        items, _ = self.mem.search(msg, recent, first_turn_in_window)
        if pq:                                                          # R4 decide: no look-alikes, no outdated values
            kind, att, who = pq
            _, al = norm_who(who)
            keep = []
            for it in items:
                p = parse_statement((it.q if it.source == "turn" else it.a).decode("utf-8", "replace"))
                if p and p[0] != "link" and p[1] == att and not (norm_who(p[2])[1] & al):
                    continue                                            # the same attribute for someone else
                keep.append(it)
            mentions = [it for it in keep if any(a in (it.q + b" " + it.a).decode("utf-8", "replace").lower() for a in al)]
            items = mentions                                            # a fact question: only items naming the person
            info = {"path": "candidates + decide (fact question)"}
        superseded = {id(h[2]) for t in (self.conv,) for r in t.records.values() for h in r.history}
        items = [it for it in items if id(it) not in superseded]
        items = [it for it in items if it.source == "turn"] + self._judged(msg, [it for it in items if it.source != "turn"])
        return items, info

    # the Memory interface the engine and the suite call
    def drop_turn(self, turn):
        self.mem.drop_turn(turn)


class JudgeScorer:
    """Loads runs/chat22/judge.pt and keeps the items scoring at or above its calibrated bar."""

    def __init__(self, path, device="cpu"):
        import torch
        from . import encoder as E
        from .judge import Judge, encode_pair
        ck = torch.load(path, map_location="cpu", weights_only=False); c = ck["config"]
        self.model = Judge(c["vocab"], c["width"], c["heads"], c["layers"]); self.model.load_state_dict(ck["state_dict"])
        self.dev = torch.device(device); self.model.to(self.dev).eval()
        self.enc = E.load(ROOT / c["dict"]); self.th = c["threshold"]; self._pair = encode_pair; self._torch = torch

    def keep(self, q: str, texts: list) -> list:
        import numpy as np
        torch = self._torch
        codes = [self._pair(self.enc, q, t) for t in texts]
        L = max(len(c) for c in codes)
        x = np.zeros((len(codes), L), np.int64); n = np.zeros(len(codes), np.int64)
        for j, c in enumerate(codes):
            x[j, :len(c)] = c; n[j] = len(c)
        with torch.no_grad():
            sc = torch.sigmoid(self.model(torch.from_numpy(x).to(self.dev), torch.from_numpy(n).to(self.dev))).float().cpu().numpy()
        return [bool(v >= self.th) for v in sc]
