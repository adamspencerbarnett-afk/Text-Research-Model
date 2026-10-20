"""Memory outside the weights: two pools searched together and ranked apart (16 October design, decisions 2, 9, 10).

  facts    the stored knowledge (Q&A exchanges, /add): kept across conversations
  session  this conversation: every turn, long turns as pieces of about PIECE bytes, the assistant's replies too;
           cleared by forget(), never mixed into the facts (the old single counter let /add turn a session permanent)

Scores: facts by idf-weighted overlap over the union of both keys (the old literal score) plus the dense key; session
pieces by how much of the question they contain (a long piece is not penalised for its length) plus the dense key.
The gate keeps an entry only if its word score or its meaning score reaches the threshold; an exact stored question
is ranked first but removes nothing. Two extra searches: the recent turns' words (what "it" / "she" refer to), and the
names found in the best session items (two-step: "my sister" -> "Kim" -> "Kim lives in Oslo").
The dense key is a plug-in (dense_keys.DenseKeys = MiniLM, labelled as the outside reference in every result)."""
from __future__ import annotations

import collections
import math
import re
from dataclasses import dataclass, field

import torch

from . import encoder  # noqa: F401  (puts compiler/ on the path)
import compiler_v0 as cv  # noqa: E402

from .window import Item

STOP = set("""a an the of to in on at for by with and or but if then than that this these those is are was were be been being
am do does did have has had will would can could should may might must not no yes it its i you he she we they me him her us them
my your his our their what which who whom whose when where why how as from into about over under again further once here there
all any both each few more most other some such only own same so very s t just now please give write describe explain list
name tell make create generate provide following sentence""".split())
PIECE = 200
NAME = re.compile(rb"\b[A-Z][a-z]{2,}\b")
# words that start sentences or replies, not names (retrieval study, 16 Oct: "Got it." made "got" a followed name)
NAME_STOP = {b"got", b"thanks", b"thank", b"okay", b"noted", b"yes", b"no", b"the", b"this", b"that", b"sure", b"actually",
             b"here", b"there", b"what", b"when", b"where", b"who", b"how", b"please", b"hello", b"hi", b"my", b"your", b"it",
             b"i", b"we", b"they", b"he", b"she", b"and", b"but", b"so", b"also", b"great", b"good", b"well", b"assistant", b"user"}


def key_of(text: bytes) -> frozenset:
    return frozenset(w for w in (u.strip(b" ").lower() for u in cv.scan(text) if u.strip(b" ").isalpha())
                     if w.decode("latin-1") not in STOP and len(w) > 1)


def split_text(b: bytes, limit: int = PIECE) -> list[bytes]:
    """Sentences grouped into pieces of at most ``limit`` bytes (a longer sentence is cut at a space)."""
    sents = [x for x in re.split(rb"(?<=[.!?\n])\s+", b.strip()) if x]
    out, cur = [], b""
    for x in sents:
        while len(x) > limit:
            cut = x.rfind(b" ", 0, limit); cut = cut if cut > 0 else limit
            if cur:
                out.append(cur); cur = b""
            out.append(x[:cut]); x = x[cut:].strip()
        if cur and len(cur) + 1 + len(x) > limit:
            out.append(cur); cur = x
        else:
            cur = (cur + b" " + x).strip()
    if cur:
        out.append(cur)
    return out or [b.strip()[:limit]]


@dataclass
class Pool:
    items: list = field(default_factory=list)            # Item per entry (what the window shows)
    keys: list = field(default_factory=list)             # word key per entry (what is searched)
    index: dict = field(default_factory=lambda: collections.defaultdict(list))
    df: collections.Counter = field(default_factory=collections.Counter)
    exact: dict = field(default_factory=dict)
    chunks: list = field(default_factory=list)           # dense vectors, added in pieces and joined only when searched
    joined: torch.Tensor | None = None                   # (joining on every add copied the whole store each time)

    @property
    def vectors(self):
        if self.chunks:
            self.joined = torch.cat(([self.joined] if self.joined is not None else []) + self.chunks); self.chunks = []
        return self.joined

    @vectors.setter
    def vectors(self, v):
        self.joined, self.chunks = v, []

    def add(self, item: Item, key_text: bytes, vec=None) -> int:
        k = key_of(key_text)
        i = len(self.items)
        self.items.append(item); self.keys.append(k)
        self.exact.setdefault(k, i)
        for w in k:
            self.index[w].append(i); self.df[w] += 1
        if vec is not None:
            self.chunks.append(vec)
        return i


class Memory:
    def __init__(self, dense=None, gate: tuple[float, float] | None = (0.15, 0.55), max_df_share: float = 0.05,
                 gate_facts: tuple[float, float] | None = None, k_facts: int = 6, k_session: int = 8, use_facts: bool = True,
                 names_fixed: bool = False, recent_search: bool = True):
        # gate: (word, meaning) bar for an entry to enter the memory block; gate_facts overrides it for the stored facts
        # (16 Oct step 1: with 50,937 general facts, a one-word match let six look-alike facts into every block)
        self.dense, self.gate, self.max_df_share = dense, gate, max_df_share
        self.gate_facts, self.k_facts, self.k_session, self.use_facts = gate_facts, k_facts, k_session, use_facts
        # retrieval study (16 Oct): fixed name-following keeps two-step at 90% without "got"; the recent-turns search
        # added look-alikes (52% -> 38% without it) and lowered two-step (88% -> 95% without it)
        self.names_fixed, self.recent_search = names_fixed, recent_search
        self.facts, self.session = Pool(), Pool()

    # ------------------------------------------------------------------ writing
    def _vec(self, texts: list[bytes]):
        return self.dense.embed(texts) if self.dense is not None else None

    def add_facts(self, pairs: list, vecs=None) -> None:
        """Stored (uploaded) facts: (q, a) found by the question's words, or (q, a, key_text) found by key_text (an
        uploaded document piece is found by its own words). vecs: precomputed dense vectors in the same order."""
        keys = [p[2] if len(p) > 2 else p[0] for p in pairs]
        if vecs is None and pairs:
            vecs = self._vec(keys)
        for (q, a, *_), k in zip(pairs, keys):
            self.facts.add(Item(q, a, "fact"), k, None)
        if vecs is not None:
            self.facts.chunks.append(vecs)

    def add_turn(self, user: bytes, reply: bytes, turn: int, vecs=None) -> None:
        """A finished turn: one entry if short, else one per piece; the reply's pieces too (not 'Noted.').
        vecs: precomputed dense vectors of turn_entries(...) (the session generator reuses them)."""
        entries = self.turn_entries(user, reply, turn)
        if vecs is None:
            vecs = self._vec([t for _, t in entries])
        for j, (it, t) in enumerate(entries):
            self.session.add(it, t, vecs[j:j + 1] if vecs is not None else None)

    @staticmethod
    def turn_entries(user: bytes, reply: bytes, turn: int) -> list:
        entries = []
        pieces = split_text(user)
        if len(pieces) == 1 and len(reply) <= PIECE:
            entries.append((Item(user, reply, "turn", turn), user))
        else:
            entries += [(Item(p, b"", "turn", turn), p) for p in pieces]
        if len(reply) >= 40 and not (len(pieces) == 1 and len(reply) <= PIECE):
            entries += [(Item(user[:120], p, "turn", turn), p) for p in split_text(reply)]
        elif len(reply) >= 40:
            entries += [(Item(user[:120], reply, "turn", turn), reply)]
        return entries

    def forget_session(self) -> None:
        self.session = Pool()

    def drop_turn(self, turn: int) -> None:
        """Remove every session entry of one turn (the engine's undo); the pool is rebuilt without them."""
        old, keep = self.session, [i for i, it in enumerate(self.session.items) if it.turn != turn]
        new = Pool()
        for i in keep:
            new.add(old.items[i], b" ".join(sorted(old.keys[i])) or old.items[i].q)
        if old.vectors is not None and keep:
            new.vectors = old.vectors[keep]
        new.keys = [old.keys[i] for i in keep]
        self.session = new

    # ------------------------------------------------------------------ reading
    def _idf(self, w) -> float:
        n = len(self.facts.items) + len(self.session.items)
        return math.log((1 + n) / (1 + self.facts.df[w] + self.session.df[w]))

    def _common(self, w) -> bool:
        n = max(1, len(self.facts.items) + len(self.session.items))
        return self.facts.df[w] + self.session.df[w] > self.max_df_share * n

    def _scores(self, pool: Pool, qk: frozenset, qvec, coverage: bool, skip=lambda i: False, top: int = 50, gate="default"):
        sums: dict = collections.defaultdict(float)
        for w in qk:
            if self._common(w):
                continue
            idf = self._idf(w)
            for i in pool.index.get(w, ()):
                sums[i] += idf
        qd = sum(self._idf(w) for w in qk) or 1.0
        word = {}
        for i, s in sums.items():
            word[i] = s / qd if coverage else s / max(qd + sum(self._idf(w) for w in pool.keys[i] if w not in qk), 1e-9)
        cand = set(sorted(word, key=word.get, reverse=True)[:top])
        cos = None
        if qvec is not None and pool.vectors is not None and len(pool.items):
            cos = (pool.vectors @ qvec).float().cpu()
            cand |= set(torch.topk(cos, min(top, len(cos))).indices.tolist())
        w_d = getattr(self.dense, "weight", 0.0)
        out = []
        for i in cand:
            if skip(i):
                continue
            m = float(cos[i]) if cos is not None else 0.0
            wd = word.get(i, 0.0)
            g = self.gate if gate == "default" else gate
            if g is not None and not (wd >= g[0] or (cos is not None and m >= g[1])):
                continue
            out.append((wd + w_d * m, wd, m, i))
        out.sort(reverse=True)
        return out

    def search(self, msg: bytes, recent: list[bytes] = (), first_turn_in_window: int | None = None,
               k_facts: int | None = None, k_session: int | None = None) -> tuple[list[Item], dict]:
        """Items in relevance order (best first) from both pools, and what was found (for the reply's info).
        first_turn_in_window: session entries of this turn or later are already in the window and are skipped."""
        qk = key_of(msg)
        qvec = self._vec([msg])[0] if self.dense is not None else None
        skip_s = (lambda i: self.session.items[i].turn >= first_turn_in_window) if first_turn_in_window is not None else (lambda i: False)
        k_facts = self.k_facts if k_facts is None else k_facts
        k_session = self.k_session if k_session is None else k_session
        gf = self.gate_facts if self.gate_facts is not None else "default"
        facts = self._scores(self.facts, qk, qvec, False, gate=gf)[:k_facts] if self.use_facts and k_facts > 0 else []
        ex = self.facts.exact.get(qk) if (len(qk) >= 2 and self.use_facts) else None
        if ex is not None:                                   # an exact stored question is first, removes nothing
            facts = [(9.0, 1.0, 1.0, ex)] + [f for f in facts if f[3] != ex][:k_facts - 1]
        sess = {i: s for s, _, _, i in self._scores(self.session, qk, qvec, True, skip_s)}
        if recent and self.recent_search:                    # what the recent turns are about
            rk = key_of(b" ".join([msg] + list(recent)))
            for s, _, _, i in self._scores(self.session, rk, None, True, skip_s):
                sess[i] = max(sess.get(i, 0.0), 0.6 * s)
        names = set()
        for i in sorted(sess, key=sess.get, reverse=True)[:3]:   # follow names for two-step questions
            it = self.session.items[i]
            found = {n.lower() for n in NAME.findall(it.q + b" " + it.a)}
            if self.names_fixed:
                found -= NAME_STOP
            names |= found - set(qk)
        if names:
            for s, _, _, i in self._scores(self.session, qk | frozenset(names), None, True, skip_s):
                sess[i] = max(sess.get(i, 0.0), 0.9 * s)
        top_s = sorted(sess, key=sess.get, reverse=True)[:k_session]
        ranked = sorted([(s, "f", i) for s, _, _, i in facts] + [(sess[i], "s", i) for i in top_s], reverse=True)
        items = [self.facts.items[i] if p == "f" else self.session.items[i] for _, p, i in ranked]
        info = {"facts": len(facts), "session": len(top_s), "exact": ex is not None, "names_followed": sorted(n.decode() for n in names)[:6],
                "best": round(ranked[0][0], 3) if ranked else 0.0}
        return items, info
