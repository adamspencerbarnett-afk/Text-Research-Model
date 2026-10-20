"""Training windows for chat22, every kind built through window.build (the layout the engine answers in).

Kinds (16 October design; v2.1's kinds plus the story and document kinds):
  qa         Alpaca Q&A: the own exchange in memory (30% of those reworded), else its nearest neighbour; 2-3 facts;
             20% beside unrelated facts (answered anyway). Leave-one-out on the tables (their text is the Q&A text).
  skills     invented facts (data_skills train_v4c), told in earlier turns: near misses, updates, unknown, clarify,
             combined, rewriting. Facts are [turn N] items in the order told, so the newest is nearest the message.
  convo      real conversations (deduplicated file), greetings, story requests; 30% with an unrelated fact.
  recall     facts stated across 1-20 real turns; same-field distractors; corrections (some by name); two-step; 25%
             never told; statements that scrolled out come back as [turn N] items (and only those: in-window turns are
             never repeated in memory, as the engine skips them).
  buried     the evidence of a Q&A or skills window at 10-90% depth inside a long pasted document, with distractors.
  docqa      reading a real document: a sentence at 10-90% depth, asked as "how does this sentence end" (the answer
             is in the text itself, not inserted).
  story      book continuation with memory (the next passage of a whole book, earlier pieces of the same book in
             memory) and generated story worlds (next scene with bible facts, questions about earlier scenes, repairs).
  rehearse   whole-document text from the rebuilt rehearsal sample.

Each maker returns a Window: codes (inputs + targets, at most ctx + 1), prompt_len (targets start here), copy_end,
gold spans (the evidence the reply copies), point spans ('nothing here' / 'ask which'), loo (table leave-one-out)."""
from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import encoder as E
from . import window as W
from .data import story_world as SW
from .memory import split_text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import gen_skills_v4 as G4  # noqa: E402

ACKS = ["Got it. I'll remember that.", "Thanks, noted.", "Okay, I'll keep that in mind.", "Thanks for telling me.", "Noted."]
ANSWERABLE = {"single", "nearmiss", "update", "update_earlier", "combo", "followup", "short", "yesno", "list"}
GREET = ["Hi", "Hello!", "Hey there", "Good morning", "Good evening!", "How are you today?", "Hi, how's it going?", "Hello, how are you?",
         "Hey, nice to meet you.", "Thanks!", "Thank you so much.", "Bye for now.", "Good night."]
GREET_R = {"thank": ["You're welcome!", "Happy to help.", "Any time."], "bye": ["Goodbye! Talk soon.", "Bye for now!"],
           "night": ["Good night!", "Sleep well."], "how": ["I'm doing well, thanks for asking. How can I help you today?",
           "I'm good, thank you. What can I do for you?"], "": ["Hello! How can I help you today?", "Hi! What can I do for you?",
           "Hello! What would you like to talk about?"]}
DOC_INTRO = ["Here is a text:", "Read this:", "Some notes for you:", "Here's a passage from something I'm reading.", "Please keep this in mind:",
             "I found this:"]
WORD = re.compile(r"[a-z0-9][a-z0-9@.\-#']*")
STOPW = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "your", "my", "yes", "no", "now", "it", "on", "at", "for"}


def words(s) -> set:
    s = s.decode("utf-8", "replace") if isinstance(s, bytes) else s
    return {w.strip(".'") for w in WORD.findall(s.lower()) if w.strip(".'") not in STOPW and len(w) > 1}


# a Q&A exchange that carries its own evidence (a passage, a context, options): a reading question, not a memory fact
READING_RX = re.compile(rb"Passage:|\nContext:|\(Options:")


def B(x) -> bytes:
    return x if isinstance(x, bytes) else x.encode("utf-8", "replace")


def perturb_question(q: bytes, rng, donor: list[bytes]) -> bytes:
    """Reworded stand-in (copied from m2_qa.perturb_question, syn=None): drop 20%, donor word 10%, swap pairs 15%."""
    ws, out = q.split(b" "), []
    for w in ws:
        r = rng.random()
        if r < 0.2:
            continue
        out.append(donor[int(rng.integers(len(donor)))] if (r < 0.3 and donor) else w)
    for k in range(len(out) - 1):
        if rng.random() < 0.15:
            out[k], out[k + 1] = out[k + 1], out[k]
    return b" ".join(out) if out else ws[0]


@dataclass
class Window:
    codes: np.ndarray
    prompt_len: int
    copy_end: int
    gold: list = field(default_factory=list)
    point: list = field(default_factory=list)
    loo: float = 0.0


class DataPaths:
    """Where each kind reads its data. Default: the folders every model so far trained on. data_dir (17 Oct): a size folder
    such as data_real/5m; each file is looked up there first, then in its sibling shared/ folder."""

    def __init__(self, data_dir=None):
        self.dir = Path(data_dir) if data_dir else None
        self.resolved = {}
        if self.dir is not None and not self.dir.is_absolute():
            self.dir = ROOT / self.dir

    def find(self, rel: str, default: Path) -> Path:
        """The size folder, then its shared/ folder, then the default (18 Oct: a hybrid folder holds only the files that
        differ from the defaults; what it resolves to is printed once per run)."""
        if self.dir is None:
            return default
        for base in (self.dir, self.dir.parent / "shared"):
            if (base / rel).exists():
                self.resolved[rel] = str(base / rel); return base / rel
        self.resolved[rel] = f"default {default}"
        return default

    def glob(self, pattern: str, default_dir: Path, default_pattern: str) -> list:
        if self.dir is None:
            return sorted(default_dir.glob(default_pattern))
        for base in (self.dir, self.dir.parent / "shared"):
            got = sorted(base.glob(pattern))
            if got:
                self.resolved[pattern] = str(base); return got
        self.resolved[pattern] = f"default {default_dir}/{default_pattern}"
        return sorted(default_dir.glob(default_pattern))


class Windows:
    def __init__(self, enc, ctx: int, mem_codes: int, reply_room: int, rng, split: str = "train", small: bool = False,
                 v23: bool = False, parts=None, data_dir=None):
        self.enc, self.ctx, self.mem_codes, self.reply_room, self.rng, self.split = enc, ctx, mem_codes, reply_room, rng, split
        # the v2.3 changes one by one (17 Oct diagnosis): longconv (the any-length kind), crowd (look-alike stored facts),
        # stored (half the skills windows as stored facts), budget (varied memory budget), longdocs (buried documents up to
        # the full window, as v2.1). v23=True is the v2.3 set; parts given override it.
        self.parts = set(parts) if parts is not None else ({"longconv", "crowd", "stored", "budget"} if v23 else set())
        self.v23 = bool(self.parts & {"longconv", "crowd"})
        self.sep = enc.encode(E.SEP_TEXT)[0]
        G3.setup(test=False)                      # the reserved test names and phrasings never appear in training
        cap = 20000 if small else None
        d = ROOT / "data_chat22"
        self.paths = P = DataPaths(data_dir)
        # Q&A
        if "alpacaqa" in self.parts:                      # 18 Oct ablation: the Q&A kind from Alpaca, everything else real
            qa_file = ROOT / "data_qa" / ("train/qa_train.txt" if split == "train" else "heldout/qa_heldout.txt")
        else:
            qa_file = (P.find("qa/qa_train.txt", ROOT / "data_qa" / "train" / "qa_train.txt") if split == "train"
                       else P.find("eval/qa_heldout.txt", ROOT / "data_qa" / "heldout" / "qa_heldout.txt"))
        # the full list, so indices match the neighbour table; picks come from the usable ones (44 have no answer)
        self.qa = E.parse_exchanges(qa_file.read_bytes())[:cap]
        self.usable = [i for i, (_, x) in enumerate(self.qa) if x.strip()]
        self.donor = [w for q, _ in self.qa[:5000] for w in q.split(b" ")]
        nb = (ROOT / "runs" / "chat5m_v2" / "dense_top4.npy" if "alpacaqa" in self.parts
              else P.find("qa/dense_top4.npy", ROOT / "runs" / "chat5m_v2" / "dense_top4.npy"))   # MiniLM neighbours of the training questions
        self.nbrs = np.load(nb) if (split == "train" and nb.exists() and not small) else None
        assert self.nbrs is None or len(self.nbrs) == len(self.qa), "neighbour table does not match the Q&A list"
        # skills: the last 2,000 records are held out (never trained on), the test files are never read here
        sk = [json.loads(l) for l in open(P.find("skills/train_v4c.jsonl", ROOT / "data_skills" / "train_v4c.jsonl"), encoding="utf-8")]
        self.skills = sk[:-2000][:cap] if split == "train" else sk[-2000:]
        # conversations
        dfile = P.find("convo/dialogues_train.jsonl", d / "dialogues_train.jsonl") if split == "train" else ROOT / "data_mt" / "heldout" / "dialogues.jsonl"
        self.dialogues = []
        with open(dfile, encoding="utf-8") as f:
            for i, l in enumerate(f):
                if cap and i >= cap:
                    break
                self.dialogues.append(json.loads(l)["turns"])
        self.stories_idx = [j for j, t in enumerate(self.dialogues) if t and t[0][0] == "user" and "story" in t[0][1][:300].lower()]
        # text: rehearsal codes, documents for buried / docqa, books for story continuation
        if split == "train":
            reh = P.glob("text/rehearse_*_whole_*MB_seed1.txt", d, "rehearse_P100_whole_*MB_seed1.txt")[-1]
            mb = int(re.search(r"_(\d+)MB_", reh.name).group(1))
            assert abs(reh.stat().st_size / 1e6 - mb) < 2, f"{reh} is not the size its name says"   # the 15 MB bug
            text = reh.read_bytes()[: (30_000_000 if small else None)]
            # story continuation: the Gutenberg novels by default; a data folder gives long non-fiction instead
            bsrc = P.find("long/long_over8k_real.txt", ROOT / "data_core" / "mix_P100" / "train" / "books.txt")
            self.real_long = bsrc.name != "books.txt"
            books = [b for b in bsrc.read_bytes().split(b"\f") if len(b) > (28_000 if self.real_long else 60_000)]
        else:
            ev = d / "eval_clean"
            self.real_long = P.dir is not None and (P.find("long/long_over8k_real.txt", ROOT / "data_core" / "mix_P100" / "train" / "books.txt").name != "books.txt")
            if not self.real_long:
                text = b"\f".join((ev / f).read_bytes() for f in ("wiki.txt", "fineweb.txt", "books.txt"))
                books = [(ev / "books.txt").read_bytes()]
            else:                                         # held-out windows from non-fiction only
                text = b"\f".join((ev / f).read_bytes() for f in ("wiki.txt", "fineweb.txt"))
                books = [(ev / "wiki.txt").read_bytes(), (ev / "fineweb.txt").read_bytes()]
        self.docs = [x for x in text.split(b"\f") if len(x) > 2500]
        self.books = books[:40] if small else books
        work = ROOT / "runs" / "chat22_cache"; work.mkdir(parents=True, exist_ok=True)
        if split == "train" and not small:                # the rehearsal file itself (no copy)
            src = reh
        else:
            src = work / f"rehearse_{split}{'_small' if small else ''}.txt"
            if not src.exists() or src.stat().st_size != len(text):
                src.write_bytes(text)
        self.rehearse_codes = enc.encode_file(str(src), str(work))[0]
        if self.v23:
            self._init_v23(small)
        if P.dir is not None and split == "train":
            print("data sources: " + "; ".join(f"{k} -> {v}" for k, v in P.resolved.items()), flush=True)

    # ------------------------------------------------------------------ helpers
    def _budget(self) -> int:
        if "budget" not in self.parts:
            return self.mem_codes
        return int(self.rng.choice([256, 512, 768, 1024, 1536], p=[0.15, 0.30, 0.25, 0.20, 0.10]))

    def _win(self, items, turns, q, ans, loo=0.0):
        sp = {}
        codes, ce, pl, kept = W.build(self.enc, self.sep, items, turns, B(q), B(ans), self.ctx, self._budget(), self.reply_room, sp)
        return codes, ce, pl, kept, sp

    # ------------------------------------------------------------------ v2.3: look-alike facts, any-length rows
    def _init_v23(self, small):
        from .memory import Memory
        d = ROOT / "data_chat22"
        rows = [json.loads(l) for l in open(self.paths.find("look/facts_pool_real.jsonl", d / "facts_pool.jsonl"), encoding="utf-8")]
        if small:
            rows = rows[::10]
        self.pool = Memory(None, gate=None)                   # word key only: look-alikes as the engine's word match finds them
        self.pool.add_facts([(B(r["q"]), B(r["a"]), B(r["key"])) for r in rows])
        files = [f for f in self.paths.glob("longconv/longconv_*.jsonl", d, "longconv_s1_*.jsonl") if not f.name.endswith("_99.jsonl")]
        if small and not files:                                 # a trial run (shard 99) is enough for an audit or smoke test
            files = sorted(d.glob("longconv_s1_99.jsonl"))
        lc = []
        for f in files:
            with open(f, encoding="utf-8") as h:
                for i, l in enumerate(h):
                    if small and i >= 3000:
                        break
                    lc.append(l)
        cut = max(1, len(lc) // 50)                              # the last 2% are held out
        self.longconv = lc[:-cut] if self.split == "train" else lc[-cut:]

    def _lookalikes(self, q: bytes, avoid: set, n: int) -> list:
        """Stored facts that share words with the question but do not answer it (the engine's word match)."""
        from .memory import key_of
        qk = key_of(q)
        out = []
        for _, wd, _, i in self.pool._scores(self.pool.facts, qk, None, False, gate=None)[:n * 4]:
            it = self.pool.facts.items[i]
            if not (words(it.a) & avoid):
                out.append(W.Item(it.q, it.a, "fact"))
            if len(out) >= n:
                break
        return out

    def _crowd(self, items: list, q, target) -> list:
        """v2.3: half the windows get 1-6 look-alike stored facts mixed into the ranked list; the target ignores them."""
        if "crowd" not in self.parts or self.rng.random() >= 0.5:
            return items
        extra = self._lookalikes(B(q), words(target), int(self.rng.integers(1, 7)))
        out = list(items)
        for it in extra:
            out.insert(int(self.rng.integers(0, len(out) + 1)), it)
        return out

    def longconv_(self):
        """Any-length conversations: the memory block is what the real search returned. If the answer is neither in the
        window nor in the block (the search missed), the target is "I don't have that": training never teaches inventing."""
        r = json.loads(self.longconv[int(self.rng.integers(len(self.longconv)))])
        items = [W.Item(B(q), B(a), src, int(t)) for q, a, src, t in r["items"]]
        turns = [(B(u), B(a)) for u, a in r["turns"]]
        target, absent = r["target"], r["absent"]
        c, ce, pl, kept, sp = self._win(items, turns, r["q"], target)
        val = (r.get("value") or "").lower()
        tw = words(target)
        def holds(text):
            t = text.decode("utf-8", "replace").lower() if isinstance(text, bytes) else text.lower()
            return (val in t) if val else (len(words(t) & tw) >= max(1, len(tw) // 2))
        gold_i = [sp["items"][k] for k, it in enumerate(items) if sp["items"][k] and holds(it.q + b" " + it.a)]
        first = len(turns) - kept
        gold_t = [sp["turns"][j] for j in sp["turns"] if j >= first and holds(turns[j][0] + b" " + turns[j][1])]
        if not absent and not gold_i and not gold_t:          # the search missed it and it scrolled out: say so
            target, absent = G.pick(self.rng, G.ABSTAIN), True
            c, ce, pl, kept, sp = self._win(items, turns, r["q"], target)
        if absent:
            return self._done(c, ce, pl, [], [sp["sink"]])
        return self._done(c, ce, pl, gold_i + gold_t)

    def _done(self, codes, ce, pl, gold=(), point=(), loo=0.0):
        return Window(codes[:self.ctx + 1], pl, ce, [g for g in gold if g], [p for p in point if p], loo)

    def _filler(self, n):
        out = []
        while len(out) < n:
            t = self.dialogues[int(self.rng.integers(len(self.dialogues)))]
            p = [(B(t[j][1])[:600], B(t[j + 1][1])[:700]) for j in range(0, len(t) - 1, 2) if t[j][0] == "user" and t[j + 1][0] == "assistant"]
            if p:
                out.append(p[int(self.rng.integers(len(p)))])
        return out

    def _rand_i(self) -> int:
        return self.usable[int(self.rng.integers(len(self.usable)))]

    def _rand_qa(self):
        if "realev" in self.parts:                       # a stored fact is a real fact, never a whole passage
            if not hasattr(self, "_mem_idx"):
                self._mem_idx = [i for i in self.usable if not READING_RX.search(self.qa[i][0][:400])] or self.usable
            return self.qa[self._mem_idx[int(self.rng.integers(len(self._mem_idx)))]]
        return self.qa[self._rand_i()]

    def _alpaca_evidence(self):
        """alpacabury (18 Oct): the buried-evidence Q&A taken from the Alpaca exchanges variant A trained on (full-sentence
        answers make findable evidence), with A's neighbour table for decoys; every other kind keeps its own data."""
        rng = self.rng
        if not hasattr(self, "_alp"):
            self._alp = E.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
            self._alp_ok = [i for i, (_, x) in enumerate(self._alp) if x.strip()]
            nb = ROOT / "runs" / "chat5m_v2" / "dense_top4.npy"
            self._alp_nb = np.load(nb) if nb.exists() else None
        i = self._alp_ok[int(rng.integers(len(self._alp_ok)))]; q, target = self._alp[i]
        if rng.random() < 0.3:
            q = perturb_question(q, rng, self.donor)
        facts = [] if rng.random() < 0.15 else [(target, True)]
        nb = self._alp_nb[i, :int(rng.integers(1, 3))] if self._alp_nb is not None else [self._alp_ok[int(rng.integers(len(self._alp_ok)))] for _ in range(2)]
        facts += [(self._alp[int(j)][1], False) for j in nb if self._alp[int(j)][1].strip()]
        return q, target, facts

    def _real_evidence(self):
        """realev (18 Oct): buried evidence from a real reading question: the passage sentence that holds the answer is
        the evidence, other sentences of the same passage are hard decoys, the question is asked on its own."""
        rng = self.rng
        if not hasattr(self, "_read_idx"):
            self._read_idx = [i for i in self.usable if self.qa[i][0].startswith(b"Passage:") and b"\nQuestion: " in self.qa[i][0]]
        for _ in range(20):
            i = self._read_idx[int(rng.integers(len(self._read_idx)))]
            qtext, target = self.qa[i]
            passage, q = qtext[len(b"Passage: "):].split(b"\nQuestion: ", 1)
            key = target.strip().rstrip(b".").lower()
            if len(key) < 2 or key in (b"yes", b"no"):
                continue
            sents = [x.strip() for x in re.split(rb"(?<=[.!?])\s+", passage) if 30 <= len(x.strip()) <= 300]
            gold = [x for x in sents if key in x.lower()]
            if not gold:
                continue
            others = [x for x in sents if x not in gold]
            rng.shuffle(others)
            facts = [(gold[0], True)] + [(x, False) for x in others[:int(rng.integers(1, 3))]]
            return q, target, facts
        return None

    def _fact_item(self, pair, rank_noise=True):
        return W.Item(B(pair[0]), B(pair[1]), "fact")

    def _with_doc(self, doc_codes, spans_in_doc, items, post_turns, q, ans):
        """A window whose first turn is a pasted document given as codes (so spans inside it are exact)."""
        rng = self.rng
        head = self.enc.encode(b"User: " + G.pick(rng, DOC_INTRO).encode() + b"\n")[0]
        tail = self.enc.encode(b"\nAssistant: " + G.pick(rng, ["Okay.", "Got it.", "Thanks, I've read it."]).encode() + b"\n\n")[0]
        hist = np.concatenate([head, doc_codes, tail])
        post = [self.enc.encode(E.render(*t))[0] for t in post_turns]
        hist = np.concatenate([hist] + post) if post else hist
        sp = {}
        rest, ce, pl, _ = W.build(self.enc, self.sep, items, [], B(q), B(ans), self.ctx - len(hist), self.mem_codes, self.reply_room, sp)
        off = len(hist)
        codes = np.concatenate([hist, rest]).astype(np.int64)
        doc_spans = [(len(head) + a, len(head) + b) for a, b in spans_in_doc]
        item_spans = [(a + off, b + off) if s else None for s in sp["items"] for a, b in [s or (0, 0)]]
        return codes, ce + off, pl + off, doc_spans, item_spans, (sp["sink"][0] + off, sp["sink"][1] + off)

    def _doc_with(self, pieces, depth_first, total):
        """Document codes of about ``total`` codes of real text with ``pieces`` (codes, is_gold) inserted: the first
        gold piece at ``depth_first`` of the document, the others anywhere. Returns (codes, spans per piece)."""
        rng = self.rng
        n_fill = max(64, total - sum(len(p) for p, _ in pieces))
        j0 = int(rng.integers(0, len(self.rehearse_codes) - n_fill - 1))
        doc = self.rehearse_codes[j0:j0 + n_fill].astype(np.int64)
        doc = doc[doc != E.EOS]
        offs = [int(rng.integers(0, len(doc) + 1)) for _ in pieces]
        g = next((k for k, (_, gg) in enumerate(pieces) if gg), None)
        if g is not None:
            offs[g] = int(np.clip(depth_first * len(doc), 0, len(doc)))
        order = sorted(range(len(pieces)), key=lambda k: offs[k])
        parts, spans, prev, pos = [], {}, 0, 0
        for k in order:
            parts.append(doc[prev:offs[k]]); pos += offs[k] - prev; prev = offs[k]
            parts.append(pieces[k][0]); spans[k] = (pos, pos + len(pieces[k][0])); pos += len(pieces[k][0])
        parts.append(doc[prev:])
        return np.concatenate(parts), [spans[k] for k in range(len(pieces))]

    # ------------------------------------------------------------------ kinds
    def qa_(self):
        rng, qa = self.rng, self.qa
        i = self._rand_i(); q, ans = qa[i]
        if "memqa" in self.parts and not READING_RX.search(q[:400]):
            return self._qa_from_memory(i, q, ans)
        if rng.random() < 0.2:
            items = [self._fact_item(self._rand_qa()) for _ in range(int(rng.integers(1, 4)))]
            c, ce, pl, _, _ = self._win(items, [], q, ans)
            return self._done(c, ce, pl, loo=1.0)
        own = rng.random() < 0.5 or self.nbrs is None
        first = (perturb_question(q, rng, self.donor), ans) if own and rng.random() < 0.3 else (q, ans)
        if not own:
            first = qa[int(self.nbrs[i, 0])]
        items = [self._fact_item(first)]
        if rng.random() < 0.5:
            pool = [qa[int(j)] for j in (self.nbrs[i, 1:] if self.nbrs is not None else [self._rand_i() for _ in range(3)]) if qa[int(j)][1].strip()]
            items += [self._fact_item(p) for p in pool[:int(rng.integers(1, 3))]]
            if rng.random() < 0.3:                       # the right fact is not always ranked first
                rng.shuffle(items)
        items = self._crowd(items, q, ans)
        c, ce, pl, _, sp = self._win(items, [], q, ans)
        gold = [sp["items"][j] for j, it in enumerate(items) if own and it.a == B(ans)]
        return self._done(c, ce, pl, gold, loo=1.0)

    def _qa_from_memory(self, i, q, ans):
        """memqa (Adam, 18 Oct: facts come from permanent memory): a fact question is answered only from a memory item
        that holds it; when the block lacks it, the target is "I don't have that". 70% own fact in the block (sometimes
        reworded, with 0-2 look-alikes, order varied), 30% not in the block (look-alikes, unrelated facts or nothing)."""
        rng, qa = self.rng, self.qa
        nb = [qa[int(j)] for j in (self.nbrs[i] if self.nbrs is not None else [self._rand_i() for _ in range(4)])
              if qa[int(j)][1].strip() and qa[int(j)][1] != ans]
        if rng.random() < 0.7:
            own = (perturb_question(q, rng, self.donor), ans) if rng.random() < 0.3 else (q, ans)
            items = [self._fact_item(own)] + [self._fact_item(p) for p in nb[:int(rng.integers(0, 3))]]
            if rng.random() < 0.3:
                rng.shuffle(items)
            c, ce, pl, _, sp = self._win(items, [], q, ans)
            gold = [sp["items"][j] for j, it in enumerate(items) if it.a == B(ans)]
            return self._done(c, ce, pl, gold, loo=1.0)
        r = rng.random()
        items = ([self._fact_item(p) for p in nb[:int(rng.integers(1, 4))]] if r < 0.6 else
                 [self._fact_item(self._rand_qa()) for _ in range(int(rng.integers(1, 3)))] if r < 0.85 else [])
        items = [it for it in items if it.a != B(ans)]
        target = G.pick(rng, G.ABSTAIN)
        c, ce, pl, _, sp = self._win(items, [], q, target)
        return self._done(c, ce, pl, [], [sp["sink"]])

    def skills_(self, w=None):
        rng = self.rng
        w = w or self.skills[int(rng.integers(len(self.skills)))]
        t0 = int(rng.integers(1, 40)); items = []
        stored = "stored" in self.parts and w["kind"] not in ("update", "update_earlier") and rng.random() < 0.5   # v2.3: half uploaded
        for c in w["cands"]:                              # told in this order: the newest is nearest the message
            items.append(W.Item(B(c[0]), B(c[1]), "fact" if stored else "turn", 0 if stored else t0)); t0 += int(rng.integers(1, 15))
        order = list(range(len(items))); rng.shuffle(order)   # relevance order is not the told order
        ranked = self._crowd([items[j] for j in order], w["q"], w["target"])
        turns = [(B(t[0]), B(t[1])) for t in w["turns"]]
        c, ce, pl, _, sp = self._win(ranked, turns, w["q"], w["target"])
        gold, point = [], []
        tw = words(w["target"])
        ov = [len(words(it.a) & tw) for it in ranked]
        if w["kind"] in ANSWERABLE and ranked and max(ov) > 0:
            gold = [sp["items"][j] for j, o in enumerate(ov) if (o > 0 if w["kind"] == "list" else o == max(ov))]
        elif w["kind"] == "unknown":
            point = [sp["sink"]]
        elif w["kind"] == "clarify":
            point = [sp["items"][j] for j, o in enumerate(ov) if o > 0]
        return self._done(c, ce, pl, gold, point)

    def _pairs(self, t):
        return [(B(t[j][1]), B(t[j + 1][1])) for j in range(0, len(t) - 1, 2) if t[j][0] == "user" and t[j + 1][0] == "assistant"]

    def convo_(self):
        rng = self.rng
        if rng.random() < 0.10:                          # greetings and thanks, often beside an unrelated fact
            g = G.pick(rng, GREET)
            k = next((x for x in ("thank", "bye", "night", "how") if x in g.lower()), "")
            items = [self._fact_item(self._rand_qa())] if rng.random() < 0.6 else []
            c, ce, pl, _, _ = self._win(items, self._filler(int(rng.integers(0, 3))) if k in ("thank", "bye", "night") else [], g, G.pick(rng, GREET_R[k]))
            return self._done(c, ce, pl)
        story = bool(self.stories_idx) and rng.random() < 0.10
        for _ in range(50):
            t = self.dialogues[self.stories_idx[int(rng.integers(len(self.stories_idx)))] if story else int(rng.integers(len(self.dialogues)))]
            pairs = self._pairs(t)
            if len(pairs) < (1 if story else 2):
                continue
            k = 0 if story else int(rng.integers(1, len(pairs)))
            if len(self.enc.encode(pairs[k][1])[0]) <= 1200:
                break
        items = [self._fact_item(self._rand_qa())] if rng.random() < (0.6 if story else 0.3) else []
        c, ce, pl, _, _ = self._win(items, pairs[:k], pairs[k][0][-1200:], pairs[k][1])
        return self._done(c, ce, pl)

    def recall_(self):
        rng = self.rng
        tgt = G.Fact(rng)
        nm = tgt.ent["id"].split()[-1] if tgt.kind == "p" and tgt.ent["id"].split()[-1] in G.NAMES else None
        name_fmt = lambda n: {"e": n, "E": n, "ep": n + "'s", "Ep": n + "'s"}
        r = rng.random()
        absent, twostep, stmt_end = r < 0.25, 0.25 <= r < 0.40, 0.40 <= r < 0.50
        others = [G._other(rng, {tgt.ent["id"]}) for _ in range(int(rng.integers(0, 3)))]
        distract = []
        if absent or rng.random() < 0.6:
            for _ in range(int(rng.integers(1, 3))):
                d = G.Fact(rng, tgt.kind, tgt.field)
                if d.ent["id"] not in tgt.ent["id"] and tgt.ent["id"] not in d.ent["id"] and d.value != tgt.value:
                    distract.append(d)
        stmts = [(o.stored()[1], False) for o in others + distract]
        corr = None
        if twostep:
            rel, n = G.pick(rng, G.RELS), G.pick(rng, G.NAMES)
            f2 = G.Fact(rng, "p", None, dict(name_fmt(n), id=n))
            old = f2.fmt; f2.fmt = dict(old, e=f"my {rel}", E=f"My {rel}", ep=f"my {rel}'s", Ep=f"My {rel}'s")
            q = f2._f(G.pick(rng, f2.spec[2])); f2.fmt = old
            stmts += [(G.pick(rng, [f"My {rel}'s name is {n}.", f"My {rel} is called {n}."]), True), (f2.stored()[1], True)]
            old = f2.fmt; f2.fmt = dict(old, e=f"my {rel} {n}", E=f"My {rel} {n}", ep=f"my {rel} {n}'s", Ep=f"My {rel} {n}'s")
            ans = G4.your(f2._f(f2.spec[1])); f2.fmt = old
        elif stmt_end:
            q, ans = tgt.stored()[1], G.pick(rng, ACKS)
        else:
            if nm and rng.random() < 0.3:
                old = tgt.fmt; tgt.fmt = dict(old, **name_fmt(nm)); q = tgt._f(G.pick(rng, tgt.spec[2])); tgt.fmt = old
            else:
                q = tgt.ask()
            ans = G.pick(rng, G.ABSTAIN) if absent else G4.your(tgt.stored()[1])
            if not absent:
                if tgt.can_update() and rng.random() < 0.35:
                    if nm and rng.random() < 0.5:
                        old = tgt.fmt; tgt.fmt = dict(old, **name_fmt(nm)); (_, new_s), _ = tgt.update(); tgt.fmt = old
                        new_ans = tgt._f(tgt.spec[1], tgt.new_value)
                    else:
                        (_, new_s), _ = tgt.update(); new_ans = new_s
                    corr = "Actually, " + new_s[0].lower() + new_s[1:]
                    ans = G4.your(new_ans)
                stmts.append((tgt.stored()[1], corr is None))
        turns = self._filler(int(rng.integers(1, 21)))
        gold_txt, all_txt = set(), set()
        for s_, g_ in stmts:
            st = s_.encode(); turns.insert(int(rng.integers(0, len(turns) + 1)), (st, G.pick(rng, ACKS).encode())); all_txt.add(st)
            if g_:
                gold_txt.add(st)
        if corr:
            own = next(j for j, t in enumerate(turns) if t[0] == tgt.stored()[1].encode())
            turns.insert(int(rng.integers(own + 1, len(turns) + 1)), (corr.encode(), G.pick(rng, ACKS).encode()))
            gold_txt.add(corr.encode()); all_txt.add(corr.encode())
        # which turns scrolled out decides what the memory returns (only those; the engine skips in-window turns)
        items, kept = [], self._win([], turns, q, ans)[3]
        for _ in range(2):
            out_idx = [j for j, t in enumerate(turns) if j < len(turns) - kept and t[0] in all_txt]
            pick = [j for j in out_idx if turns[j][0] in gold_txt] + [j for j in out_idx if turns[j][0] not in gold_txt and rng.random() < 0.6]
            items = [W.Item(turns[j][0], turns[j][1], "turn", j + 1) for j in pick]
            rng.shuffle(items)
            items = self._crowd(items, q, ans) if "crowd" in self.parts else items
            c, ce, pl, kept2, sp = self._win(items, turns, q, ans)
            if kept2 == kept:
                break
            kept = kept2
        gold = [sp["turns"][j] for j, t in enumerate(turns) if t[0] in gold_txt and j in sp["turns"]]
        gold += [sp["items"][k] for k, it in enumerate(items) if it.q in gold_txt]
        return self._done(c, ce, pl, gold, [sp["sink"]] if absent else [])

    def buried_(self):
        rng = self.rng
        T = int(rng.integers(600, min(1900, self.ctx - 200))); depth = rng.uniform(0.1, 0.9)
        items = [self._fact_item(self._rand_qa())] if rng.random() < 0.5 else []
        post, kind = [], None
        crowd_later = "crowd" in self.parts
        real = self._real_evidence() if ("realev" in self.parts and rng.random() < 0.6) else None
        if "alpacabury" in self.parts and rng.random() < 0.6:
            q, target, facts = self._alpaca_evidence()
        elif real is not None:
            q, target, facts = real
        elif rng.random() < 0.6 and not ({"realev", "alpacabury"} & self.parts):
            i = self._rand_i(); q, target = self.qa[i]
            if rng.random() < 0.3:
                q = perturb_question(q, rng, self.donor)
            facts = [] if rng.random() < 0.15 else [(target, True)]
            nb = self.nbrs[i, :int(rng.integers(1, 3))] if self.nbrs is not None else [self._rand_i() for _ in range(2)]
            facts += [(self.qa[int(j)][1], False) for j in nb if self.qa[int(j)][1].strip()]
        else:
            w = self.skills[int(rng.integers(len(self.skills)))]
            q, target = B(w["q"]), B(w["target"]); kind = w["kind"]
            tw = words(w["target"]); ov = [len(words(cc[1]) & tw) for cc in w["cands"]] or [0]
            good = (lambda j: ov[j] > 0) if kind in ("list", "clarify") else (lambda j: ov[j] == max(ov) > 0)
            facts = [(B(cc[1]), kind in ANSWERABLE | {"clarify"} and good(j)) for j, cc in enumerate(w["cands"])]
            post = [(B(t[0]), B(t[1])) for t in w["turns"]]
        if crowd_later:
            items = self._crowd(items, q, target)
        pieces = [(self.enc.encode(b" " + f + b" ")[0][:300].astype(np.int64), g) for f, g in facts]
        if "longdocs" in self.parts:                      # what the memory block really takes, not half the budget
            need = len(self.enc.encode(E.render(q, target))[0]) + sum(min(len(self.enc.encode(it.text())[0]), W.ITEM_CAP) for it in items) + 96
        else:
            need = len(self.enc.encode(E.render(q, target))[0]) + self.mem_codes // 2 + 64
        T = min(T, self.ctx - need)                       # the reply must always fit after the document
        for _ in range(4):
            doc, spans = self._doc_with(pieces, depth, T)
            codes, ce, pl, dspans, ispans, sink = self._with_doc(doc, spans, items, post, q, target)
            if len(codes) <= self.ctx + 1 and codes[-1] == E.EOS:
                break
            T = int(T * 0.7)
        g_sp = [dspans[k] for k, (_, g) in enumerate(pieces) if g]
        if kind == "unknown":
            return self._done(codes, ce, pl, [], [sink])
        if kind == "clarify":
            return self._done(codes, ce, pl, [], g_sp)
        return self._done(codes, ce, pl, g_sp)

    def docqa_(self):
        """Read a real document: a sentence at 10-90% depth, asked by its first half; the reply is its second half."""
        rng = self.rng
        for _ in range(20):
            d = self.docs[int(rng.integers(len(self.docs)))]
            n = int(rng.integers(2200, min(6500, len(d))))
            s0 = int(rng.integers(0, max(1, len(d) - n))); text = d[s0:s0 + n]
            sents = [m for m in re.finditer(rb"[A-Z][^.!?\n]{40,200}[.!?]", text)]
            if not sents:
                continue
            depth = rng.uniform(0.1, 0.9)
            m = min(sents, key=lambda m: abs(m.start() / len(text) - depth))
            s = m.group(0); ws = s.split(b" ")
            if len(ws) < 8:
                continue
            k = len(ws) // 2
            first, rest = b" ".join(ws[:k]), b" ".join(ws[k:])
            pre, post_t = text[:m.start()] + first + b" ", text[m.end():]
            c_pre, c_rest, c_post = (self.enc.encode(x)[0].astype(np.int64) for x in (pre, rest, post_t))
            doc = np.concatenate([c_pre, c_rest, c_post])
            q = G.pick(rng, [b'In the text, how does this sentence end: "%s ..."?', b'Finish this sentence from the text: "%s ..."',
                             b'What comes after "%s" in the text?']) % first
            if len(doc) + 400 > self.ctx:
                continue
            codes, ce, pl, dspans, _, _ = self._with_doc(doc, [(len(c_pre), len(c_pre) + len(c_rest))], [], [], q, rest)
            return self._done(codes, ce, pl, dspans)
        return self.rehearse_()

    def story_(self):
        return self._story_book() if self.rng.random() < 0.5 else self._story_world()

    def _story_book(self):
        """The next passage of a whole book; earlier pieces of the same book that share names with the recent text,
        or that name someone the next passage brings back, are in memory as [turn N] items."""
        rng = self.rng
        for _ in range(20):
            b = self.books[int(rng.integers(len(self.books)))]
            if len(b) < 60_000:
                continue
            p = int(rng.integers(40_000, len(b) - 4_000))
            p = b.find(b". ", p) + 2
            R = int(rng.integers(2_500, 4_500)); recent = b[p - R:p]
            nxt = b[p:p + int(rng.integers(500, 1000))]; nxt = nxt[:nxt.rfind(b". ") + 1] or nxt
            names_recent = {n for n in re.findall(rb"\b[A-Z][a-z]{2,}\b", recent)}
            names_next = {n for n in re.findall(rb"\b[A-Z][a-z]{2,}\b", nxt)} - names_recent
            earlier = b[max(0, p - R - 120_000):p - R]
            pieces = split_text(earlier)
            score = [(len(set(re.findall(rb"\b[A-Z][a-z]{2,}\b", x)) & (names_recent | names_next)), j) for j, x in enumerate(pieces)]
            score = [s for s in score if s[0] > 0]
            if not score:
                continue
            score.sort(reverse=True)
            pick = [j for _, j in score[:6]]
            base = int(rng.integers(5, 30))
            items = [W.Item(b"Continue the story.", pieces[j], "turn", base + j) for j in pick]
            turns = [(b"Continue the story.", recent)]
            c, ce, pl, _, sp = self._win(items, turns, b"Continue the story.", nxt)
            gold = [sp["items"][k] for k, it in enumerate(items) if set(re.findall(rb"\b[A-Z][a-z]{2,}\b", it.a)) & names_next]
            return self._done(c, ce, pl, gold)
        return self.rehearse_()

    def _story_world(self):
        rng = self.rng
        w = SW.World(rng)
        n = int(rng.integers(3, 12))
        for _ in range(n):
            w.next_scene()
        turns = [(b"Continue the story.", s[0].encode()) for s in w.scenes[:-1]]
        r = rng.random()
        facts_all = [W.Item(b"Story notes.", f.encode(), "fact") for c in range(len(w.chars)) for f in w.bible_facts(c)]
        if r < 0.5:                                      # the next scene, with the facts it uses in memory
            text, used, *_ = w.scenes[-1]
            need = {w.fact(c, f) for c, f in used}
            items = [W.Item(b"Story notes.", f.encode(), "fact") for f in need]
            items += [facts_all[int(j)] for j in rng.choice(len(facts_all), min(3, len(facts_all)), replace=False)]
            rng.shuffle(items)
            c, ce, pl, _, sp = self._win(items, turns, b"Continue the story.", text)
            gold = [sp["items"][k] for k, it in enumerate(items) if it.a.decode() in need]
            return self._done(c, ce, pl, gold)
        if r < 0.85:                                     # a question about an earlier scene
            j = int(rng.integers(len(w.scenes)))
            q, a = w.question(j)
            items = [W.Item(b"Continue the story.", w.scenes[j][0].encode(), "turn", j + 1)] if rng.random() < 0.5 else []
            items += [facts_all[int(i)] for i in rng.choice(len(facts_all), min(4, len(facts_all)), replace=False)]
            rng.shuffle(items)
            all_turns = [(b"Continue the story.", s[0].encode()) for s in w.scenes]
            c, ce, pl, kept, sp = self._win(items, all_turns, q, a)
            aw = words(a)
            gold = [sp["items"][k] for k, it in enumerate(items) if len(words(it.a) & aw) >= max(1, len(aw) - 1)]
            gold += [sp["turns"][i] for i in sp["turns"] if len(words(all_turns[i][1]) & aw) >= max(1, len(aw) - 1)]
            return self._done(c, ce, pl, gold)
        wrong, right, ci = w.contradiction()            # repair a sentence that breaks the story notes
        items = [W.Item(b"Story notes.", f.encode(), "fact") for f in w.bible_facts(ci)]
        q = f'Does this fit the story? "{wrong}"'
        a = f"No. {right}"
        c, ce, pl, _, sp = self._win(items, turns, q, a)
        gold = [sp["items"][k] for k, it in enumerate(items) if b"from" in it.a or b"comes" in it.a]
        return self._done(c, ce, pl, gold)

    def rehearse_(self):
        j = int(self.rng.integers(0, len(self.rehearse_codes) - self.ctx - 1))
        seq = self.rehearse_codes[j:j + self.ctx + 1].astype(np.int64)
        return Window(seq, 0, 0)

    MAKERS = {"qa": "qa_", "skills": "skills_", "convo": "convo_", "recall": "recall_", "buried": "buried_",
              "docqa": "docqa_", "story": "story_", "rehearse": "rehearse_", "longconv": "longconv_"}

    def make(self, kind: str) -> Window:
        """A window of this kind; one whose reply did not fit (a very long answer beside a document) is replaced."""
        for _ in range(8):
            w = getattr(self, self.MAKERS[kind])()
            if kind == "rehearse" or (len(w.codes) - w.prompt_len >= 2 and w.codes[-1] == E.EOS):
                return w
        return w
