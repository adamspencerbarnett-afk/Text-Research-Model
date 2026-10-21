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

Parts added 20 Oct (the 5M loop, recipes/LOOP.json): afterno = 30% of qa / skills / recall / any-length windows have
1-2 earlier exchanges where a never-told personal question was rightly answered "I don't have that"; the target is
unchanged (an answer when the window or memory holds it), so one refusal does not make the next reply refuse.
twomem = 20% of recall windows: a told personal fact plus the stored real fact it leads to (as the two-hop search
brings it: "My favorite song is X." + [fact] "Who wrote the song X?" -> "Who wrote my favorite song?"); 15% without
the stored fact and 15% without the told fact, both answered "I don't have that". twomem1 = the same without the
"told fact missing" case (it taught refusing while the right stored fact was in the block; c1 refused more).
twomem2 = 10% of recall windows, no "not told" case, and the answer joins both sources ('"X" was written by Dottie
Rambo.', 'The current leader of Saudi Arabia is ...'): c1/c1b answered story questions from unrelated stored facts
after twomem windows whose answer was always the stored fact's own text. Store rows of the train split only
(chat22/tests/suite_v2.store_split), statement and question wordings other than the test's.
reason = 30% of recall windows (reason15: 15%; reasoncr: 15%, counting and recaps only): counting what was told (1-5 items), the first thing told, a recap of 3-4 facts, with
wordings other than the test's.
multi = messages with several parts (multi2: the same, with look-alike stored facts as decoys in the two-fact windows) (20 Oct audit: memory facts were always told one per message and asked one per
message): 35% of skills windows become one of: several facts told in one message then one asked; two told facts asked
in one message; one told and one never told (answer the first, say the second is not known); a new statement plus a
question in one message; a quoted note with two facts and two questions. 15% of qa windows ask two real stored facts
(knowledge_real, train split) in one message, both facts in the memory block, answered in order.

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
        turns = self._afterno(turns, B(r["q"]) + b" " + b" ".join(it.q + b" " + it.a for it in items))
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

    def _afterno(self, turns: list, avoid: bytes = b"") -> list:
        """afterno (20 Oct): with p 0.3, 1-2 earlier refusals of never-told personal questions as the latest turns. The
        asked entity appears nowhere in the window, so "I don't have that" was the right reply there."""
        rng = self.rng
        if "afterno" not in self.parts or rng.random() >= 0.3:
            return turns
        seen = b" ".join([avoid] + [u + b" " + a for u, a in turns]).lower()
        out = list(turns)
        for _ in range(1 if rng.random() < 0.7 else 2):
            for _ in range(10):
                f = G.Fact(rng)
                name = f.ent["id"].split()[-1].lower().encode()
                if name not in seen:
                    out.append((f.ask().encode(), G.pick(rng, G.ABSTAIN).encode()))
                    break
        return out

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
        if ("multi" in self.parts or "multi2" in self.parts) and rng.random() < 0.15:
            return self._store_pair()
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
        c, ce, pl, _, sp = self._win(items, self._afterno([], q + b" " + b" ".join(it.q + b" " + it.a for it in items)), q, ans)
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
        if w is None and ("multi" in self.parts or "multi2" in self.parts) and rng.random() < 0.35:
            return self._multi()
        w = w or self.skills[int(rng.integers(len(self.skills)))]
        t0 = int(rng.integers(1, 40)); items = []
        stored = "stored" in self.parts and w["kind"] not in ("update", "update_earlier") and rng.random() < 0.5   # v2.3: half uploaded
        for c in w["cands"]:                              # told in this order: the newest is nearest the message
            items.append(W.Item(B(c[0]), B(c[1]), "fact" if stored else "turn", 0 if stored else t0)); t0 += int(rng.integers(1, 15))
        order = list(range(len(items))); rng.shuffle(order)   # relevance order is not the told order
        ranked = self._crowd([items[j] for j in order], w["q"], w["target"])
        turns = [(B(t[0]), B(t[1])) for t in w["turns"]]
        if w["kind"] not in ("followup", "clarify", "combo"):     # these read the latest turn: a refusal there would change them
            turns = self._afterno(turns, B(w["q"]) + b" " + b" ".join(it.q + b" " + it.a for it in items))
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

    # ------------------------------------------------------------------ 20 Oct: two memories, reasoning over the chat
    TM_TRAIN = [   # (store question pattern, statements, chained questions): wordings the test (suite_v2.TWOMEM) never uses
        (r"^who wrote the song (.+?)\??$", ['I love the song "{x}".', 'The song "{x}" is stuck in my head.', 'My go-to song is "{x}".'],
         ["Who wrote the song I love?", "Who is the writer of the song stuck in my head?", "Who wrote my go-to song?"]),
        (r"^who sings the song (.+?)\??$", ['I keep humming "{x}".', 'My favourite tune is "{x}".', 'I just heard "{x}" on the radio.'],
         ["Who sings the song I keep humming?", "Who is the singer of my favourite tune?", "Who sings the song I heard on the radio?"]),
        (r"^where was the movie (.+?) filmed\??$", ["I just rewatched {X}.", "We're watching {X} tonight.", "{X} is my all-time favourite film."],
         ["Where was the movie I rewatched filmed?", "Where did they film the movie we're watching tonight?", "Where was my all-time favourite film shot?"]),
        (r"^when did the movie (.+?) come out\??$", ["My kids love {X}.", "I saw {X} at the cinema.", "{X} is the movie my dad likes."],
         ["When did the movie my kids love come out?", "When was the movie I saw at the cinema released?", "What year did my dad's movie come out?"]),
        (r"^what is the meaning of the name (.+?)\??$", ["We named our son {X}.", "My best friend is called {X}.", "My grandmother's name was {X}."],
         ["What does our son's name mean?", "What is the meaning of my best friend's name?", "What did my grandmother's name mean?"]),
        (r"^where does the last name (.+?) come from\??$", ["My surname is {X}.", "My wife's maiden name is {X}.", "Our family name is {X}."],
         ["Where does my surname come from?", "What is the origin of my wife's maiden name?", "Where does our family name come from?"]),
        (r"^who is the current leader of (.+?)\??$", ["My parents live in {X}.", "I was born in {X}.", "I'm visiting {X} this summer."],
         ["Who leads the country my parents live in?", "Who is the current leader of the country I was born in?", "Who is in charge of the country I'm visiting?"]),
        (r"^what is the name of the biggest city in (.+?)\??$", ["My aunt moved to {X}.", "I'm studying in {X}.", "My pen pal is from {X}."],
         ["What is the biggest city in the country my aunt moved to?", "Which is the largest city in the country I'm studying in?", "What's the biggest city where my pen pal is from?"]),
    ]

    def _tm_rows(self):
        if not hasattr(self, "_tm"):
            from .tests.suite_v2 import store_split
            rx = [re.compile(p, re.I) for p, _, _ in self.TM_TRAIN]
            self._tm = []
            store = self.paths.find("knowledge/knowledge_real.jsonl", ROOT / "data_real" / "shared" / "knowledge" / "knowledge_real.jsonl")
            want = "train" if self.split == "train" else "test"
            for line in open(store, encoding="utf-8"):
                r = json.loads(line)
                if r.get("source") != "qa_memory":
                    continue
                q, a = r["q"].strip(), r["a"].replace("\xa0", " ").strip()
                if not a or len(a.rstrip(".").split()) > 3 or store_split(q) != want:
                    continue
                for j, x in enumerate(rx):
                    m = x.match(q)
                    if m:
                        self._tm.append((j, m.group(1).strip(), q, a))
                        break
        return self._tm

    TM_JOIN = ['"{x}" was written by {a}.', '"{x}" is sung by {a}.', "{X} was filmed in {a}.", "{X} came out {a3}.",
               "The name {X} means {a}.", "The name {X} comes from {a5}.", "The current leader of {X} is {a}.",
               "The biggest city in {X} is {a}."]

    def _tm_join(self, j, x, a):
        a0 = a.rstrip(".")
        X = x.title() if x.islower() else x
        a3 = a0[0].lower() + a0[1:] if a0.lower().startswith(("in ", "on ", "circa")) else (("in " if a0.isdigit() else "on ") + a0)
        a5 = a0[3:] if a0.lower().startswith("in ") else a0
        return self.TM_JOIN[j].format(x=x, X=X, a=a0, a3=a3, a5=a5)

    def _twomem(self, joined: bool = False):
        rng = self.rng
        rows = self._tm_rows()
        j, x, q, a = rows[int(rng.integers(len(rows)))]
        _, stms, asks = self.TM_TRAIN[j]
        k = int(rng.integers(len(stms)))
        stm = stms[k].format(x=x, X=x.title() if x.islower() else x)
        ask = asks[k]
        r = rng.random()
        no_fact, no_told = r < 0.15, 0.15 <= r < 0.30 and not ({"twomem1", "twomem2"} & self.parts)   # twomem1/2: never "not told"
        turns = self._filler(int(rng.integers(0, 4)))
        if not no_told:
            turns.insert(int(rng.integers(0, len(turns) + 1)), (B(stm), G.pick(rng, ACKS).encode()))
        same = [rw for rw in rows if rw[0] == j and rw[2] != q]
        n_look = min(len(same), int(rng.integers(0, 3)))
        items = [W.Item(B(same[int(i)][2][0].upper() + same[int(i)][2][1:]), B(same[int(i)][3]), "fact")
                 for i in (rng.choice(len(same), n_look, replace=False) if n_look else [])]
        if not no_fact:
            items.insert(int(rng.integers(0, len(items) + 1)), W.Item(B(q[0].upper() + q[1:]), B(a), "fact"))
        target = G.pick(rng, G.ABSTAIN) if (no_fact or no_told) else (self._tm_join(j, x, a) if joined else a)
        c, ce, pl, kept, sp = self._win(items, turns, ask, target)
        if no_fact or no_told:
            return self._done(c, ce, pl, [], [sp["sink"]])
        gold = [sp["items"][i] for i, it in enumerate(items) if it.a == B(a)]
        gold += [sp["turns"][i] for i, t in enumerate(turns) if t[0] == B(stm) and i in sp["turns"]]
        return self._done(c, ce, pl, gold)

    COUNT_CATS = [  # (statement templates, category, ask wordings): wordings the test (suite_v2.reason) never uses
        (["I've got a {v} called {n}.", "We also have a {v} named {n}.", "There's our {v}, {n}."], "pets",
         ["How many pets have I mentioned?", "Count the pets I told you about.", "How many animals do I have?"]),
        (["I drive a {v} to work.", "We also keep an old {v}.", "My other car is a {v}."], "cars",
         ["How many cars have I mentioned?", "How many vehicles do I have?"]),
        (["My kid {n} just turned {a}.", "{n}, my youngest, is {a}.", "Our child {n} is {a} now."], "kids",
         ["How many kids have I mentioned?", "How many children did I say I have?"]),
        (["I play the {v}.", "I'm also learning the {v}.", "I can play the {v} a little."], "instruments",
         ["How many instruments do I play?", "How many instruments have I mentioned?"]),
        (["I speak {v}.", "I'm fluent in {v}.", "I also know some {v}."], "languages",
         ["How many languages do I speak?", "How many languages did I mention?"]),
        (["Last year I visited {v}.", "I've been to {v} too.", "I once spent a week in {v}."], "cities",
         ["How many cities have I visited?", "How many places did I say I've been to?"]),
    ]
    COUNT_POOLS = {"instruments": ["violin", "piano", "guitar", "drums", "flute", "cello", "trumpet", "harp", "banjo", "saxophone"],
                   "languages": ["Spanish", "French", "German", "Italian", "Japanese", "Portuguese", "Hindi", "Arabic", "Korean", "Dutch"]}
    NUM_WORDS = {1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}

    @staticmethod
    def _lc(s: str) -> str:
        """Lower-case the first word only when it is not a name ("Your sister ..." -> "your sister ...", "Kim ..." stays)."""
        w = s.split(" ", 1)[0]
        return s[0].lower() + s[1:] if w in ("Your", "The", "Parking", "My", "Our") else s

    def _reason(self):
        rng = self.rng
        r = rng.random()
        if r < 0.45:                                     # counting
            stms_t, cat, asks = self.COUNT_CATS[int(rng.integers(len(self.COUNT_CATS)))]
            k = int(rng.integers(1, 6))
            if cat == "pets":
                vals = [str(v) for v in rng.choice(G.PETS, k, replace=False)]
            elif cat == "cars":
                vals = [f"{c} {m}" for c, m in zip(rng.choice(G.COLOURS, k, replace=False), rng.choice(G.CARS, k, replace=False))]
            elif cat == "cities":
                vals = [str(v) for v in rng.choice(G.CITIES, k, replace=False)]
            elif cat == "kids":
                vals = [str(v) for v in rng.choice(G.NAMES, k, replace=False)]
            else:
                vals = [str(v) for v in rng.choice(self.COUNT_POOLS[cat], k, replace=False)]
            names = [str(v) for v in rng.choice(G.PETNAMES, k, replace=False)]
            stm = [G.pick(rng, stms_t).format(v=v, n=(v if cat == "kids" else n), a=int(rng.integers(2, 17))) for v, n in zip(vals, names)]
            listed = ", ".join(vals[:-1]) + (" and " if k > 1 else "") + vals[-1]
            noun = cat if k > 1 else {"kids": "kid", "cities": "city"}.get(cat, cat[:-1])
            target = f"You mentioned {self.NUM_WORDS[k]} {noun}: {listed}."
            ask = G.pick(rng, asks)
        elif r < 0.75 and "reasoncr" in self.parts:      # reasoncr: a recap instead of 'the first thing told'
            r = 0.8
        if 0.45 <= r < 0.75:                             # the first thing told
            a = G.Fact(rng)
            b = G._other(rng, {a.ent["id"]}, (a.field,))
            stm = [a.stored()[1], b.stored()[1]]
            if rng.random() < 0.4:
                stm.append(G._other(rng, {a.ent["id"], b.ent["id"]}).stored()[1])
            first = G4.your(a.stored()[1])
            target = "The first thing you told me was that " + self._lc(first)
            ask = G.pick(rng, ["What's the first thing I mentioned?", "What did I say first?", "Which fact did I give you first?"])
        elif r >= 0.75:                                  # a recap
            fs = [G.Fact(rng)]
            want_n = int(rng.integers(3, 5))
            while len(fs) < want_n:
                fs.append(G._other(rng, {f.ent["id"] for f in fs}))
            stm = [f.stored()[1] for f in fs]
            parts = [G4.your(x).rstrip(".") for x in stm]
            target = "You told me that " + "; ".join(self._lc(x) for x in parts) + "."
            ask = G.pick(rng, ["Give me a recap of what I've said.", "What do you know about me so far?", "Recap what I told you, please."])
        told = {B(x) for x in stm}
        for n_fill in (2, 1, 0):
            turns = []
            for x in stm:
                turns.append((B(x), G.pick(rng, ACKS).encode()))
                turns += self._filler(int(rng.integers(0, n_fill + 1)))
            c, ce, pl, kept, sp = self._win([], turns, ask, target)
            if kept == len(turns):
                break
        gold = [sp["turns"][i] for i, t in enumerate(turns) if t[0] in told and i in sp["turns"]]
        return self._done(c, ce, pl, gold)

    # ------------------------------------------------------------------ 20 Oct: messages with several parts (multi)
    FIELD_NOUN = {"city": "home town", "job": "job", "food": "favourite food", "colour": "favourite colour", "sport": "favourite sport",
                  "band": "favourite band", "drink": "favourite drink", "car": "car", "phone": "phone number", "birthday": "birthday",
                  "pet": "pet's name", "age": "age", "allergy": "allergy", "hobby": "hobby", "work": "workplace", "surname": "last name",
                  "book": "favourite book", "school": "school", "language": "language", "instrument": "instrument", "address": "address",
                  "email": "email address"}
    TH_NOUN = {"when": "time", "where": "place", "code": "code", "price": "price", "opens": "opening time", "wifi": "wifi password",
               "parking": "parking spot", "deadline": "deadline"}
    JOIN2 = ["{A} Also, {b}", "{A} Oh, and {b}", "{A} And {b}", "Two things: {A} {B}", "Quick update: {A} {B}", "{A} Another thing: {b}"]
    JOIN3 = ["A few things about me: {A} {B} {C}", "{A} {B} Also, {c}", "Let me catch you up. {A} {B} {C}", "So, {a} {B} And {c}"]
    ASK2 = ["{Q1} And {q2}", "Two questions: {Q1} {Q2}", "{Q1n}, and {q2}", "{Q1} Also, {q2}", "Can you tell me two things? {Q1} {Q2}"]
    STORE2 = ["{Q1} Also, {q2}", "Quick quiz: {Q1} {Q2}", "{Q1} And {q2}", "Two questions for you. {Q1} {Q2}"]
    WITHQ = ["{S} Anyway, {q}", "{S} By the way, {q}", "{S} {Q}", "{S} Oh, and {q}"]
    SENDERS = ["landlord", "coach", "manager", "neighbour", "kid's school", "dentist's office", "team lead"]
    NOTE = ["Here's a text from my {who}: \"{A} {B}\" {Q1n} and {q2}", "My {who} sent this: \"{A} {B}\" Can you tell me {q1i} and {q2i}"]

    @staticmethod
    def _lcw(s: str) -> str:
        """Lower-case a leading 'My / The / Our / Your / Parking / What / When ...' (sentence glue), never a name."""
        w = s.split(" ", 1)[0]
        lead = {"My", "The", "Our", "Your", "Parking", "What", "When", "Where", "Which", "Who", "How", "Can", "Is", "Do", "Does"}
        return s[0].lower() + s[1:] if w in lead else s

    def _topic(self, f) -> str:
        if f.kind == "p":
            own = G4.your(f.ent["e"])
            return f"{own}'s {self.FIELD_NOUN.get(f.field, f.field)}"
        return f"the {f.ent['t']} {self.TH_NOUN.get(f.field, 'details')}"

    def _told(self, stmts: list):
        """Told statements as turns in the window (60%) or, as if they scrolled out, as [turn N] memory items (40%)."""
        rng = self.rng
        if rng.random() < 0.4:
            t0 = int(rng.integers(1, 30))
            items = [W.Item(B(x), G.pick(rng, ACKS).encode(), "turn", t0 + 3 * j) for j, x in enumerate(stmts)]
            return self._filler(int(rng.integers(0, 3))), items
        turns = []
        for x in stmts:
            turns += self._filler(int(rng.integers(0, 2)))
            turns.append((B(x), G.pick(rng, ACKS).encode()))
        return turns + self._filler(int(rng.integers(0, 2))), []

    def _gold_for(self, sp, items, turns, values):
        vals = [v.lower().encode() for v in values]
        hit = lambda t: any(v in t.lower() for v in vals)
        g = [sp["items"][k] for k, it in enumerate(items) if sp["items"][k] and hit(it.q + b" " + it.a)]
        return g + [sp["turns"][i] for i in sp["turns"] if hit(turns[i][0] + b" " + turns[i][1])]

    def _multi(self):
        rng = self.rng
        r = rng.random()
        fs = [G.Fact(rng)]
        while len(fs) < 3:
            fs.append(G._other(rng, {f.ent["id"] for f in fs}))
        st = [f.stored()[1] for f in fs]
        if r < 0.25:                                     # several facts in one message, one asked later
            k = 2 if rng.random() < 0.5 else 3
            msg = (G.pick(rng, self.JOIN2).format(A=st[0], B=st[1], b=self._lcw(st[1])) if k == 2 else
                   G.pick(rng, self.JOIN3).format(A=st[0], a=self._lcw(st[0]), B=st[1], C=st[2], c=self._lcw(st[2])))
            ask = fs[int(rng.integers(k))]
            turns, items = self._told([msg])
            q, target, vals = ask.ask(), G4.your(ask.stored()[1]), [ask.value]
        elif r < 0.50:                                   # two told facts asked in one message
            turns, items = self._told(st[:2] if rng.random() < 0.6 else [G.pick(rng, self.JOIN2).format(A=st[0], B=st[1], b=self._lcw(st[1]))])
            q1, q2 = fs[0].ask(), fs[1].ask()
            if rng.random() < 0.5:
                q1, q2 = q2, q1; fs[0], fs[1] = fs[1], fs[0]
            q = G.pick(rng, self.ASK2).format(Q1=q1, Q2=q2, q2=self._lcw(q2), Q1n=q1.rstrip("?"))
            target = G4.your(fs[0].stored()[1]) + " " + G4.your(fs[1].stored()[1])
            vals = [fs[0].value, fs[1].value]
        elif r < 0.70:                                   # one told, one never told
            turns, items = self._told([st[0]])
            known, unknown = fs[0], fs[1]
            first_known = rng.random() < 0.5
            q1, q2 = (known.ask(), unknown.ask()) if first_known else (unknown.ask(), known.ask())
            q = G.pick(rng, self.ASK2).format(Q1=q1, Q2=q2, q2=self._lcw(q2), Q1n=q1.rstrip("?"))
            ans = G4.your(known.stored()[1])
            target = (f"{ans} But I don't have {self._topic(unknown)} in my memory." if first_known else
                      f"I don't have {self._topic(unknown)} in my memory. {ans}")
            vals = [known.value]
        elif r < 0.85:                                   # a new statement and a question in one message
            turns, items = self._told([st[0]])
            q = G.pick(rng, self.WITHQ).format(S=st[1], Q=fs[0].ask(), q=self._lcw(fs[0].ask()))
            target = G.pick(rng, ["Noted.", "Got it.", "Thanks, noted."]) + " " + G4.your(fs[0].stored()[1])
            vals = [fs[0].value]
        else:                                            # a quoted note with two facts, both asked
            ev = [G.Fact(rng, "t")]
            ev.append(G._other(rng, {ev[0].ent["id"]}))
            while ev[1].kind != "t":
                ev[1] = G.Fact(rng, "t")
            sa, sb = ev[0].stored()[1], ev[1].stored()[1]
            q1, q2 = ev[0].ask(), ev[1].ask()
            q = G.pick(rng, self.NOTE).format(who=G.pick(rng, self.SENDERS), A=sa, B=sb, Q1n=q1.rstrip("?"), q2=self._lcw(q2),
                                              q1i=self._lcw(ev[0].stored()[0]).rstrip("?"), q2i=self._lcw(ev[1].stored()[0]).rstrip("?") + "?")
            turns, items = self._filler(int(rng.integers(0, 3))), []
            target, vals = f"{sa} {sb}", [ev[0].value, ev[1].value]
            c, ce, pl, _, sp = self._win(items, turns, q, target)
            return self._done(c, ce, pl)                 # the facts are in the message itself
        c, ce, pl, _, sp = self._win(items, turns, q, target)
        return self._done(c, ce, pl, self._gold_for(sp, items, turns, vals))

    def _store_rows(self):
        if not hasattr(self, "_st2"):
            from .tests.suite_v2 import store_split
            want = "train" if self.split == "train" else "test"
            store = self.paths.find("knowledge/knowledge_real.jsonl", ROOT / "data_real" / "shared" / "knowledge" / "knowledge_real.jsonl")
            self._st2 = []
            for line in open(store, encoding="utf-8"):
                r = json.loads(line)
                if r.get("source") != "qa_memory":
                    continue
                q, a = r["q"].strip(), r["a"].replace("\xa0", " ").strip()
                if q.endswith("?") and 0 < len(a.rstrip(".").split()) <= 4 and len(q) < 90 and store_split(q) == want:
                    self._st2.append((q[0].upper() + q[1:], a if a.endswith(".") else a + "."))
        return self._st2

    def _store_pair(self):
        rng = self.rng
        rows = self._store_rows()
        (q1, a1), (q2, a2) = [rows[int(i)] for i in rng.choice(len(rows), 2, replace=False)]
        items = [W.Item(B(q1), B(a1), "fact"), W.Item(B(q2), B(a2), "fact")]
        if "multi2" in self.parts:                       # 20 Oct: decoys that look like one of the asked facts (c2 picked the
            if not hasattr(self, "_st2_pre"):            # wrong one of two similar stored facts on 4 of 48 direct questions)
                self._st2_pre = {}
                for r_ in rows:
                    self._st2_pre.setdefault(" ".join(r_[0].lower().split()[:4]), []).append(r_)
            for qq in (q1, q2):
                if rng.random() < 0.5:
                    same = [r_ for r_ in self._st2_pre.get(" ".join(qq.lower().split()[:4]), []) if r_[0] not in (q1, q2)]
                    if same:
                        o = same[int(rng.integers(len(same)))]
                        items.append(W.Item(B(o[0]), B(o[1]), "fact"))
        elif rng.random() < 0.4:
            o = rows[int(rng.integers(len(rows)))]
            items.append(W.Item(B(o[0]), B(o[1]), "fact"))
        rng.shuffle(items)
        q = G.pick(rng, self.STORE2).format(Q1=q1, Q2=q2, q2=self._lcw(q2))
        target = f"To your first question: {a1} To your second: {a2}"
        c, ce, pl, _, sp = self._win(items, [], q, target)
        gold = [sp["items"][k] for k, it in enumerate(items) if sp["items"][k] and it.a in (B(a1), B(a2))]
        return self._done(c, ce, pl, gold)

    def recall_(self):
        rng = self.rng
        if ("twomem" in self.parts or "twomem1" in self.parts) and rng.random() < 0.2:
            return self._twomem()
        if "twomem2" in self.parts and rng.random() < 0.1:
            return self._twomem(joined=True)
        if "reason" in self.parts and rng.random() < 0.3:
            return self._reason()
        if "reason15" in self.parts and rng.random() < 0.15:  # 20 Oct: half the dose (c1c: 100% on its tests, but documents and updates fell)
            return self._reason()
        if "reasoncr" in self.parts and rng.random() < 0.15:  # 20 Oct: counting and recaps only (no 'first thing told')
            return self._reason()
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
        turns = self._afterno(turns, B(q) + b" " + b" ".join(all_txt))
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
