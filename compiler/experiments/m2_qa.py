"""Design v2, first screen: an exchange memory keyed on the question (M2), with the compiler as
the key generator, on the chat-formatted Q&A data.

    python compiler/experiments/m2_qa.py DATA_QA WORK --native ./cv0 --budget 600 --out results/m2_qa.json

What it does
  1. Splits the training text into exchanges (User: question / Assistant: answer).
  2. M2: the compiler's scanner turns each question into a canonical key, the set of its
     content words (lower-cased, stopwords dropped). An inverted index finds the nearest
     stored exchange by idf-weighted word overlap; an exact-key table answers directly.
  3. M1: exact bigram and trigram tables over the training codes, as before.
  4. M3: the usual small core, trained on windows of [nearest other exchange] + SEP +
     [this exchange], so it learns to use a retrieved exchange, never seeing itself.
  5. Evaluation, four groups of questions, each answered through the compiler:
     seen (training questions: exact key hits memory), paraphrase (held-out questions whose
     nearest training key overlaps strongly), unseen (the rest of held-out), and added
     (held-out pairs put into M2 after training, no training). Scores: word-level F1 against
     the reference answer, exact match, and for the memory path the retrieval hit rate.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_v0 as cv  # noqa: E402
import ladder1  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import train_lm  # noqa: E402

STOP = set("""a an the of to in on at for by with and or but if then than that this these those is are was were be been being
am do does did have has had will would can could should may might must not no yes it its i you he she we they me him her us them
my your his our their what which who whom whose when where why how as from into about over under again further once here there
all any both each few more most other some such only own same so very s t just now please give write describe explain list
name tell make create generate provide following sentence""".split())


def words_of(text: bytes) -> list[bytes]:
    return [u.strip(b" ").lower() for u in cv.scan(text) if u.strip(b" ").isalpha()]


FOLD_KEYS = False   # set by --fold-keys: inflections folded in the question key


def fold(w: bytes) -> bytes:
    """A deterministic stem for the key: common English inflections stripped when the stem stays
    at least four letters, so "describes", "described" and "describing" share one key word."""
    for suffix, repl in ((b"ies", b"y"), (b"ing", b""), (b"ed", b""), (b"es", b""), (b"ly", b""), (b"s", b"")):
        if w.endswith(suffix) and len(w) - len(suffix) >= 4:
            return w[:-len(suffix)] + repl
    return w


class V2LM(train_lm.LM):
    """The research core plus the two Design v2 sources that read the retrieved exchange.

    ``copy_head``: a learned pointer. Attention from the state at each position to the states
    of the retrieved span predicts the code that follows the attended position, so copying a
    passage is a walk along it; its confidence is the top weight and the entropy of the weights.
    ``evidence``: the local tables conditioned on the retrieved exchange, concretely a bigram
    table counted over the retrieved span only (code -> next code), with log(1 + count) as the
    confidence. Both enter the trust head as extra sources beside the network and the global
    tables; the mixture is per code, as before. ``src`` marks the retrieved span in the window.
    """

    def __init__(self, enc, width, layers, heads, ctx, ngrams, copy_head=False, evidence=False, sim_feature=False, block="legacy",
                 dropout=0.0):
        super().__init__(enc, width, layers, heads, ctx, 0, None, (2,), ngrams, block=block, dropout=dropout)
        self.copy_head, self.evidence, self.sim_feature = copy_head, evidence, sim_feature
        n_src = 1 + len(self.ngrams) + copy_head + evidence
        # sim_feature: the retrieval similarity between the asked question and the retrieved one
        # (0 to 1) is a trust-head input, so the head can learn how far to trust the copy.
        n_conf = 3 * len(self.ngrams) + 2 * copy_head + 1 * evidence + 1 * sim_feature
        self.mix_head = torch.nn.Linear(width + n_conf, n_src)
        torch.nn.init.normal_(self.mix_head.weight, std=0.02); torch.nn.init.zeros_(self.mix_head.bias)
        if copy_head:
            self.copy_q, self.copy_k = torch.nn.Linear(width, width, bias=False), torch.nn.Linear(width, width, bias=False)
            torch.nn.init.normal_(self.copy_q.weight, std=0.02); torch.nn.init.normal_(self.copy_k.weight, std=0.02)

    def _span(self, x, src):
        """Positions j whose follower j+1 is inside the retrieved span, and the follower codes."""
        follow = src & torch.cat([src[:, 1:], torch.zeros_like(src[:, :1])], 1)
        x_next = torch.cat([x[:, 1:], torch.full_like(x[:, :1], -1)], 1)
        return follow, x_next

    def extra_parts(self, h, x, src, y=None, sim=None):
        """Probability of the targets (or dense distributions when y is None) and confidence
        features from the copy head and the evidence table."""
        B, Tq, D = h.shape
        T = x.shape[1]
        follow, x_next = self._span(x, src)
        # Query t may point only at j < t: the follower of j = t is the very code being predicted,
        # so allowing it leaks the target (found with --copy-question, where the span can
        # contain the current position).
        causal = torch.ones(Tq, T, dtype=torch.bool, device=x.device).tril(T - Tq - 1)
        allowed = follow[:, None, :] & causal[None]
        parts, confs = [], []
        if self.copy_head:
            scores = (self.copy_q(h) @ self.copy_k(h if Tq == T else self._keys).transpose(1, 2)) / math.sqrt(D)
            scores = scores.masked_fill(~allowed, float("-inf"))
            attn = torch.softmax(scores, -1).nan_to_num(0.0)
            if y is not None:
                parts.append((attn * (x_next[:, None, :] == y[..., None])).sum(-1))
            else:
                dense = torch.zeros(B, Tq, self.vocab, device=h.device)
                dense.scatter_add_(-1, x_next.clamp_min(0)[:, None, :].expand(B, Tq, T), attn * allowed)
                parts.append(dense)
            top = attn.max(-1).values
            confs += [top, -(attn * torch.log2(attn.clamp_min(1e-9))).sum(-1)]
        if self.evidence:
            cur = x[:, -Tq:]
            same = (cur[:, :, None] == x[:, None, :]) & allowed
            cnt = same.sum(-1).float()
            if y is not None:
                hit = (same & (x_next[:, None, :] == y[..., None])).sum(-1).float()
                parts.append(hit / cnt.clamp_min(1.0))
            else:
                dense = torch.zeros(B, Tq, self.vocab, device=h.device)
                dense.scatter_add_(-1, x_next.clamp_min(0)[:, None, :].expand(B, Tq, T), same.float())
                parts.append(dense / cnt.clamp_min(1.0)[..., None])
            confs.append(torch.log1p(cnt))
        if self.sim_feature:
            s = sim if sim is not None else torch.zeros(B, device=h.device)
            confs.append(s.to(h.device, h.dtype)[:, None].expand(B, Tq))
        return parts, confs

    def nll(self, x, y, src=None, sim=None):
        if src is None:
            src = torch.zeros_like(x, dtype=torch.bool)
        h = self.hidden(x)
        logits = h @ self.emb.weight.T
        logp_net = F.log_softmax(logits, -1).gather(-1, y[..., None])[..., 0]
        p_parts, confs = [logp_net.exp()], []
        loo = 1.0 if self.training else 0.0
        for table in self.ngrams:
            ids, counts, totals = table.features(x)
            c_target = (counts * (ids == y[..., None])).sum(-1)
            kept = counts.sum(-1)
            c_target = (c_target - loo * (c_target > 0)).clamp_min(0.0)
            kept = (kept - loo * (totals > 0)).clamp_min(0.0)
            p_parts.append(torch.where(kept > 0, c_target / kept.clamp_min(1.0), torch.zeros_like(kept)))
            confs.extend(train_lm.table_features(counts, totals, loo, ids, y))
        parts, extra = self.extra_parts(h, x, src, y, sim)
        p_parts += parts; confs += extra
        mix = F.softmax(self.mix_head(torch.cat([h] + [c[..., None] for c in confs], -1)), -1)
        p = (mix * torch.stack(p_parts, -1)).sum(-1)
        return -torch.log(p.clamp_min(1e-9))

    def mixed_logits(self, h_last, x, src=None, sim=None):
        """Full next-code log-probabilities at the last position; ``h_last`` is (B, D) and the
        copy head needs every state of the window, so ``hidden_all`` must be called first."""
        logits = h_last @ self.emb.weight.T
        if not self.ngrams:                       # the "network alone" ablation, as in the base class
            return logits
        p, confs = [F.softmax(logits, -1)], []
        for table in self.ngrams:
            # only the last position is used: its context is the last table.k codes, so look up
            # just those (the whole window cost a 512 x 64 gather per table per generated code)
            ids, counts, totals = table.features(x[:, -(table.k + 1):], h_last.device)
            dense = torch.zeros_like(p[0])
            ok = ids[:, -1] >= 0
            probs = counts[:, -1] / counts[:, -1].sum(-1, keepdim=True).clamp_min(1.0)
            dense.scatter_add_(-1, ids[:, -1].clamp_min(0), probs * ok)
            p.append(dense)
            confs.extend(f[:, -1] for f in train_lm.table_features(counts, totals, 0.0))
        if src is None:
            src = torch.zeros_like(x, dtype=torch.bool)
        parts, extra = self.extra_parts(h_last[:, None, :], x, src, None, sim)
        p += [d[:, 0] for d in parts]; confs += [c[:, 0] for c in extra]
        mix = F.softmax(self.mix_head(torch.cat([h_last] + [c[..., None] for c in confs], -1)), -1)
        return torch.log((mix[..., None] * torch.stack(p, 1)).sum(1).clamp_min(1e-9))

    def hidden_all(self, x):
        """Hidden states of the whole window, kept for the copy head's keys; returns the last."""
        h = self.hidden(x)
        self._keys = h
        return h[:, -1]


class ExchangeMemory:
    """Key: content-word set of the question. Value: the exchange. Exact and nearest lookup."""

    def __init__(self):
        self.exchanges: list[tuple[bytes, bytes]] = []
        self.keys: list[frozenset] = []
        self.exact: dict[frozenset, int] = {}
        self.index: dict[bytes, list[int]] = collections.defaultdict(list)
        self.df: collections.Counter = collections.Counter()
        self.learned = None          # optional LearnedKeys: a second, learned key per question (K1)

    def attach_learned(self, learned: "LearnedKeys") -> None:
        """Index every stored question by the learned key as well; later adds are indexed too."""
        self.learned = learned
        learned.vectors = learned.embed([q for q, _ in self.exchanges])

    @staticmethod
    def key_of(question: bytes) -> frozenset:
        words = (w for w in words_of(question) if w.decode("latin-1") not in STOP and len(w) > 1)
        return frozenset(fold(w) for w in words) if FOLD_KEYS else frozenset(words)

    def add(self, question: bytes, answer: bytes) -> int:
        k = self.key_of(question)
        i = len(self.exchanges)
        self.exchanges.append((question, answer)); self.keys.append(k)
        self.exact.setdefault(k, i)
        for w in k:
            self.index[w].append(i); self.df[w] += 1
        if self.learned is not None:
            self.learned.vectors = torch.cat([self.learned.vectors, self.learned.embed([question])])
        return i

    def idf(self, w: bytes) -> float:
        return math.log((1 + len(self.exchanges)) / (1 + self.df[w]))

    def retrieve(self, question: bytes, k: int = 1, exclude: int | None = None, max_df_share: float = 0.05):
        key = self.key_of(question)
        n = max(1, len(self.exchanges))
        scores: dict[int, float] = collections.defaultdict(float)
        for w in key:
            if self.df[w] > max_df_share * n:
                continue
            idf = self.idf(w)
            for i in self.index[w]:
                if i != exclude:
                    scores[i] += idf
        denom = {w: self.idf(w) for w in key}
        lit = {}
        for i, s in scores.items():
            union = sum(denom.values()) + sum(self.idf(w) for w in self.keys[i] if w not in key)
            lit[i] = s / max(union, 1e-9)
        if self.learned is not None and (exclude is None or getattr(self.learned, "for_training", False)):
            # hybrid: rank by literal + learned similarity (weights fitted on held-out rewordings in K1);
            # the returned score stays the literal similarity, so group definitions do not change
            L = self.learned
            cos = (L.vectors @ L.embed([question])[0]).cpu()
            if exclude is not None:
                cos[exclude] = -2.0
            cand = set(sorted(lit, key=lit.get, reverse=True)[:50]) | set(torch.topk(cos, min(50, len(cos))).indices.tolist())
            ranked = sorted(cand, key=lambda i: lit.get(i, 0.0) / L.lit_std + L.weight * float(cos[i]) / L.key_std, reverse=True)
        else:
            if not lit:
                return []
            ranked = sorted(lit, key=lambda i: (lit[i], -i), reverse=True)
        rr = getattr(self, "reranker", None)
        if rr is not None and exclude is None and len(ranked) > 1:
            # reranking by the core itself (HYBRID_DESIGN 3.3): the top candidates reordered by how well the
            # model predicts this question with each candidate exchange in its window; the score returned
            # stays the literal similarity, so the gate and the evaluation groups keep their meaning
            n = rr.top
            ranked = rr(question, ranked[:n]) + ranked[n:]
        return [(lit.get(i, 0.0), i) for i in ranked[:k]]

    def exact_hit(self, question: bytes):
        return self.exact.get(self.key_of(question))

    def similarity_keys(self, key: frozenset, other: frozenset, max_df_share: float = 0.05) -> float:
        """The retrieval score of ``other`` for a question with ``key``, by the same formula as
        retrieve(): idf-weighted shared words (common words excluded) over the weighted union."""
        if not key:
            return 0.0
        n = max(1, len(self.exchanges))
        shared = sum(self.idf(w) for w in key & other if self.df[w] <= max_df_share * n)
        union = sum(self.idf(w) for w in key) + sum(self.idf(w) for w in other - key)
        return 1.0 if key == other else shared / max(union, 1e-9)


def build_synonyms(questions: list[bytes], device: str, work: Path, min_count: int = 5, threshold: float = 0.76,
                   per_word: int = 6) -> dict[bytes, list[bytes]]:
    """Meaning-equivalent substitutions for question words, from the questions' own vocabulary: every
    content word and content bigram seen at least min_count times is embedded by the dense key, and a
    word's substitutes are the phrases at cosine >= threshold that neither contain it nor are contained
    in it (food -> meal, cooking -> culinary, famous -> renowned). Cached in the work folder."""
    import hashlib as _h
    key = _h.sha256(b"\0".join(questions[:2000]) + repr((len(questions), min_count, threshold, per_word)).encode()).hexdigest()[:16]
    path = work / f"synonyms_{key}.json"
    if path.exists():
        return {k.encode("latin-1"): [v.encode("latin-1") for v in vs] for k, vs in json.loads(path.read_text()).items()}
    import dense_keys
    cnt = collections.Counter()
    for q in questions:
        w = re.findall(rb"[a-z][a-z0-9'-]+", q.lower())
        for i, x in enumerate(w):
            if x.decode("latin-1") in STOP:
                continue
            if len(x) > 2:
                cnt[x] += 1
            if i + 1 < len(w) and w[i + 1].decode("latin-1") not in STOP:
                cnt[x + b" " + w[i + 1]] += 1
    vocab = [p for p, c in cnt.items() if c >= min_count]
    E = dense_keys.DenseKeys(device).embed(vocab, batch=1024)
    words = [i for i, p in enumerate(vocab) if b" " not in p]
    syn: dict[bytes, list[bytes]] = {}
    for s in range(0, len(words), 1024):
        rows = words[s:s + 1024]
        sims = E[rows] @ E.T
        top = torch.topk(sims, per_word * 3, dim=1)
        for r, (vals, ids) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
            w = vocab[rows[r]]
            # not containing / contained (food items), and not the same stem (virus / viral, cook / cooked)
            out = [vocab[j] for v, j in zip(vals, ids) if v >= threshold and w not in vocab[j] and vocab[j] not in w
                   and vocab[j][:4] != w[:4]][:per_word]
            if out:
                syn[w] = out
    path.write_text(json.dumps({k.decode("latin-1"): [v.decode("latin-1") for v in vs] for k, vs in syn.items()}))
    return syn


def add_typos(q: bytes, rng, rate: float = 0.15) -> bytes:
    """A question as people type it: each word longer than three letters gets, with probability `rate`, one
    swap of two neighbouring letters, a dropped letter or a doubled letter; one question in five is lower-cased,
    and one in three loses its final question mark. Only noise that real users make (no reordering, no new words)."""
    out = []
    for w in q.split(b" "):
        if len(w) > 3 and rng.random() < rate:
            k = int(rng.integers(1, len(w) - 1))
            r = rng.random()
            if r < 1 / 3:
                w = w[:k] + w[k + 1:k + 2] + w[k:k + 1] + w[k + 2:]
            elif r < 2 / 3:
                w = w[:k] + w[k + 1:]
            else:
                w = w[:k] + w[k:k + 1] + w[k:]
        out.append(w)
    t = b" ".join(out)
    if rng.random() < 0.2:
        t = t.lower()
    if t.endswith(b"?") and rng.random() < 1 / 3:
        t = t[:-1]
    return t


def perturb_question(q: bytes, rng, donor: list[bytes], syn: dict | None = None) -> bytes:
    """A reworded stand-in for a question, for training the copy head on near matches: each word
    is dropped (20%) or replaced by a word from another question (10%), then neighbouring words
    are swapped (15% per pair). With ``syn`` (build_synonyms), a word that has meaning-equivalent
    substitutes is replaced by one of them 40% of the time instead (capitalised words inside the question,
    names, never are), and the drop, donor and swap rates fall to 5%, 2% and 5%, so the rewording reads
    like a paraphrase. Deterministic given the generator."""
    words, out = q.split(b" "), []
    for n, w in enumerate(words):
        r = rng.random()
        if syn is not None:
            core = w.lower().strip(b".,?!:;\"'()")
            subs = None if (n > 0 and w[:1].isupper()) else syn.get(core)   # names keep their meaning: never substituted
            if subs and r < 0.4:
                out.append(w.lower().replace(core, subs[int(rng.integers(len(subs)))], 1))
                continue
            if r > 0.95:
                continue
            out.append(donor[int(rng.integers(len(donor)))] if (r > 0.93 and donor) else w)
            continue
        if r < 0.2:
            continue
        out.append(donor[int(rng.integers(len(donor)))] if (r < 0.3 and donor) else w)
    for k in range(len(out) - 1):
        if rng.random() < (0.05 if syn is not None else 0.15):
            out[k], out[k + 1] = out[k + 1], out[k]
    return b" ".join(out) if out else words[0]


class LearnedKeys:
    """K1 key: the frozen stage-1 core's mean hidden state over the question, through the trained
    key head, as a unit vector (archive/experiments_oct2026/scripts/k1_question_key.py; superseded by dense_keys.py)."""

    def __init__(self, head_path: str, native: str | None, device: str = "cpu"):
        import core_eval
        ck = torch.load(head_path, map_location="cpu", weights_only=False)
        self.enc, self.core = core_eval.load_core(ck["core"], native, device)
        D = ck["mu"].shape[1]
        self.head = torch.nn.Sequential(torch.nn.Linear(D, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)).to(device)
        self.head.load_state_dict(ck["head"]); self.head.eval()
        self.mu, self.sd, self.dev = ck["mu"].to(device), ck["sd"].to(device), device
        self.weight, self.lit_std, self.key_std = ck["weight"], ck["lit_std"], ck["key_std"]
        self.vectors = torch.zeros(0, 128, device=device)

    @torch.no_grad()
    def embed(self, questions: list[bytes], batch: int = 256) -> torch.Tensor:
        out = []
        for i in range(0, len(questions), batch):
            codes = [self.enc.encode(b"Question: " + q)[0][: self.core.ctx] for q in questions[i:i + batch]]
            L = max(len(c) for c in codes)
            x = np.zeros((len(codes), L), dtype=np.int64); m = np.zeros((len(codes), L), dtype=np.float32)
            for r, c in enumerate(codes):
                x[r, :len(c)] = c; m[r, :len(c)] = 1
            h = self.core.hidden(torch.from_numpy(x).to(self.dev)).float()
            mt = torch.from_numpy(m).to(self.dev)[..., None]
            v = (h * mt).sum(1) / mt.sum(1).clamp_min(1)
            out.append(F.normalize(self.head((v - self.mu) / self.sd), dim=-1))
        return torch.cat(out) if out else torch.zeros(0, 128, device=self.dev)


class CoreReranker:
    """Rerank retrieval candidates with the Q&A model itself (HYBRID_DESIGN 3.3, research item 15): for each
    candidate exchange, the window [candidate tail, SEP, "User: question\\nAssistant:"] goes through the model
    and the bits it needs for the question codes are summed; fewer bits = the candidate explains the question
    better. One batched forward pass per question, no parameters of its own."""

    def __init__(self, model, enc, codes_of, sep, neighbor_codes: int, ctx: int, device, top: int = 10):
        self.model, self.enc, self.codes_of, self.sep = model, enc, codes_of, np.asarray(sep, dtype=np.int64)
        self.nb, self.ctx, self.dev, self.top = neighbor_codes, ctx, device, top

    @torch.no_grad()
    def __call__(self, question: bytes, cand: list[int]) -> list[int]:
        qc = np.asarray(self.enc.encode(render(question, None))[0], dtype=np.int64)
        rows, spans = [], []
        for c in cand:
            nb = np.asarray(self.codes_of(c), dtype=np.int64)[-self.nb:]
            seq = np.concatenate([nb, self.sep, qc])[: self.ctx + 1]
            rows.append(seq); spans.append((len(nb), len(nb) + len(self.sep), len(seq)))
        L = max(len(r) for r in rows)
        x = np.zeros((len(rows), L), dtype=np.int64); src = np.zeros((len(rows), L), dtype=bool)
        for r, (seq, (nl, q0, end)) in enumerate(zip(rows, spans)):
            x[r, :len(seq)] = seq; src[r, :nl] = True
        xt = torch.from_numpy(x).to(self.dev)
        was = self.model.training; self.model.eval()
        nll = self.model.nll(xt[:, :-1], xt[:, 1:], torch.from_numpy(src[:, :-1]).to(self.dev),
                             torch.zeros(len(rows), device=self.dev))
        self.model.train(was)
        score = [float(nll[r, q0 - 1:end - 1].sum()) for r, (nl, q0, end) in enumerate(spans)]
        return [c for _, c in sorted(zip(score, cand), key=lambda t: t[0])]


def parse_exchanges(text: bytes) -> list[tuple[bytes, bytes]]:
    """Exchanges are separated by a blank line followed by a new "User: " turn. Splitting at every
    blank line (as before 10 October 2026) cut the 6.6% of Alpaca answers that have paragraphs
    (letters, stories, code) to their first paragraph and dropped the rest."""
    out = []
    for block in re.split(rb"\n\n(?=User: )", text):
        m = re.match(rb"User: (.*?)\nAssistant: (.*)\Z", block, re.S)
        if m:
            out.append((m.group(1).strip(), m.group(2).strip()))
    return out


def render(q: bytes, a: bytes | None) -> bytes:
    return b"User: " + q + b"\nAssistant:" + (b" " + a + b"\n\n" if a is not None else b"")


NUM = re.compile(rb"Answer:\s*(-?[\d,]*\.?\d+)")


def final_number(text: bytes) -> bytes | None:
    """The checkable target of a GSM-style answer: the number after the last "Answer:"."""
    m = NUM.findall(text)
    return m[-1].replace(b",", b"") if m else None


EXPR = re.compile(rb"([\d][\d,]*\.?\d*(?:\s*[-+*/x]\s*[\d$][\d,]*\.?\d*)+)\s*=\s*$")


def evaluate_tail(text: bytes) -> str | None:
    """The executor. When ``text`` ends in an arithmetic expression followed by '=', its value as
    the model should write it (integer when integral, else up to two decimals); else None."""
    m = EXPR.search(text[-80:])
    if not m:
        return None
    expr = m.group(1).replace(b",", b"").replace(b"$", b"").replace(b"x", b"*").replace(b"\xc3\x97", b"*").decode("ascii", "ignore")
    try:
        import ast
        node = ast.parse(expr, mode="eval")
        def ev(n):
            if isinstance(n, ast.Expression): return ev(n.body)
            if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return float(n.value)
            if isinstance(n, ast.BinOp) and type(n.op) in (ast.Add, ast.Sub, ast.Mult, ast.Div):
                l, r = ev(n.left), ev(n.right)
                return {ast.Add: l + r, ast.Sub: l - r, ast.Mult: l * r, ast.Div: l / r if r else float("nan")}[type(n.op)]
            raise ValueError
        v = ev(node)
    except (ValueError, SyntaxError, ZeroDivisionError, RecursionError):
        return None
    if v != v or abs(v) > 1e15:
        return None
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:.2f}".rstrip("0").rstrip(".")


def num_exact(pred: bytes, ref: bytes) -> float | None:
    """1/0 when the reference carries a final number (GSM8K), None otherwise."""
    r = final_number(ref)
    return None if r is None else float(final_number(pred) == r)


def f1(pred: bytes, ref: bytes) -> float:
    p, r = collections.Counter(words_of(pred)), collections.Counter(words_of(ref))
    common = sum((p & r).values())
    if common == 0:
        return 0.0
    prec, rec = common / max(1, sum(p.values())), common / max(1, sum(r.values()))
    return 2 * prec * rec / (prec + rec)


EOS = 12   # end-of-answer code (the form-feed byte, also the document separator in the core mixes)


def repeat_banned(gen: list[int], n: int) -> list[int]:
    """Codes that would complete an n-gram already present in ``gen`` (the reply so far): the
    standard block against greedy loops ("It was a major part of ... It was a major part of")."""
    if n < 2 or len(gen) < n - 1:
        return []
    prefix = tuple(gen[len(gen) - n + 1:])
    return [gen[i + n - 1] for i in range(len(gen) - n + 1) if tuple(gen[i:i + n - 1]) == prefix]


class QAEngine:
    """A trained Q&A model ready to answer: the compiler, the exchange memory (M2) and count
    tables (M1) rebuilt from the model's training text, and the saved core with its copy head.
    Used by run_model.py's chat; the scoring in main() keeps its own loop so earlier results stay
    reproducible.

        eng = QAEngine("runs/r1b/alpaca_5M_init_s2.pt", native="data/cv0.exe")
        reply, info = eng.ask(b"Name 3 characters in the movie Frozen.")
        eng.add(b"Who leads the research?", b"Adam Barnett leads the research.")
    """

    def __init__(self, checkpoint: str, native: str | None = None, device: str = "cpu", data: str | None = None, log=print,
                 precision: str = "fp32", kernels: str = "fp32", dense: float = 0.0, rerank: int = 0):
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.a = a = argparse.Namespace(**ck["args"])
        global FOLD_KEYS
        FOLD_KEYS = getattr(a, "fold_keys", False)
        root = Path(__file__).resolve().parents[2]
        resolve = lambda p: p if p is None or Path(p).is_absolute() or Path(p).exists() else str(root / p)
        self.data = resolve(data or a.data)
        work = Path(resolve(a.work))
        work.mkdir(parents=True, exist_ok=True)
        dict_path = resolve(a.dict) if getattr(a, "dict", None) else str(work / "v0_8k_plain.cv0d")
        if not Path(dict_path).exists():
            # runs/ is not in git: a saved model keeps its dictionary beside it (same name, .cv0d); the
            # first saved models used models/v0_8k_plain_corpus_v2.cv0d
            beside = Path(checkpoint).with_suffix(".cv0d")
            kept = beside if beside.exists() else root / "models" / "v0_8k_plain_corpus_v2.cv0d"
            if kept.exists():
                log(f"dictionary {dict_path} not found; using {kept}")
                dict_path = str(kept)
        self.enc = enc = LE.V0Dict(dict_path, native)
        train_file = f"{self.data}/train/qa_train.txt"
        train_x = parse_exchanges(Path(train_file).read_bytes())
        if getattr(a, "max_train", 0):
            train_x = train_x[:a.max_train]
        self.mem = ExchangeMemory()
        for q, ans in train_x:
            self.mem.add(q, ans)
        if getattr(a, "key_head", None) and Path(resolve(a.key_head)).exists():
            self.mem.attach_learned(LearnedKeys(resolve(a.key_head), native, device if torch.cuda.is_available() or device == "cpu" else "cpu"))
        dense = dense or getattr(a, "dense", 0.0)
        if dense:                      # pretrained sentence encoder as the second key (dense_keys.py)
            import dense_keys
            self.mem.attach_learned(dense_keys.DenseKeys(device, weight=dense))
            log(f"dense question keys: {dense_keys.MODEL}, weight {dense}")
        t0 = time.time()
        # the file is the rendered exchanges back to back, so encoding it whole gives the same codes
        all_codes = enc.encode_file(train_file, work)[0] if native else np.concatenate([enc.encode(render(q, x))[0] for q, x in train_x])
        self.ngrams = [train_lm.NgramTable(all_codes, o, a.topk) for o in (2, 3)]
        self.never_seen = torch.from_numpy(np.bincount(all_codes, minlength=enc.vocab) == 0)
        if getattr(a, "eos", False):
            self.never_seen[EOS] = False
        self.dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.model = V2LM(enc, a.width, a.layers, a.heads, a.ctx, self.ngrams, a.copy_head, getattr(a, "evidence", False),
                          getattr(a, "sim_feature", False), getattr(a, "block", "legacy")).to(self.dev)
        self.model.load_state_dict(ck["state_dict"], strict=False)
        if precision != "fp32":
            import precision as _precision
            log(f"weights rounded: {_precision.apply(self.model, precision)}")
        self.kernels = self.model._kernels = kernels
        if kernels != "fp32":
            import precision as _precision
            log(f"answer kernels: {_precision.to_kernels(self.model, kernels)}")
        self.model.eval()
        self.sep = enc.encode(b"\n\n")[0]
        rerank = rerank or getattr(a, "rerank_core", 0)
        if rerank:
            self.mem.reranker = CoreReranker(self.model, enc, lambda i: enc.encode(render(*self.mem.exchanges[i]))[0],
                                             self.sep, a.neighbor_codes, a.ctx, self.dev, rerank)
            log(f"retrieval reranked by the core: top {rerank}")
        self.equals = int(enc.encode(b"=")[0][-1])
        self.newline = int(enc.encode(bytes([10]))[0][-1])
        log(f"loaded {Path(checkpoint).name}: {self.model.param_counts()['total']:,} learned parameters, "
            f"{len(train_x):,} exchanges in memory, tables rebuilt in {time.time() - t0:.0f}s")

    def add(self, question: bytes, answer: bytes) -> int:
        """Put a new pair into memory: answerable at once, no training."""
        return self.mem.add(question.strip(), answer.strip())

    def retrieve(self, q: bytes, exclude: int | None = None):
        """(exchange index, similarity, how): exact key first, else the nearest key. With ``exclude``,
        that stored exchange is hidden (verified self-training must solve, not copy)."""
        if exclude is not None:
            r = self.mem.retrieve(q, 1, exclude=exclude)
            return (r[0][1], r[0][0], "nearest") if r else (None, 0.0, "none")
        hit = self.mem.exact_hit(q)
        if hit is not None:
            return hit, 1.0, "exact"
        r = self.mem.retrieve(q, 1)
        return (r[0][1], r[0][0], "nearest") if r else (None, 0.0, "none")

    def ask(self, q: bytes, *args, **kw):
        """See _ask. With kernels="bf16" the whole answer runs under bfloat16 autocast."""
        if getattr(self, "kernels", "fp32") == "bf16" and self.dev.type == "cuda":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self._ask(q, *args, **kw)
        return self._ask(q, *args, **kw)

    @torch.no_grad()
    def _ask(self, q: bytes, max_codes: int = 120, no_repeat: int = 3, calculator: bool | None = None, use_memory: bool = True,
            temperature: float = 0.0, top_k: int = 20, gen: torch.Generator | None = None, exclude: int | None = None,
            min_sim: float = 0.0, fast: bool | str = True):
        """Answer through the compiler: retrieve, place the exchange in the window, generate greedily
        with the copy head, decode. Returns (reply bytes, info dict)."""
        a, enc, model = self.a, self.enc, self.model
        calculator = getattr(a, "calculator", False) if calculator is None else calculator
        idx, sim, how = self.retrieve(q, exclude) if use_memory else (None, 0.0, "off")
        if idx is not None and how == "nearest" and sim < min_sim:   # a weak match: answer from the network alone
            idx, sim, how = None, 0.0, f"skipped (similarity {sim:.2f} < {min_sim})"
        if idx is not None and a.neighbor_codes:
            q_codes = enc.encode(render(*self.mem.exchanges[idx]))[0]
            nb = q_codes[-a.neighbor_codes:]
        else:
            nb = np.array([], dtype=np.int64)
        seq = [int(t) for t in np.concatenate([nb, self.sep, enc.encode(render(q, None))[0]])]
        start = len(seq)
        n = min(len(seq), a.ctx)
        computed, after_value = [], False
        cache, keys, done = None, None, 0       # KV cache: states of codes already processed (fast=True)
        use_cache = fast
        dev = self.dev
        # Everything the loop needs lives on the device once: the window grows by one code per step.
        # (Rebuilding the window from the Python list each step cost 10x the network at 50M.)
        seq_dev = torch.tensor(seq, device=dev)
        sim_t = torch.tensor([sim], device=dev)
        if getattr(self, "_never_dev", None) is None or self._never_dev.device != seq_dev.device:
            self._never_dev = self.never_seen.to(dev)
            self._digit_dev = torch.tensor([c for c in range(enc.vocab) if enc.d.id_bytes[c][:1].isdigit()], device=dev)
        graph = None
        if fast == "graph" and len(seq) + max_codes + 8 <= a.ctx:
            import fast_decode
            if fast_decode.supported(model):
                if getattr(self, "_graph", None) is None:
                    self._graph = fast_decode.GraphStep(model)
                graph = self._graph
                pos0 = torch.arange(len(seq), device=dev)
                span0 = pos0 < len(nb)
                if getattr(a, "copy_question", False):
                    span0 |= (pos0 >= len(nb) + len(self.sep)) & (pos0 < start)
                pending = graph.prefill(seq_dev, span0, sim)
        while graph is not None and len(seq) - start < max_codes:
            logits = pending.masked_fill(self._never_dev, float("-inf"))
            banned = repeat_banned(seq[start:], no_repeat)
            if banned:
                logits[0, banned] = float("-inf")
            if after_value:
                logits[0, self._digit_dev] = float("-inf")
                after_value = False
            if temperature > 0:
                lg = logits[0] / temperature
                kth = torch.topk(lg, top_k).values[-1]
                lg = lg.masked_fill(lg < kth, float("-inf"))
                nxt = int(torch.multinomial(F.softmax(lg, -1).cpu(), 1, generator=gen).item())
            else:
                nxt = int(logits.argmax(-1).item())
            if getattr(a, "eos", False) and nxt == EOS:
                break
            seq.append(nxt)
            added = [nxt]
            if calculator and nxt == self.equals:
                value = evaluate_tail(enc.decode(seq[start:]))
                if value is not None:
                    computed.append(value)
                    vc = [int(t) for t in enc.encode(b" " + value.encode())[0]]
                    seq.extend(vc); added.extend(vc)
                    after_value = True
            for j, c in enumerate(added):
                pending = graph.step(c, len(seq) - len(added) + j)
            text = enc.decode(seq[max(start, len(seq) - len(added) - 16):])
            if b"\nUser" in text or b"\nAssistant:" in text:
                break
        if graph is not None:
            out = enc.decode(seq[start:]).split(b"\nUser")[0].split(b"\nAssistant:")[0].strip()
            info = {"memory": how, "similarity": round(sim, 3), "codes": len(seq) - start, "computed": computed}
            if idx is not None:
                info["retrieved_question"] = self.mem.exchanges[idx][0][:120].decode("utf-8", "replace")
            return out, info
        while len(seq) - start < max_codes:
            x = seq_dev[-a.ctx:][None]
            off = max(0, len(seq) - a.ctx)
            pos = torch.arange(min(len(seq), a.ctx), device=dev) + off
            span = pos < len(nb)
            if getattr(a, "copy_question", False):
                span |= (pos >= len(nb) + len(self.sep)) & (pos < start)
            if use_cache and len(seq) <= a.ctx:
                # only the codes added since the last step go through the network
                h_new, cache = model.hidden_cached(seq_dev[done:][None], done, cache)
                keys = h_new if keys is None else torch.cat([keys, h_new], 1)
                done = len(seq)
                model._keys = keys
                h_last = keys[:, -1]
            else:                               # past the window the positions shift: recompute it whole
                use_cache = False
                h_last = model.hidden_all(x)
            logits = model.mixed_logits(h_last, x, span[None], sim_t)
            logits = logits.masked_fill(self._never_dev, float("-inf"))
            banned = repeat_banned(seq[start:], no_repeat)
            if banned:
                logits[0, banned] = float("-inf")
            if after_value:                  # no digit may extend a value the calculator just wrote
                logits[0, self._digit_dev] = float("-inf")
                after_value = False
            if temperature > 0:              # sampled answers (verified self-training needs several different tries)
                lg = logits[0] / temperature
                kth = torch.topk(lg, top_k).values[-1]
                lg = lg.masked_fill(lg < kth, float("-inf"))
                nxt = int(torch.multinomial(F.softmax(lg, -1).cpu(), 1, generator=gen).item())
            else:
                nxt = int(logits.argmax(-1).item())
            if getattr(a, "eos", False) and nxt == EOS:
                break
            seq.append(nxt)
            added = [nxt]
            if calculator and nxt == self.equals:
                value = evaluate_tail(enc.decode(seq[start:]))
                if value is not None:
                    computed.append(value)
                    vc = [int(t) for t in enc.encode(b" " + value.encode())[0]]
                    seq.extend(vc); added.extend(vc)
                    after_value = True
            seq_dev = torch.cat([seq_dev, torch.tensor(added, device=dev)])
            # a stop marker can only complete inside the codes just added; every code is at least one
            # byte, so the last len(added) + 16 codes cover the 11-byte marker (no full decode per step)
            text = enc.decode(seq[max(start, len(seq) - len(added) - 16):])
            if b"\nUser" in text or b"\nAssistant:" in text:   # the next turn, or an invented one
                break
        out = enc.decode(seq[start:]).split(b"\nUser")[0].split(b"\nAssistant:")[0].strip()
        info = {"memory": how, "similarity": round(sim, 3), "codes": len(seq) - start, "computed": computed}
        if idx is not None:
            info["retrieved_question"] = self.mem.exchanges[idx][0][:120].decode("utf-8", "replace")
        return out, info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("work"); ap.add_argument("--native"); ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--ctx", type=int, default=256); ap.add_argument("--batch", type=int, default=16); ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--neighbor-codes", type=int, default=110); ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--max-train", type=int, default=0); ap.add_argument("--out", default="results/m2_qa.json"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--self-context-p", type=float, default=0.0, help="share of training windows whose context is the exchange itself, to teach copying from memory")
    ap.add_argument("--copy-bias", type=float, default=0.0, help="generation-time bonus (in nats) on codes that appear in the retrieved answer: the soft form of answering from evidence")
    ap.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N"); ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--block", choices=["legacy", "modern"], default="legacy", help="must match the stage-1 core given to --init")
    ap.add_argument("--precision", choices=["fp32", "bf16", "int8", "int4"], default="fp32", help="round the weights to this format after --init (with --budget 0: the accuracy cost of a quantised deployment)")
    ap.add_argument("--greedy", action="store_true", help="scored answers take the most likely code at each step instead of sampling")
    ap.add_argument("--no-repeat", type=int, default=0, help="block any n-gram of this many codes from repeating within a reply (0: off; 3 is standard)")
    ap.add_argument("--patience", type=int, default=0, help="stop when validation bits/byte has not improved by --min-delta for this many evaluations (0: never); the best weights are kept")
    ap.add_argument("--min-delta", type=float, default=0.005)
    ap.add_argument("--max-steps", type=int, default=0, help="stop after this many training steps (0: budget only); lets cores of different speed get equal training")
    ap.add_argument("--dict", default=None, help="use this dictionary (.cv0d) instead of fitting one on the Q&A text, e.g. the scale run's")
    ap.add_argument("--init", default=None, help="start the core from this checkpoint (train_lm or m2_qa); heads whose shape differs start fresh. With --budget 0 the run only evaluates")
    ap.add_argument("--copy-head", action="store_true", help="learned pointer over the retrieved exchange as a trust-head source")
    ap.add_argument("--evidence", action="store_true", help="bigram table counted over the retrieved exchange as a trust-head source")
    ap.add_argument("--fold-keys", action="store_true", help="fold inflections in the question key")
    ap.add_argument("--copy-answer-only", action="store_true", help="the copy head and evidence table see only the answer part of the retrieved exchange")
    ap.add_argument("--copy-question", action="store_true", help="the copy head may also point into the current question (operands, names)")
    ap.add_argument("--calculator", action="store_true", help="check loop, rule 1: when the reply has written 'a op b =', the executor writes the result; the final Answer: is checked against the last computed value")
    ap.add_argument("--perturb-self", type=float, default=0.0, help="share of self-context windows whose retrieved question is reworded (words dropped, swapped, replaced): teaches copying from near matches")
    ap.add_argument("--sim-feature", action="store_true", help="the retrieval similarity is a trust-head input")
    ap.add_argument("--no-retrieval-p", type=float, default=0.0, help="share of non-self windows trained with nothing retrieved")
    ap.add_argument("--distractor-p", type=float, default=0.0, help="share of the remaining non-self windows whose retrieved exchange is replaced by a random wrong one")
    ap.add_argument("--loss-answer-only", action="store_true", help="train and validate on the answer codes only, not the question")
    ap.add_argument("--stop-at-blank", action="store_true", help="a reply ends at its first blank line (single-span answers, e.g. SQuAD)")
    ap.add_argument("--lr", type=float, default=1e-3, help="AdamW peak learning rate")
    ap.add_argument("--key-head", default=None, help="K1 key head (runs/<core>/key_head.pt): retrieve by literal + learned question key")
    ap.add_argument("--amp", action="store_true", help="train in bfloat16 autocast (scores stay in full precision)")
    ap.add_argument("--dropout", type=float, default=0.0, help="residual dropout in every block (the Q&A stage reads its data ~10 times; 0: off)")
    ap.add_argument("--freeze-core-steps", type=int, default=0, help="train only the new heads for this many steps, then the core joins at --lr (two-phase fine-tuning)")
    ap.add_argument("--typo-p", type=float, default=0.0, help="share of windows whose question carries realistic typing mistakes (add_typos)")
    ap.add_argument("--track-gap", action="store_true", help="also score 120 training exchanges as if held out (own entry hidden) at every evaluation: the overfitting gap")
    ap.add_argument("--accum", type=int, default=1, help="split each batch into this many slices with summed gradients (same update, less GPU memory)")
    ap.add_argument("--perturb-synonyms", action="store_true", help="with --perturb-self: reword with meaning-equivalent substitutes from the dense key (food -> meal) instead of mostly random edits")
    ap.add_argument("--rerank-core", type=int, default=0, help="rerank the top N retrieval candidates by the core's own surprise at the question (0: off; 10 is the design's value)")
    ap.add_argument("--dense-train", action="store_true", help="with --dense: training neighbours are retrieved densely as well")
    ap.add_argument("--dense", type=float, default=0.0, help="weight of a pretrained sentence encoder's cosine in retrieval (dense_keys.py; 0: off, 2: best on QQP rewordings)")
    ap.add_argument("--eos", action="store_true", help="train an end-of-answer code after every exchange (inside the loss) and stop generation at it")
    ap.add_argument("--head-lr", type=float, default=0.0, help="learning rate for the new heads (copy head, trust head); 0: same as --lr")
    ap.add_argument("--schedule", choices=["constant", "cosine"], default="constant", help="cosine: 100-step warm-up, decay to 10%% of the peak")
    ap.add_argument("--net-n", type=int, default=60, help="questions per network-path scoring group (60 gives about +-0.07 F1 noise; 200 about +-0.04)")
    ap.add_argument("--rehearse", default=None, help="a stage-1 data folder (train/*.txt) whose text is mixed into stage-2 batches")
    ap.add_argument("--rehearse-p", type=float, default=0.3, help="share of stage-2 windows taken from the rehearsal text")
    ap.add_argument("--save", default=None, help="save the core's weights here after training")
    a = ap.parse_args()
    global FOLD_KEYS
    FOLD_KEYS = a.fold_keys
    if a.threads:
        torch.set_num_threads(a.threads)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)

    train_x = parse_exchanges(Path(f"{a.data}/train/qa_train.txt").read_bytes())
    held_x = parse_exchanges(Path(f"{a.data}/heldout/qa_heldout.txt").read_bytes())
    if a.max_train:
        train_x = train_x[:a.max_train]
    # Held-out: half for evaluation (unseen/paraphrase), half to be added to memory after training.
    rng.shuffle(held_x)
    eval_x, add_x = held_x[:len(held_x) // 2], held_x[len(held_x) // 2:]
    say(f"{len(train_x):,} training exchanges, {len(eval_x)} for evaluation, {len(add_x)} to add after training")

    enc = LE.V0Dict(a.dict or str(ladder1.fit_dictionary("v0_8k_plain", [f"{a.data}/train/qa_train.txt"], work)), a.native)
    mem = ExchangeMemory()
    for q, ans in train_x:
        mem.add(q, ans)
    t0 = time.time()
    # The nearest-neighbour pass is deterministic in the training questions and the key settings,
    # and costs minutes on Alpaca, so it is cached beside the work files.
    import hashlib
    SYN = None
    if a.perturb_synonyms:          # reworded self windows use meaning-equivalent substitutions (build_synonyms)
        SYN = build_synonyms([q for q, _ in train_x], a.device, work)
        say(f"synonym table: {len(SYN):,} words with substitutes, e.g. food -> {SYN.get(b'food', [])[:3]}")
    if a.dense and a.dense_train:   # training windows also use dense retrieval, so the network learns to copy from it
        import dense_keys
        mem.attach_learned(dense_keys.DenseKeys(a.device, weight=a.dense))
        mem.learned.for_training = True
        say(f"dense keys for training neighbours too: {len(mem.learned.vectors):,} questions embedded")
    nb_key = hashlib.sha256(b"\0".join(q for q, _ in train_x) + repr((FOLD_KEYS, sorted(STOP))).encode()
                            + (repr(("dense", a.dense)).encode() if a.dense and a.dense_train else b"")).hexdigest()[:16]
    nb_cache = work / f"neighbors_{nb_key}.json"
    if not a.neighbor_codes:             # no retrieved exchange in the window: the pass is unused
        neighbors = [None] * len(train_x)
    elif nb_cache.exists():
        neighbors = json.loads(nb_cache.read_text())
    else:
        neighbors = [r[0][1] if (r := mem.retrieve(q, 1, exclude=i)) else None for i, (q, _) in enumerate(train_x)]
        nb_cache.write_text(json.dumps(neighbors))
    if a.key_head:     # K1: retrieval at answer time ranks by literal + learned key (training windows keep the literal neighbours)
        mem.attach_learned(LearnedKeys(a.key_head, a.native, a.device))
    if a.dense and not a.dense_train:   # dense retrieval at answer time only; training windows keep the literal neighbours
        import dense_keys
        mem.attach_learned(dense_keys.DenseKeys(a.device, weight=a.dense))
        say(f"learned question key attached: {len(mem.learned.vectors):,} questions embedded")
    say(f"M2 built: {len(mem.exact):,} distinct keys; nearest-neighbour pass {time.time() - t0:.0f}s; "
        f"{sum(n is None for n in neighbors)} exchanges without a neighbour")

    codes = [enc.encode(render(q, ans))[0] for q, ans in train_x]
    all_codes = np.concatenate(codes)

    def codes_of(i: int) -> np.ndarray:
        """Codes of exchange i in memory; exchanges added after training are encoded on demand."""
        while len(codes) <= i:
            q, ans = mem.exchanges[len(codes)]
            codes.append(enc.encode(render(q, ans))[0])
        return codes[i]
    q_lens: list[int] = []

    def q_len(i: int) -> int:
        """Codes of exchange i's question part ("User: ...\\nAssistant:")."""
        while len(q_lens) <= i:
            q_lens.append(len(enc.encode(render(mem.exchanges[len(q_lens)][0], None))[0]))
        return q_lens[i]

    def answer_start(i: int, nb_len: int) -> int:
        """Index inside the retrieved tail (the last nb_len codes of exchange i) where its answer
        begins; 0 when the whole tail may be copied from."""
        if not a.copy_answer_only:
            return 0
        return max(0, q_len(i) - (len(codes_of(i)) - nb_len))

    ngrams = [train_lm.NgramTable(all_codes, o, a.topk) for o in (2, 3)]
    say(f"M1 built: {ngrams[0].contexts:,} bigram and {ngrams[1].contexts:,} trigram contexts")
    sep = enc.encode(b"\n\n")[0]
    PAD = 0
    # Codes the training text never contains (case flags on a plain dictionary, unused pieces)
    # are never emitted: their logits are untrained and a flat top-k can otherwise pick one.
    never_seen = torch.from_numpy(np.bincount(all_codes, minlength=enc.vocab) == 0)
    if a.eos:
        never_seen[EOS] = False
    EQUALS = int(enc.encode(b"=")[0][-1])
    NEWLINE = int(enc.encode(bytes([10]))[0][-1])
    checks = collections.Counter()
    dev = torch.device(a.device)
    model = V2LM(enc, a.width, a.layers, a.heads, a.ctx, ngrams, a.copy_head, a.evidence, a.sim_feature, a.block, a.dropout).to(dev)
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        own = model.state_dict()
        sd = {k: v for k, v in ck["state_dict"].items() if k in own and own[k].shape == v.shape and not k.startswith("table")}
        missing, unexpected = model.load_state_dict(sd, strict=False)
        say(f"initialised {len(sd)} tensors from {a.init}; fresh: {sorted({k.split('.')[0] for k in missing})}")
    if getattr(a, "precision", "fp32") != "fp32":
        import precision
        say(f"weights rounded: {precision.apply(model, a.precision)}")
    if a.rerank_core:
        mem.reranker = CoreReranker(model, enc, codes_of, sep, a.neighbor_codes, a.ctx, dev, a.rerank_core)
        say(f"retrieval reranked by the core: top {a.rerank_core} candidates")
    say(f"core: {model.param_counts()['total']:,} learned parameters, sources: network, 2 tables"
        + (", copy head" if a.copy_head else "") + (", evidence table" if a.evidence else ""))
    dense = [p for n, p in model.named_parameters()]
    # --head-lr: the heads that start untrained (copy head, trust head) learn at their own rate while
    # the pre-trained core moves gently (S2a-1: a low rate everywhere kept the core but slowed copying)
    fresh = ({n for n, _ in model.named_parameters() if n.split(".")[0] in ("copy_q", "copy_k", "mix_head")}
             if (a.head_lr or a.freeze_core_steps) else set())
    groups = []
    for is_head in (False, True):
        ps = [(n, p) for n, p in model.named_parameters() if (n in fresh) == is_head]
        rate = (a.head_lr or a.lr) if is_head else a.lr
        groups += [{"params": [p for n, p in ps if p.dim() >= 2], "weight_decay": 0.1, "lr": rate, "peak": rate, "core": not is_head},
                   {"params": [p for n, p in ps if p.dim() < 2], "weight_decay": 0.0, "lr": rate, "peak": rate, "core": not is_head}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, 0.95))
    rehearse_tokens = None
    if a.rehearse:
        # stage-1 text mixed into stage 2 so the core keeps what stage 1 taught it
        import glob as _glob
        files = sorted(_glob.glob(f"{a.rehearse}/train/*.txt"))
        concat = work / "rehearse_concat.txt"
        if not concat.exists():
            concat.write_bytes(b"".join(Path(f).read_bytes() for f in files))
        rehearse_tokens = enc.encode_file(str(concat), work)[0]
        say(f"rehearsal: {len(rehearse_tokens):,} stage-1 codes from {len(files)} files, share {a.rehearse_p}")

    def sample(i: int):
        """[nearest other exchange, tail] + SEP + [this exchange, head], padded to ctx+1, with a loss mask."""
        own_codes, ql = codes[i], None
        if a.typo_p and rng.random() < a.typo_p:
            # --typo-p: this window's question is typed with realistic mistakes; the answer is unchanged
            q_t = add_typos(train_x[i][0], rng)
            own_codes = enc.encode(render(q_t, train_x[i][1]))[0]
            ql = len(enc.encode(render(q_t, None))[0])
        ql = q_len(i) if ql is None else ql
        own = (np.append(own_codes, EOS) if a.eos else own_codes)[:a.ctx - a.neighbor_codes - len(sep)]
        src_i = i if (a.self_context_p and rng.random() < a.self_context_p) else neighbors[i]
        if a.no_retrieval_p and src_i != i and rng.random() < a.no_retrieval_p:
            src_i = None                     # nothing retrieved: the core must answer alone, as at test time when memory has nothing
        elif getattr(a, "distractor_p", 0.0) and src_i not in (i, None) and rng.random() < a.distractor_p:
            # a wrong exchange in the window (HYBRID_DESIGN 3.6): copying from it is wrong, so the model
            # must learn from the low similarity and the content when not to copy
            src_i = int(rng.integers(len(train_x) - 1)); src_i += src_i >= i
        nb = codes_of(src_i)[-a.neighbor_codes:] if a.neighbor_codes and src_i is not None else np.array([], dtype=np.int64)
        sim = 0.0 if src_i is None else (1.0 if src_i == i else mem.similarity_keys(mem.keys[i], mem.keys[src_i]))
        perturbed = False
        if src_i == i and a.perturb_self and rng.random() < a.perturb_self and len(nb):
            # the exchange itself, but with its question reworded: copying is right, and the
            # question no longer matches word for word (the case the chat probe showed failing)
            q_i, ans_i = train_x[i]
            q_p = perturb_question(q_i, rng, train_x[int(rng.integers(len(train_x)))][0].split(b" "), SYN)
            nb = enc.encode(render(q_p, ans_i))[0][-a.neighbor_codes:]
            sim = mem.similarity_keys(mem.keys[i], mem.key_of(q_p))
            perturbed = True
        seq = np.concatenate([nb, sep, own])[:a.ctx + 1]
        mask = np.zeros(a.ctx + 1, dtype=np.float32)
        # --loss-answer-only: learn to answer, not to predict the question (a SQuAD passage recurs
        # with 3-5 questions, so predicting it is memorisation)
        mask[len(nb) + len(sep) + (ql if a.loss_answer_only else 0):len(seq)] = 1.0
        src = np.zeros(a.ctx + 1, dtype=bool)                               # the retrieved span
        src[(answer_start(src_i, len(nb)) if src_i is not None and not perturbed else 0):len(nb)] = True
        if a.copy_question:
            q0 = len(nb) + len(sep)
            src[q0:min(q0 + ql, len(seq))] = True
        return np.pad(seq, (0, a.ctx + 1 - len(seq)), constant_values=PAD), mask, src, sim

    def rehearsal_sample():
        j = int(rng.integers(0, len(rehearse_tokens) - a.ctx - 1))
        seq = rehearse_tokens[j:j + a.ctx + 1].astype(np.int64)
        return seq, np.ones(a.ctx + 1, dtype=np.float32), np.zeros(a.ctx + 1, dtype=bool), 0.0

    def batch(idx):
        xs, ms, ss, sims = zip(*(rehearsal_sample() if rehearse_tokens is not None and rng.random() < a.rehearse_p else sample(i) for i in idx))
        x = torch.from_numpy(np.stack(xs)).to(dev); m = torch.from_numpy(np.stack(ms)).to(dev)
        s = torch.from_numpy(np.stack(ss)).to(dev)
        return x[:, :-1], x[:, 1:], m[:, 1:], s[:, :-1], torch.tensor(sims, dtype=torch.float32, device=dev)

    def span_mask(seq_len: int, nb_len: int, start: int = 0, question: tuple[int, int] | None = None) -> torch.Tensor:
        """Which positions of the window seq[-ctx:] belong to the retrieved span (from start),
        plus the current question's positions when --copy-question."""
        offset = max(0, seq_len - a.ctx)
        n = min(seq_len, a.ctx)
        pos = torch.arange(n, device=dev) + offset
        m = (pos < nb_len) & (pos >= start)
        if a.copy_question and question is not None:
            m |= (pos >= question[0]) & (pos < question[1])
        return m[None]

    @torch.no_grad()
    def valid_bpb(exchanges, n=120, ids=None):
        """Bits per byte of the answers with the nearest memory in the window. With ``ids`` (training exchanges),
        each exchange's own entry is hidden from retrieval, so the training side is scored exactly as held-out
        questions are and the two numbers form an overfitting gap (--track-gap)."""
        model.eval(); bits = nbytes = 0.0
        for k, (q, ans) in enumerate(exchanges[:n]):
            own = enc.encode(render(q, ans))[0]
            if a.eos:
                own = np.append(own, EOS)
            nb_i = mem.retrieve(q, 1, exclude=None if ids is None else ids[k])
            nb = codes_of(nb_i[0][1])[-a.neighbor_codes:] if a.neighbor_codes and nb_i else np.array([], dtype=np.int64)
            seq = np.concatenate([nb, sep, own])[:a.ctx + 1]
            if len(seq) < 3:
                continue
            x = torch.from_numpy(seq[None, :-1]).to(dev); y = torch.from_numpy(seq[None, 1:]).to(dev)
            skip = len(enc.encode(render(q, None))[0]) if a.loss_answer_only else 0
            scored = seq[len(nb) + len(sep) + skip:]                # the exchange's own codes (or its answer), each predicted
            start = answer_start(nb_i[0][1], len(nb)) if nb_i else 0
            q0 = len(nb) + len(sep); qspan = (q0, q0 + len(enc.encode(render(q, None))[0]))
            sim_v = torch.tensor([nb_i[0][0] if nb_i else 0.0], device=dev)
            nll = model.nll(x, y, span_mask(len(seq) - 1, len(nb), start, qspan), sim_v)[0, len(nb) + len(sep) - 1 + skip:]
            bits += float(nll.sum()) / math.log(2); nbytes += float(enc.lens[scored].sum())
        model.train(); return bits / max(1, nbytes)

    history, train_time, step, next_eval = [], 0.0, 0, 60.0
    gap_ids = [int(j) for j in np.random.default_rng(a.seed + 7).choice(len(train_x), size=min(120, len(train_x)), replace=False)] if a.track_gap else []
    best_v, bad_evals, best_state, stop_reason = float("inf"), 0, None, "budget"
    import cost as _cost
    meter = _cost.Meter(dev).__enter__()
    model.train()
    while train_time < a.budget and not (a.max_steps and step >= a.max_steps):
        t = time.perf_counter()
        x, y, m, s, sims = batch(rng.integers(0, len(train_x), size=a.batch))
        # --amp: the training step in bfloat16 autocast (weights and optimiser stay float32; validation
        # and every score stay in full precision), as the core stage has trained since the scale run
        if a.schedule == "cosine":   # warm-up over 100 steps, cosine decay to 10% over the run (the usual fine-tuning schedule)
            frac = step / a.max_steps if a.max_steps else train_time / a.budget
            for g in opt.param_groups:
                g["lr"] = g["peak"] * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))
        if a.freeze_core_steps:
            # --freeze-core-steps: the untrained heads learn alone first, so their early, large errors never
            # reach the pre-trained core; then the core joins at its gentle rate (two-phase fine-tuning)
            for g in opt.param_groups:
                if g.get("core"):
                    g["lr"] = 0.0 if step < a.freeze_core_steps else (g["lr"] if a.schedule == "cosine" else g["peak"])
        opt.zero_grad(set_to_none=True)
        # --accum: the batch in equal slices with summed gradients, normalised by the whole batch's mask count,
        # so the update equals the full-batch one with less activation memory (13 October, for the 100M stage)
        denom = m.sum().clamp_min(1.0); loss = 0.0
        for part in np.array_split(np.arange(x.shape[0]), getattr(a, "accum", 1)):
            with torch.autocast(device_type="cuda" if dev.type == "cuda" else "cpu", dtype=torch.bfloat16, enabled=a.amp):
                nll = model.nll(x[part], y[part], s[part], sims[part])
            part_loss = (nll.float() * m[part]).sum() / denom
            part_loss.backward(); loss = loss + part_loss.detach()
        torch.nn.utils.clip_grad_norm_(dense, 1.0); opt.step()
        step += 1; train_time += time.perf_counter() - t
        if train_time >= next_eval or train_time >= a.budget or (a.max_steps and step >= a.max_steps):
            v = valid_bpb(eval_x)
            row = {"train_s": round(train_time, 1), "step": step, "train_bits_per_code": round(float(loss) / math.log(2), 3), "valid_bits_per_byte": round(v, 4)}
            gap_note = ""
            if a.track_gap:
                tv = valid_bpb([train_x[j] for j in gap_ids], ids=gap_ids)
                row["train_heldout_style_bits_per_byte"] = round(tv, 4); row["gap_bits_per_byte"] = round(v - tv, 4)
                gap_note = f"  train-side {tv:.4f}  gap {v - tv:+.4f}"
            history.append(row)
            say(f"{train_time:6.0f}s step {step:5d} train {float(loss)/math.log(2):.3f} b/code  valid {v:.4f} b/byte{gap_note}"); next_eval += 60
            if v < best_v - a.min_delta:
                best_v, bad_evals = v, 0
                best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
            else:
                bad_evals += 1
                if a.patience and bad_evals >= a.patience:
                    stop_reason = f"plateau: no improvement over {a.min_delta} in {a.patience} evaluations"
                    say(f"stopping at {train_time:.0f}s: {stop_reason}"); break
    if a.patience and best_state is not None:
        model.load_state_dict(best_state); say(f"restored the best weights (valid {best_v:.4f} b/byte)")
    if a.save:   # saved before scoring, so an interrupted evaluation does not lose the training
        Path(a.save).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(a)}, a.save)

    @torch.no_grad()
    def answer(q: bytes, context_exchange: int | None, n_tokens=80, temperature=0.5, top_k=10, use_m1=True, copy_bias=0.0):
        model.eval(); gen = torch.Generator().manual_seed(a.seed)
        saved = model.ngrams
        if not use_m1:
            model.ngrams = []
        nb = codes_of(context_exchange)[-a.neighbor_codes:] if a.neighbor_codes and context_exchange is not None else np.array([], dtype=np.int64)
        bias = torch.zeros(enc.vocab, device=dev)
        if copy_bias and context_exchange is not None:
            ans_codes = enc.encode(mem.exchanges[context_exchange][1])[0]
            bias[torch.from_numpy(np.unique(ans_codes))] = copy_bias
        a_start = answer_start(context_exchange, len(nb)) if context_exchange is not None else 0
        sim_v = torch.tensor([mem.similarity_keys(mem.key_of(q), mem.keys[context_exchange]) if context_exchange is not None else 0.0], device=dev)
        seq = [int(t) for t in np.concatenate([nb, sep, enc.encode(render(q, None))[0]])]
        start = len(seq)
        qspan = (len(nb) + len(sep), start)
        computed: list[str] = []            # results the executor wrote, in order
        close_number = False
        while len(seq) - start < n_tokens:
            x = torch.tensor([seq[-a.ctx:]], device=dev)
            h = model.hidden_all(x)
            logits = (model.mixed_logits(h, x, span_mask(len(seq), len(nb), a_start, qspan), sim_v) + bias) / temperature
            logits = logits.masked_fill(never_seen.to(logits.device), float("-inf"))
            if a.no_repeat:
                banned = repeat_banned(seq[start:], a.no_repeat)
                if banned:
                    logits[0, banned] = float("-inf")
            if a.greedy:
                seq.append(int(logits.argmax(-1).item()))
            else:
                kth = torch.topk(logits, top_k).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
                seq.append(torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=gen).item())
            if a.eos and seq[-1] == EOS:          # the model ended its answer
                seq.pop()
                break
            if a.stop_at_blank and b"\n\n" in enc.decode(seq[start:]):
                break
            if a.calculator and seq[-1] == EQUALS:
                value = evaluate_tail(enc.decode(seq[start:]))
                if value is not None:
                    computed.append(value)
                    seq.extend(int(t) for t in enc.encode(b" " + value.encode())[0])
                    close_number = True
                    continue
            if a.calculator and close_number:
                # the codes after an executor value may not extend the number: no digit right after
                # it, and no digit right after a point that follows it ("= 0" + ".5")
                first = enc.d.id_bytes[seq[-1]][:1]
                if first.isdigit():
                    seq[-1] = NEWLINE; close_number = False
                elif first == b"." and close_number is True and enc.d.id_bytes[seq[-1]] == b".":
                    close_number = "point"
                else:
                    close_number = False
        out = enc.decode(seq[start:]).split(b"\nUser:")[0].split(b"User:")[0]
        out = (out.split(b"\n\n")[0] if a.stop_at_blank else out).strip()
        if a.calculator and computed:
            # Rule 2: the stated final answer must be a value the executor computed; otherwise the
            # last computed value is the answer (a refine, counted in results["checks"]).
            said = final_number(out)
            if said is None or said.decode() not in computed:
                checks["answer_replaced"] += 1
                out = NUM.sub(b"Answer: " + computed[-1].encode(), out) if said is not None else out + b"\nAnswer: " + computed[-1].encode()
            else:
                checks["answer_confirmed"] += 1
        model.ngrams = saved; model.train(); return out

    def score(rows):
        out = {"n": len(rows), "f1": round(float(np.mean([r["f1"] for r in rows])), 4) if rows else None,
               "exact": round(float(np.mean([r["exact"] for r in rows])), 4) if rows else None}
        nums = [r["num_exact"] for r in rows if r.get("num_exact") is not None]
        if nums:
            out["num_exact"] = round(float(np.mean(nums)), 4)
            out["num_exact_bias"] = round(float(np.mean([r["num_exact_bias"] for r in rows if r.get("num_exact_bias") is not None])), 4)
        if rows and "f1_no_m1" in rows[0]:
            out["f1_no_m1"] = round(float(np.mean([r["f1_no_m1"] for r in rows])), 4)
        if rows and "f1_bias" in rows[0]:
            out["f1_bias"] = round(float(np.mean([r["f1_bias"] for r in rows])), 4)
            out["exact_bias"] = round(float(np.mean([r["exact_bias"] for r in rows])), 4)
        return out

    results = {"setup": vars(a) | {"train_exchanges": len(train_x), "eval": len(eval_x), "added": len(add_x), "distinct_keys": len(mem.exact)},
               "training": {"steps": step, "train_s": round(train_time, 1), "stop_reason": stop_reason, "best_valid": None if best_v == float("inf") else round(best_v, 4), "history": history,
                            "cost": (meter.__exit__(None, None, None), meter.summary(train_time, step))[1]}, "groups": {}, "samples": {}}
    # 1. seen: training questions. Memory path (exact key) and network path (own exchange retrieved).
    seen_idx = rng.choice(len(train_x), size=min(a.eval_n, len(train_x)), replace=False)
    mem_rows, net_rows = [], []
    for i in seen_idx:
        q, ref = train_x[i]
        hit = mem.exact_hit(q)
        mem_rows.append({"f1": f1(mem.exchanges[hit][1], ref) if hit is not None else 0.0, "exact": float(hit is not None and mem.exchanges[hit][1] == ref)})
        if len(net_rows) < a.net_n:
            out = answer(q, hit); out2 = answer(q, hit, use_m1=False); out3 = answer(q, hit, copy_bias=a.copy_bias)
            net_rows.append({"q": q.decode("utf-8", "replace")[:80], "a": out.decode("utf-8", "replace")[:160], "f1": f1(out, ref), "exact": float(out == ref),
                             "a_no_m1": out2.decode("utf-8", "replace")[:160], "f1_no_m1": f1(out2, ref),
                             "a_bias": out3.decode("utf-8", "replace")[:160], "f1_bias": f1(out3, ref), "exact_bias": float(out3 == ref),
                             "num_exact": num_exact(out, ref), "num_exact_bias": num_exact(out3, ref)})
    results["groups"]["seen_memory_path"] = score(mem_rows)
    results["groups"]["seen_network_with_own_exchange_in_context"] = score(net_rows)
    results["samples"]["seen_network"] = net_rows[:6]
    # 2./3. held-out: paraphrase (strong overlap with a training key) vs unseen; network path with the nearest exchange.
    para_rows, unseen_rows, nomem_rows = [], [], []
    for q, ref in eval_x[:a.eval_n]:
        r = mem.retrieve(q, 1)
        sim = r[0][0] if r else 0.0
        nb = r[0][1] if r else None
        out = answer(q, nb); out2 = answer(q, nb, use_m1=False); out3 = answer(q, nb, copy_bias=a.copy_bias)
        row = {"q": q.decode("utf-8", "replace")[:80], "a": out.decode("utf-8", "replace")[:160], "f1": f1(out, ref), "exact": float(out == ref),
               "a_no_m1": out2.decode("utf-8", "replace")[:160], "f1_no_m1": f1(out2, ref),
               "a_bias": out3.decode("utf-8", "replace")[:160], "f1_bias": f1(out3, ref), "exact_bias": float(out3 == ref),
               "num_exact": num_exact(out, ref), "num_exact_bias": num_exact(out3, ref), "sim": round(sim, 3), "f1_vs_retrieved": f1(out, mem.exchanges[nb][1]) if nb is not None else 0.0,
               "retrieved_answer_f1_vs_ref": f1(mem.exchanges[nb][1], ref) if nb is not None else 0.0}
        (para_rows if sim >= 0.5 else unseen_rows).append(row)
        if len(nomem_rows) < a.net_n:
            out0 = answer(q, None)
            nomem_rows.append({"f1": f1(out0, ref), "exact": float(out0 == ref), "num_exact": num_exact(out0, ref), "num_exact_bias": num_exact(out0, ref)})
    results["groups"]["paraphrase_network_with_nearest"] = score(para_rows) | {"retrieved_answer_f1_vs_ref": round(float(np.mean([r["retrieved_answer_f1_vs_ref"] for r in para_rows])), 4) if para_rows else None}
    results["groups"]["unseen_network_with_nearest"] = score(unseen_rows) | {"retrieved_answer_f1_vs_ref": round(float(np.mean([r["retrieved_answer_f1_vs_ref"] for r in unseen_rows])), 4) if unseen_rows else None}
    results["groups"]["heldout_network_without_memory"] = score(nomem_rows)
    pol = [r["retrieved_answer_f1_vs_ref"] if r["sim"] >= 0.5 else r["f1"] for r in para_rows + unseen_rows]
    results["groups"]["heldout_policy_retrieved_if_similar_else_generated"] = {"n": len(pol), "f1": round(float(np.mean(pol)), 4) if pol else None}
    results["samples"]["paraphrase"] = para_rows[:6]; results["samples"]["unseen"] = unseen_rows[:6]
    # 4. learning by adding: new pairs into M2 only, then asked at once (memory path and network path).
    for q, ans in add_x:
        mem.add(q, ans)
    add_mem, add_net = [], []
    for q, ref in add_x[:a.eval_n]:
        hit = mem.exact_hit(q)
        add_mem.append({"f1": f1(mem.exchanges[hit][1], ref) if hit is not None else 0.0, "exact": float(hit is not None and mem.exchanges[hit][1] == ref)})
        if len(add_net) < a.net_n:
            out = answer(q, hit); out2 = answer(q, hit, use_m1=False); out3 = answer(q, hit, copy_bias=a.copy_bias)
            add_net.append({"q": q.decode("utf-8", "replace")[:80], "a": out.decode("utf-8", "replace")[:160], "f1": f1(out, ref), "exact": float(out == ref),
                            "a_no_m1": out2.decode("utf-8", "replace")[:160], "f1_no_m1": f1(out2, ref),
                            "a_bias": out3.decode("utf-8", "replace")[:160], "f1_bias": f1(out3, ref), "exact_bias": float(out3 == ref),
                             "num_exact": num_exact(out, ref), "num_exact_bias": num_exact(out3, ref)})
    results["checks"] = dict(checks)
    results["groups"]["added_memory_path"] = score(add_mem)
    results["groups"]["added_network_with_added_exchange_in_context"] = score(add_net)
    results["samples"]["added_network"] = add_net[:6]
    Path(a.out).write_text(json.dumps(results, indent=1))
    for g, v in results["groups"].items():
        say(f"{g:48s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
