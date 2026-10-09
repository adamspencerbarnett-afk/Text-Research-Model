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

    def __init__(self, enc, width, layers, heads, ctx, ngrams, copy_head=False, evidence=False):
        super().__init__(enc, width, layers, heads, ctx, 0, None, (2,), ngrams)
        self.copy_head, self.evidence = copy_head, evidence
        n_src = 1 + len(self.ngrams) + copy_head + evidence
        n_conf = 3 * len(self.ngrams) + 2 * copy_head + 1 * evidence
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

    def extra_parts(self, h, x, src, y=None):
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
        return parts, confs

    def nll(self, x, y, src=None):
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
        parts, extra = self.extra_parts(h, x, src, y)
        p_parts += parts; confs += extra
        mix = F.softmax(self.mix_head(torch.cat([h] + [c[..., None] for c in confs], -1)), -1)
        p = (mix * torch.stack(p_parts, -1)).sum(-1)
        return -torch.log(p.clamp_min(1e-9))

    def mixed_logits(self, h_last, x, src=None):
        """Full next-code log-probabilities at the last position; ``h_last`` is (B, D) and the
        copy head needs every state of the window, so ``hidden_all`` must be called first."""
        logits = h_last @ self.emb.weight.T
        if not self.ngrams:                       # the "network alone" ablation, as in the base class
            return logits
        p, confs = [F.softmax(logits, -1)], []
        for table in self.ngrams:
            ids, counts, totals = table.features(x, h_last.device)
            dense = torch.zeros_like(p[0])
            ok = ids[:, -1] >= 0
            probs = counts[:, -1] / counts[:, -1].sum(-1, keepdim=True).clamp_min(1.0)
            dense.scatter_add_(-1, ids[:, -1].clamp_min(0), probs * ok)
            p.append(dense)
            confs.extend(f[:, -1] for f in train_lm.table_features(counts, totals, 0.0))
        if src is None:
            src = torch.zeros_like(x, dtype=torch.bool)
        parts, extra = self.extra_parts(h_last[:, None, :], x, src)
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
        if not scores:
            return []
        denom = {w: self.idf(w) for w in key}
        out = []
        for i, s in scores.items():
            union = sum(denom.values()) + sum(self.idf(w) for w in self.keys[i] if w not in key)
            out.append((s / max(union, 1e-9), i))
        out.sort(reverse=True)
        return out[:k]

    def exact_hit(self, question: bytes):
        return self.exact.get(self.key_of(question))


def parse_exchanges(text: bytes) -> list[tuple[bytes, bytes]]:
    out = []
    for block in text.split(b"\n\n"):
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
    ap.add_argument("--greedy", action="store_true", help="scored answers take the most likely code at each step instead of sampling")
    ap.add_argument("--copy-head", action="store_true", help="learned pointer over the retrieved exchange as a trust-head source")
    ap.add_argument("--evidence", action="store_true", help="bigram table counted over the retrieved exchange as a trust-head source")
    ap.add_argument("--fold-keys", action="store_true", help="fold inflections in the question key")
    ap.add_argument("--copy-answer-only", action="store_true", help="the copy head and evidence table see only the answer part of the retrieved exchange")
    ap.add_argument("--copy-question", action="store_true", help="the copy head may also point into the current question (operands, names)")
    ap.add_argument("--calculator", action="store_true", help="check loop, rule 1: when the reply has written 'a op b =', the executor writes the result; the final Answer: is checked against the last computed value")
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

    enc = LE.V0Dict(str(ladder1.fit_dictionary("v0_8k_plain", [f"{a.data}/train/qa_train.txt"], work)), a.native)
    mem = ExchangeMemory()
    for q, ans in train_x:
        mem.add(q, ans)
    t0 = time.time()
    neighbors = [r[0][1] if (r := mem.retrieve(q, 1, exclude=i)) else None for i, (q, _) in enumerate(train_x)]
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
    EQUALS = int(enc.encode(b"=")[0][-1])
    checks = collections.Counter()
    dev = torch.device(a.device)
    model = V2LM(enc, a.width, a.layers, a.heads, a.ctx, ngrams, a.copy_head, a.evidence).to(dev)
    say(f"core: {model.param_counts()['total']:,} learned parameters, sources: network, 2 tables"
        + (", copy head" if a.copy_head else "") + (", evidence table" if a.evidence else ""))
    dense = [p for n, p in model.named_parameters()]
    opt = torch.optim.AdamW([{"params": [p for p in dense if p.dim() >= 2], "weight_decay": 0.1},
                             {"params": [p for p in dense if p.dim() < 2], "weight_decay": 0.0}], lr=1e-3, betas=(0.9, 0.95))

    def sample(i: int):
        """[nearest other exchange, tail] + SEP + [this exchange, head], padded to ctx+1, with a loss mask."""
        own = codes[i][:a.ctx - a.neighbor_codes - len(sep)]
        src_i = i if (a.self_context_p and rng.random() < a.self_context_p) else neighbors[i]
        nb = codes_of(src_i)[-a.neighbor_codes:] if src_i is not None else np.array([], dtype=np.int64)
        seq = np.concatenate([nb, sep, own])[:a.ctx + 1]
        mask = np.zeros(a.ctx + 1, dtype=np.float32); mask[len(nb) + len(sep):len(seq)] = 1.0
        src = np.zeros(a.ctx + 1, dtype=bool)                               # the retrieved span
        src[(answer_start(src_i, len(nb)) if src_i is not None else 0):len(nb)] = True
        if a.copy_question:
            q0 = len(nb) + len(sep)
            src[q0:min(q0 + q_len(i), len(seq))] = True
        return np.pad(seq, (0, a.ctx + 1 - len(seq)), constant_values=PAD), mask, src

    def batch(idx):
        xs, ms, ss = zip(*(sample(i) for i in idx))
        x = torch.from_numpy(np.stack(xs)).to(dev); m = torch.from_numpy(np.stack(ms)).to(dev)
        s = torch.from_numpy(np.stack(ss)).to(dev)
        return x[:, :-1], x[:, 1:], m[:, 1:], s[:, :-1]

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
    def valid_bpb(exchanges, n=120):
        model.eval(); bits = nbytes = 0.0
        for q, ans in exchanges[:n]:
            own = enc.encode(render(q, ans))[0]
            nb_i = mem.retrieve(q, 1)
            nb = codes_of(nb_i[0][1])[-a.neighbor_codes:] if nb_i else np.array([], dtype=np.int64)
            seq = np.concatenate([nb, sep, own])[:a.ctx + 1]
            if len(seq) < 3:
                continue
            x = torch.from_numpy(seq[None, :-1]).to(dev); y = torch.from_numpy(seq[None, 1:]).to(dev)
            scored = seq[len(nb) + len(sep):]                       # the exchange's own codes, each predicted
            start = answer_start(nb_i[0][1], len(nb)) if nb_i else 0
            q0 = len(nb) + len(sep); qspan = (q0, q0 + len(enc.encode(render(q, None))[0]))
            nll = model.nll(x, y, span_mask(len(seq) - 1, len(nb), start, qspan))[0, len(nb) + len(sep) - 1:]
            bits += float(nll.sum()) / math.log(2); nbytes += float(enc.lens[scored].sum())
        model.train(); return bits / max(1, nbytes)

    history, train_time, step, next_eval = [], 0.0, 0, 60.0
    model.train()
    while train_time < a.budget:
        t = time.perf_counter()
        x, y, m, s = batch(rng.integers(0, len(train_x), size=a.batch))
        nll = model.nll(x, y, s)
        loss = (nll * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(dense, 1.0); opt.step()
        step += 1; train_time += time.perf_counter() - t
        if train_time >= next_eval or train_time >= a.budget:
            v = valid_bpb(eval_x)
            history.append({"train_s": round(train_time, 1), "step": step, "train_bits_per_code": round(float(loss) / math.log(2), 3), "valid_bits_per_byte": round(v, 4)})
            say(f"{train_time:6.0f}s step {step:5d} train {float(loss)/math.log(2):.3f} b/code  valid {v:.4f} b/byte"); next_eval += 60

    @torch.no_grad()
    def answer(q: bytes, context_exchange: int | None, n_tokens=80, temperature=0.5, top_k=10, use_m1=True, copy_bias=0.0):
        model.eval(); gen = torch.Generator().manual_seed(a.seed)
        saved = model.ngrams
        if not use_m1:
            model.ngrams = []
        nb = codes_of(context_exchange)[-a.neighbor_codes:] if context_exchange is not None else np.array([], dtype=np.int64)
        bias = torch.zeros(enc.vocab, device=dev)
        if copy_bias and context_exchange is not None:
            ans_codes = enc.encode(mem.exchanges[context_exchange][1])[0]
            bias[torch.from_numpy(np.unique(ans_codes))] = copy_bias
        a_start = answer_start(context_exchange, len(nb)) if context_exchange is not None else 0
        seq = [int(t) for t in np.concatenate([nb, sep, enc.encode(render(q, None))[0]])]
        start = len(seq)
        qspan = (len(nb) + len(sep), start)
        computed: list[str] = []            # results the executor wrote, in order
        while len(seq) - start < n_tokens:
            x = torch.tensor([seq[-a.ctx:]], device=dev)
            h = model.hidden_all(x)
            logits = (model.mixed_logits(h, x, span_mask(len(seq), len(nb), a_start, qspan)) + bias) / temperature
            logits = logits.masked_fill(never_seen.to(logits.device), float("-inf"))
            if a.greedy:
                seq.append(int(logits.argmax(-1).item()))
            else:
                kth = torch.topk(logits, top_k).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
                seq.append(torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=gen).item())
            if a.calculator and seq[-1] == EQUALS:
                value = evaluate_tail(enc.decode(seq[start:]))
                if value is not None:
                    computed.append(value)
                    seq.extend(int(t) for t in enc.encode(b" " + value.encode())[0])
        out = enc.decode(seq[start:]).split(b"\nUser:")[0].split(b"User:")[0].strip()
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
               "training": {"steps": step, "train_s": round(train_time, 1), "history": history}, "groups": {}, "samples": {}}
    # 1. seen: training questions. Memory path (exact key) and network path (own exchange retrieved).
    seen_idx = rng.choice(len(train_x), size=min(a.eval_n, len(train_x)), replace=False)
    mem_rows, net_rows = [], []
    for i in seen_idx:
        q, ref = train_x[i]
        hit = mem.exact_hit(q)
        mem_rows.append({"f1": f1(mem.exchanges[hit][1], ref) if hit is not None else 0.0, "exact": float(hit is not None and mem.exchanges[hit][1] == ref)})
        if len(net_rows) < 60:
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
        if len(nomem_rows) < 60:
            out0 = answer(q, None)
            nomem_rows.append({"f1": f1(out0, ref), "exact": float(out0 == ref), "num_exact": num_exact(out0, ref), "num_exact_bias": num_exact(out0, ref)})
    results["groups"]["paraphrase_network_with_nearest"] = score(para_rows) | {"retrieved_answer_f1_vs_ref": round(float(np.mean([r["retrieved_answer_f1_vs_ref"] for r in para_rows])), 4) if para_rows else None}
    results["groups"]["unseen_network_with_nearest"] = score(unseen_rows) | {"retrieved_answer_f1_vs_ref": round(float(np.mean([r["retrieved_answer_f1_vs_ref"] for r in unseen_rows])), 4) if unseen_rows else None}
    results["groups"]["heldout_network_without_memory"] = score(nomem_rows)
    pol = [r["retrieved_answer_f1_vs_ref"] if r["sim"] >= 0.5 else r["f1"] for r in para_rows + unseen_rows]
    results["groups"]["heldout_policy_retrieved_if_similar_else_generated"] = {"n": len(pol), "f1": round(float(np.mean(pol)), 4) if pol else None}
    if a.save:
        torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "args": vars(a)}, a.save)
    results["samples"]["paraphrase"] = para_rows[:6]; results["samples"]["unseen"] = unseen_rows[:6]
    # 4. learning by adding: new pairs into M2 only, then asked at once (memory path and network path).
    for q, ans in add_x:
        mem.add(q, ans)
    add_mem, add_net = [], []
    for q, ref in add_x[:a.eval_n]:
        hit = mem.exact_hit(q)
        add_mem.append({"f1": f1(mem.exchanges[hit][1], ref) if hit is not None else 0.0, "exact": float(hit is not None and mem.exchanges[hit][1] == ref)})
        if len(add_net) < 60:
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
