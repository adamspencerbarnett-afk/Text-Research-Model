"""Train one small language model on one encoding for a fixed training-time budget, then
report bits per original byte on held-out text, sequence length, speed and a sample of the
model talking back through the same encoder.

Used by ladder1.py; can also be run alone:

    python compiler/experiments/train_lm.py --encoding bytes --data DATA --work WORK --budget 600 --out r.json
    python compiler/experiments/train_lm.py --encoding v0 --dict WORK/v0_8k_plain.cv0d --native ./cv0 ...
    python compiler/experiments/train_lm.py --encoding hash --hash-map WORK/hash4096x4096.json ...

The model is a small pre-norm decoder-only Transformer. With a flat vocabulary the input
and output embeddings are tied. With coordinate codes (HashCodes) the input embedding is
the sum of a group vector and a member vector, and the output predicts the group, then the
member given the true group; both are cheap softmaxes. ``--table-rows N`` adds a hashed
bigram table (previous token, current token) -> row, summed into the input embedding: many
parameters, one extra lookup per token.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_encodings as LE  # noqa: E402

LN2 = math.log(2)


# --------------------------------------------------------------------------- model

class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.fc1, self.fc2 = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        x = x + self.proj(y)
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


def mix_bigram(prev: torch.Tensor, cur: torch.Tensor, rows: int) -> torch.Tensor:
    """Hash (previous token, current token) to a table row; rows must be a power of two."""
    a = (prev * 1000003 + cur) * 2654435761
    a = a ^ (a >> 21) ^ (a >> 42)
    return a & (rows - 1)


def mix_ngram(x: torch.Tensor, order: int, rows: int, pad: int) -> torch.Tensor:
    """Hash the n-gram ending at each position (the current token and order-1 before it) to a
    table row. Positions near the start use ``pad`` for the missing history."""
    B, T = x.shape
    a = x.clone()
    for k in range(1, order):
        shifted = torch.cat([torch.full((B, k), pad, dtype=x.dtype), x[:, :-k]], dim=1)
        a = (a * 1000003 + shifted) * 2654435761
        a = a ^ (a >> 21) ^ (a >> 42)
    return a & (rows - 1)


class NgramTable:
    """Exact n-gram follower lists for a count-based component at the output.

    For every context of ``order - 1`` tokens seen in training, keeps the ``topk`` most frequent
    next tokens with their frequencies (renormalised over the kept followers) and the context's
    total count. Lookup is a binary search over the sorted context keys, so a batch costs
    little. Flat vocabularies only; tokens must be below 2^21.
    """

    def __init__(self, tokens: np.ndarray, order: int, topk: int = 16):
        self.order, self.topk, self.k = order, topk, order - 1
        n = len(tokens)
        key = np.zeros(n - self.k, dtype=np.int64)
        for j in range(self.k):
            key = (key << 21) + tokens[j:n - self.k + j]
        pair = (key << 21) + tokens[self.k:]
        uniq, counts = np.unique(pair, return_counts=True)
        ctx, nxt = uniq >> 21, uniq & ((1 << 21) - 1)
        by = np.lexsort((-counts, ctx))
        ctx, nxt, counts = ctx[by], nxt[by], counts[by]
        self.keys, first = np.unique(ctx, return_index=True)
        sizes = np.diff(np.append(first, len(ctx)))
        totals = np.add.reduceat(counts, first)
        row = np.repeat(np.arange(len(self.keys)), sizes)
        rank = np.arange(len(ctx)) - np.repeat(first, sizes)
        keep = rank < topk
        self.ids = np.full((len(self.keys), topk), -1, dtype=np.int64)
        self.counts = np.zeros((len(self.keys), topk), dtype=np.float32)
        self.ids[row[keep], rank[keep]] = nxt[keep]
        self.counts[row[keep], rank[keep]] = counts[keep]
        self.totals = totals.astype(np.float32)
        self.contexts = len(self.keys)

    def features(self, x: np.ndarray):
        """For a (B, T) token array: follower ids (B, T, K), their counts (B, T, K) and the
        context's total count (B, T); -1 / 0 where the context is unseen or incomplete."""
        B, T = x.shape
        key = np.zeros((B, T), dtype=np.int64)
        for j in range(self.k):
            shift = self.k - 1 - j
            col = np.zeros((B, T), dtype=np.int64)
            col[:, shift:] = x[:, :T - shift]
            key = (key << 21) + col
        pos = np.minimum(np.searchsorted(self.keys, key), len(self.keys) - 1)
        found = (self.keys[pos] == key)
        found[:, :self.k - 1] = False
        ids = np.where(found[..., None], self.ids[pos], -1)
        counts = np.where(found[..., None], self.counts[pos], 0.0).astype(np.float32)
        totals = np.where(found, self.totals[pos], 0.0).astype(np.float32)
        return torch.from_numpy(ids), torch.from_numpy(counts), torch.from_numpy(totals)


def table_features(counts, totals, loo: float, ids=None, y=None):
    """Three confidence features per table for the trust head: log(1 + total count), the top
    follower's share and the entropy of the follower distribution (in bits). In training the
    current occurrence is removed first (leave-one-out), consistently with the probabilities."""
    c = counts
    if loo and ids is not None:
        c = (c - loo * (ids == y[..., None])).clamp_min(0.0)
    tot = (totals - loo * (totals > 0)).clamp_min(0.0)
    kept = c.sum(-1).clamp_min(1.0)
    p = c / kept[..., None]
    top = p.max(-1).values
    ent = -(p * torch.log2(p.clamp_min(1e-9))).sum(-1)
    return [torch.log1p(tot), top, ent]


class LM(nn.Module):
    INVALID = -30.0  # logit given to a (group, member) code that no unit in training maps to

    def __init__(self, enc: LE.Encoding, width: int, layers: int, heads: int, ctx: int, table_rows: int = 0,
                 valid_codes=None, table_orders=(2,), ngrams: list | None = None):
        super().__init__()
        self.ctx, self.coords = ctx, enc.groups > 0
        self.vocab = enc.vocab
        # Exact n-gram follower tables mixed into the output: the model predicts the mixture
        # weights (network, table 1, table 2, ...) from its hidden state and each table's
        # confidence, so it learns when to trust counts and when to think.
        self.ngrams = ngrams or []
        if self.ngrams and self.coords:
            raise NotImplementedError("n-gram output tables need a flat vocabulary")
        self.mix_head = nn.Linear(width + 3 * len(self.ngrams), len(self.ngrams) + 1) if self.ngrams else None
        if self.coords:
            self.G, self.M = enc.groups, enc.members
            self.emb_g, self.emb_m = nn.Embedding(self.G, width), nn.Embedding(self.M, width)
            self.head_g, self.head_m = nn.Linear(width, self.G), nn.Linear(2 * width, self.M)
            # Only codes that some training unit maps to can be decoded; the member softmax is
            # restricted to them, so the model never spends probability on undecodable codes.
            valid = torch.zeros(self.G * self.M, dtype=torch.bool)
            if valid_codes is not None:
                valid[torch.as_tensor(np.asarray(valid_codes, dtype=np.int64))] = True
            else:
                valid[:] = True
            self.register_buffer("valid", valid.view(self.G, self.M))
            self.register_buffer("valid_group", self.valid.any(1))   # groups with at least one code in use
        else:
            self.emb = nn.Embedding(enc.vocab, width)
        self.pos = nn.Embedding(ctx, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln_f = nn.LayerNorm(width)
        self.table_rows, self.table_orders = table_rows, tuple(table_orders) if table_rows else ()
        # One hashed n-gram table per order, summed into the input embedding: many parameters,
        # one lookup each per token. ``table`` keeps the bigram table's name for old checkpoints.
        self.table = nn.Embedding(table_rows, width, sparse=True) if table_rows and 2 in self.table_orders else None
        self.tables = nn.ModuleDict({str(o): nn.Embedding(table_rows, width, sparse=True)
                                     for o in self.table_orders if o != 2}) if table_rows else nn.ModuleDict()
        for t in [self.table] + list(self.tables.values()):
            if t is not None:
                nn.init.zeros_(t.weight)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for e in [getattr(self, n) for n in ("emb", "emb_g", "emb_m", "pos") if hasattr(self, n)]:
            nn.init.normal_(e.weight, std=0.02)

    def param_counts(self) -> dict:
        n = lambda mods: sum(p.numel() for m in mods for p in m.parameters())
        core = n(self.blocks) + n([self.ln_f])
        emb = n([self.pos]) + (n([self.emb_g, self.emb_m]) if self.coords else n([self.emb]))
        out = n([self.head_g, self.head_m]) if self.coords else 0   # tied output costs nothing extra
        out += n([self.mix_head]) if self.mix_head is not None else 0
        table = (n([self.table]) if self.table is not None else 0) + n(self.tables.values())
        return {"core": core, "embedding": emb, "output_heads": out, "table": table,
                "total": core + emb + out + table}

    def embed(self, x):
        B, T = x.shape
        if self.coords:
            e = self.emb_g(x // self.M) + self.emb_m(x % self.M)
        else:
            e = self.emb(x)
        if self.table is not None:
            prev = torch.cat([torch.full((B, 1), self.vocab, dtype=x.dtype), x[:, :-1]], dim=1)
            e = e + self.table(mix_bigram(prev, x, self.table_rows))
        for o, t in self.tables.items():
            e = e + t(mix_ngram(x, int(o), self.table_rows, self.vocab))
        return e + self.pos(torch.arange(T))

    def hidden(self, x):
        h = self.embed(x)
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h)

    def nll(self, x, y):
        """Per-token negative log-likelihood in nats, shape (B, T)."""
        h = self.hidden(x)
        if self.coords:
            gy, my = y // self.M, y % self.M
            nll_g = F.cross_entropy(self.head_g(h).transpose(1, 2), gy, reduction="none")
            lm = self.head_m(torch.cat([h, self.emb_g(gy)], -1)).masked_fill(~self.valid[gy], self.INVALID)
            nll_m = F.cross_entropy(lm.transpose(1, 2), my, reduction="none")
            return nll_g + nll_m
        logits = h @ self.emb.weight.T
        # log_softmax + gather is the same loss as cross_entropy on the transposed logits and
        # about 1.7x faster on CPU (the transpose makes the logits non-contiguous). The runs up
        # to and including the side-by-side used the slower form for models without n-gram tables.
        logp_net = F.log_softmax(logits, -1).gather(-1, y[..., None])[..., 0]
        if not self.ngrams:
            return -logp_net
        p_parts, confs = [logp_net.exp()], []
        # In training the tables were counted on this very text, so the current occurrence is
        # removed from the count before the model sees it (leave-one-out); otherwise a context
        # seen once "predicts" its follower perfectly and the model learns to over-trust counts.
        loo = 1.0 if self.training else 0.0
        for table in self.ngrams:
            ids, counts, totals = table.features(x.numpy())
            c_target = (counts * (ids == y[..., None])).sum(-1)
            kept = counts.sum(-1)
            c_target = (c_target - loo * (c_target > 0)).clamp_min(0.0)
            kept = (kept - loo * (totals > 0)).clamp_min(0.0)
            p_parts.append(torch.where(kept > 0, c_target / kept.clamp_min(1.0), torch.zeros_like(kept)))
            confs.extend(table_features(counts, totals, loo, ids, y))
        mix = F.softmax(self.mix_head(torch.cat([h] + [c[..., None] for c in confs], -1)), -1)
        p = (mix * torch.stack(p_parts, -1)).sum(-1)
        return -torch.log(p.clamp_min(1e-9))

    def mixed_logits(self, h_last, x_last_np):
        """Full next-token log-probabilities at the last position, with the n-gram mixture."""
        logits = h_last @ self.emb.weight.T
        if not self.ngrams:
            return logits
        p = [F.softmax(logits, -1)]
        confs = []
        for table in self.ngrams:
            ids, counts, totals = table.features(x_last_np)
            dense = torch.zeros_like(p[0])
            ok = ids[:, -1] >= 0
            probs = counts[:, -1] / counts[:, -1].sum(-1, keepdim=True).clamp_min(1.0)
            dense.scatter_add_(-1, ids[:, -1].clamp_min(0), probs * ok)
            p.append(dense)
            confs.extend(f[:, -1] for f in table_features(counts, totals, 0.0))
        mix = F.softmax(self.mix_head(torch.cat([h_last] + [c[..., None] for c in confs], -1)), -1)
        return torch.log((mix[..., None] * torch.stack(p, 1)).sum(1).clamp_min(1e-9))

    @torch.no_grad()
    def sample_next(self, x, gen: torch.Generator, temperature: float, top_k: int) -> int:
        h = self.hidden(x)[:, -1]

        def pick(logits):
            logits = logits / temperature
            if top_k:
                kth = torch.topk(logits, top_k).values[..., -1, None]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            return torch.multinomial(F.softmax(logits, -1), 1, generator=gen).item()

        if self.coords:
            g = pick(self.head_g(h).masked_fill(~self.valid_group, float("-inf")))
            lm = self.head_m(torch.cat([h, self.emb_g(torch.tensor([g]))], -1))
            m = pick(lm.masked_fill(~self.valid[g], float("-inf")))
            return g * self.M + m
        return pick(self.mixed_logits(h, x.numpy()))


# --------------------------------------------------------------------------- evaluation

@torch.no_grad()
def total_bits(model: LM, tokens: np.ndarray, batch: int = 64) -> tuple[float, int]:
    """Sum of -log2 p over every token but the first, with the context reset every ctx tokens."""
    model.eval()
    ctx, n = model.ctx, len(tokens)
    starts = list(range(0, n - 1, ctx))
    bits, predicted = 0.0, 0
    for i in range(0, len(starts), batch):
        xs, ys, counts = [], [], []
        for s in starts[i:i + batch]:
            end = min(s + ctx + 1, n)
            take = end - s - 1          # targets this window is responsible for
            if take <= 0:
                continue
            tail = True                 # count the last `take` targets of the row
            if end - s < ctx + 1:
                if n >= ctx + 1:        # final short window: shift back, count only the new targets
                    s, end = n - ctx - 1, n
                else:                   # stream shorter than one window: pad, count the head
                    tail = False
            w = tokens[s:end]
            if len(w) < ctx + 1:
                w = np.pad(w, (0, ctx + 1 - len(w)))
            xs.append(w[:-1]); ys.append(w[1:]); counts.append((take, tail))
        if not xs:
            continue
        x = torch.from_numpy(np.stack(xs)); y = torch.from_numpy(np.stack(ys))
        nll = model.nll(x, y)
        for row, (take, tail) in zip(nll, counts):
            part = row[-take:] if tail else row[:take]
            bits += float(part.sum()) / LN2
            predicted += take
    model.train()
    return bits, predicted


@torch.no_grad()
def token_bits(model: LM, tokens: np.ndarray, batch: int = 64) -> np.ndarray:
    """-log2 p for every token but the first (context reset every ctx tokens), aligned to tokens[1:]."""
    model.eval()
    ctx, n = model.ctx, len(tokens)
    out = np.zeros(n - 1, dtype=np.float64)
    starts = list(range(0, n - 1, ctx))
    for i in range(0, len(starts), batch):
        xs, ys, spans = [], [], []
        for s in starts[i:i + batch]:
            end = min(s + ctx + 1, n)
            take = end - s - 1
            tail = True
            if end - s < ctx + 1:          # same window handling as total_bits
                if n >= ctx + 1:
                    s, end = n - ctx - 1, n
                else:
                    tail = False
            w = tokens[s:end]
            if len(w) < ctx + 1:
                w = np.pad(w, (0, ctx + 1 - len(w)))
            xs.append(w[:-1]); ys.append(w[1:]); spans.append((s, take, tail))
        nll = model.nll(torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))).numpy() / LN2
        for row, (s, take, tail) in zip(nll, spans):
            if tail:
                out[s + ctx - take:s + ctx] = row[-take:]
            else:
                out[s:s + take] = row[:take]
    model.train()
    return out


def class_bits(model: LM, enc: LE.Encoding, tokens: np.ndarray, lens: np.ndarray, train_counts: np.ndarray | None) -> dict:
    """Split a file's bits by what kind of token is being predicted: frequency band in the
    training text (top 100 / top 1,000 / top 10,000 / rarer / never seen) and token type
    (word, byte fallback, other). Each entry: share of bytes, share of bits, bits per byte."""
    bits = token_bits(model, tokens)
    y, ly = tokens[1:], lens[1:]
    groups: dict[str, np.ndarray] = {}
    if train_counts is not None and not enc.groups:
        rank = np.empty(len(train_counts), dtype=np.int64)
        rank[np.argsort(-train_counts)] = np.arange(len(train_counts))
        r = rank[y]
        groups.update({"freq_top100": r < 100, "freq_101_1000": (r >= 100) & (r < 1000),
                       "freq_1001_10000": (r >= 1000) & (r < 10000), "freq_rarer": (r >= 10000) & (train_counts[y] > 0),
                       "freq_unseen_in_training": train_counts[y] == 0})
    if not enc.groups:
        is_byte = y < 256
        kind = np.array([b.strip(b" ").isalpha() if b else False for b in [enc_bytes(enc, t) for t in range(enc.vocab)]])
        is_word = kind[y]
        groups.update({"type_byte_fallback": is_byte, "type_word": is_word & ~is_byte, "type_other": ~is_word & ~is_byte})
    total_b, total_bits = int(ly.sum()), float(bits.sum())
    return {name: {"byte_share": round(float(ly[m].sum()) / max(1, total_b), 4),
                   "bits_share": round(float(bits[m].sum()) / max(1e-9, total_bits), 4),
                   "bits_per_byte": round(float(bits[m].sum()) / max(1, float(ly[m].sum())), 4)}
            for name, m in groups.items()}


def enc_bytes(enc: LE.Encoding, t: int) -> bytes:
    if hasattr(enc, "id_bytes"):
        return enc.id_bytes[t]
    if hasattr(enc, "d"):
        return enc.d.id_bytes[t]
    return bytes([t]) if t < 256 else b""


def eval_file(model: LM, enc: LE.Encoding, path: str, work: Path, train_counts=None, classes: bool = False) -> dict:
    tokens, lens = enc.encode_file(path, work)
    nbytes = os.path.getsize(path)
    bits, predicted = total_bits(model, tokens)
    row = {"bytes": nbytes, "tokens": int(len(tokens)), "ids_per_kb": round(1000 * len(tokens) / nbytes, 1),
           "bits_per_byte": round(bits / nbytes, 4), "bits_per_token": round(bits / max(1, predicted), 4)}
    if isinstance(enc, LE.HashCodes):
        row["fidelity"] = enc.fidelity(Path(path).read_bytes())
    if classes:
        row["by_class"] = class_bits(model, enc, tokens, lens, train_counts)
    return row


# --------------------------------------------------------------------------- training

def run(enc: LE.Encoding, train_tokens: np.ndarray, train_lens: np.ndarray, eval_sets: dict[str, list[str]],
        work: Path, budget_s: float, width=128, layers=4, heads=4, ctx=256, batch=16, lr=1e-3,
        table_rows=0, seed=1, eval_every_s=120.0, quick_bytes=150_000, prompt=b"We went to the park",
        gen_tokens=200, log=print, save_path=None, table_orders=(2,), ngram_orders=(), ngram_topk=16,
        valid_share=0.02, patience=0, min_delta=0.002) -> dict:
    """Train for up to ``budget_s`` seconds of training time.

    The last ``valid_share`` of the training stream is held back as the validation split: it
    is never trained on, the n-gram tables are not counted on it, and the learning curve and
    early stopping read from it. The held-out and out-of-domain files in ``eval_sets`` are
    scored once, at the end. With ``patience`` > 0, training stops when the validation bits
    per byte have not improved by more than ``min_delta`` for ``patience`` evaluations in a row.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n_train = int(len(train_tokens) * (1 - valid_share)) if valid_share > 0 else len(train_tokens)
    valid_tokens, valid_lens = train_tokens[n_train:], train_lens[n_train:]
    train_tokens, train_lens = train_tokens[:n_train], train_lens[:n_train]
    valid = np.fromiter(enc.reverse.keys(), dtype=np.int64, count=len(enc.reverse)) if isinstance(enc, LE.HashCodes) else None
    ngrams, ngram_info = [], []
    for order in ngram_orders:
        t0 = time.perf_counter()
        table = NgramTable(train_tokens, order, ngram_topk)
        ngrams.append(table)
        ngram_info.append({"order": order, "contexts": table.contexts, "topk": ngram_topk,
                           "entries": int((table.ids >= 0).sum()), "build_s": round(time.perf_counter() - t0, 1)})
        log(f"[{enc.name}] {order}-gram table: {table.contexts:,} contexts, {ngram_info[-1]['entries']:,} entries ({ngram_info[-1]['build_s']}s)")
    model = LM(enc, width, layers, heads, ctx, table_rows, valid, table_orders, ngrams)
    dense = [p for n, p in model.named_parameters() if not n.startswith("table")]
    sparse = [p for n, p in model.named_parameters() if n.startswith("table")]
    decay = [p for p in dense if p.dim() >= 2]
    no_decay = [p for p in dense if p.dim() < 2]
    opts = [torch.optim.AdamW([{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
                              lr=lr, betas=(0.9, 0.95))]
    if sparse:
        opts.append(torch.optim.SparseAdam(sparse, lr=lr))
    counts = model.param_counts()
    log(f"[{enc.name}] params {counts}  train tokens {len(train_tokens):,}  budget {budget_s:.0f}s")

    # Quick validation slice: the first ~quick_bytes bytes of the validation split (or, with no
    # validation split, of the first held-out file, as the earliest runs did).
    if len(valid_tokens) > ctx + 1:
        q_tokens, q_lens = valid_tokens, valid_lens
    else:
        q_tokens, q_lens = enc.encode_file(eval_sets["heldout"][0], work)
    n_quick = min(len(q_tokens), int(np.searchsorted(np.cumsum(q_lens), quick_bytes)) + 1)
    q_tokens, q_bytes = q_tokens[:n_quick], int(q_lens[:n_quick].sum())

    def lr_at(frac, step):
        warm = min(1.0, step / 100)
        return lr * warm * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, frac))))

    n = len(train_tokens)
    offsets_base = np.arange(ctx + 1)
    train_time, step, tokens_seen, bytes_seen, ema = 0.0, 0, 0, 0, None
    history, next_eval = [], eval_every_s
    best_quick, bad_evals, stop_reason = float("inf"), 0, "budget"
    model.train()
    while train_time < budget_s:
        t0 = time.perf_counter()
        idx = rng.integers(0, n - ctx - 1, size=batch)
        win = train_tokens[idx[:, None] + offsets_base]
        x = torch.from_numpy(win[:, :-1]); y = torch.from_numpy(win[:, 1:])
        cur_lr = lr_at(train_time / budget_s, step)
        for o in opts:
            for g in o.param_groups:
                g["lr"] = cur_lr
        loss = model.nll(x, y).mean()
        for o in opts:
            o.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense, 1.0)
        for o in opts:
            o.step()
        step += 1
        tokens_seen += batch * ctx
        bytes_seen += int(train_lens[idx[:, None] + offsets_base[1:]].sum())
        bpt = loss.item() / LN2
        ema = bpt if ema is None else 0.98 * ema + 0.02 * bpt
        train_time += time.perf_counter() - t0
        if train_time >= next_eval or train_time >= budget_s:
            qbits, _ = total_bits(model, q_tokens)
            row = {"train_s": round(train_time, 1), "step": step, "tokens_seen": tokens_seen, "bytes_seen": bytes_seen,
                   "train_bits_per_token": round(ema, 4), "quick_bits_per_byte": round(qbits / q_bytes, 4), "lr": cur_lr}
            history.append(row)
            log(f"[{enc.name}] {row['train_s']:7.1f}s step {step:6d} seen {bytes_seen/1e6:6.2f} MB  "
                f"train {ema:.3f} b/tok  valid {row['quick_bits_per_byte']:.4f} b/byte")
            next_eval += eval_every_s
            if row["quick_bits_per_byte"] < best_quick - min_delta:
                best_quick, bad_evals = row["quick_bits_per_byte"], 0
            else:
                bad_evals += 1
                if patience and bad_evals >= patience:
                    stop_reason = f"plateau: no improvement over {min_delta} bits/byte in {patience} evaluations"
                    log(f"[{enc.name}] stopping early at {train_time:.0f}s: {stop_reason}")
                    break

    result = {"encoding": enc.describe(), "model": {"width": width, "layers": layers, "heads": heads, "ctx": ctx,
                                                      "batch": batch, "peak_lr": lr, "table_rows": table_rows,
                                                      "table_orders": list(model.table_orders), "params": counts,
                                                      "ngram_tables": ngram_info},
              "training": {"budget_s": budget_s, "train_s": round(train_time, 1), "steps": step, "tokens_seen": tokens_seen,
                           "bytes_seen": bytes_seen, "tokens_per_s": round(tokens_seen / train_time, 1),
                           "bytes_per_s": round(bytes_seen / train_time, 1), "final_train_bits_per_token": round(ema, 4),
                           "train_tokens": int(n), "valid_tokens": int(len(valid_tokens)), "valid_bytes": int(valid_lens.sum()),
                           "patience": patience, "min_delta": min_delta, "stop_reason": stop_reason,
                           "stopped_early": stop_reason != "budget", "history": history},
              "eval": {}, "seed": seed}
    if len(valid_tokens) > ctx + 1:
        vbits, _ = total_bits(model, valid_tokens)
        result["eval"]["validation"] = {"bytes": int(valid_lens.sum()), "tokens": int(len(valid_tokens)),
                                        "bits_per_byte": round(vbits / max(1, int(valid_lens.sum())), 4)}
        log(f"[{enc.name}] validation split: {result['eval']['validation']}")

    train_counts = np.bincount(train_tokens, minlength=enc.vocab) if not enc.groups else None
    for split, paths in eval_sets.items():
        rows = {Path(p).stem: eval_file(model, enc, p, work, train_counts, classes=(split == "heldout")) for p in paths}
        result["eval"][split] = rows
        tb = sum(r["bits_per_byte"] * r["bytes"] for r in rows.values())
        nb = sum(r["bytes"] for r in rows.values())
        nt = sum(r["tokens"] for r in rows.values())
        result["eval"][f"{split}_overall"] = {"bytes": nb, "tokens": nt, "ids_per_kb": round(1000 * nt / nb, 1),
                                              "bits_per_byte": round(tb / nb, 4)}
        log(f"[{enc.name}] {split}: {result['eval'][f'{split}_overall']}")

    # The model talks back through the same encoder: encode the prompt, generate, decode.
    gen = torch.Generator().manual_seed(seed)
    p_tokens, _ = enc.encode(prompt)
    seq = list(int(t) for t in p_tokens)
    model.eval()
    t0 = time.perf_counter()
    for _ in range(gen_tokens):
        x = torch.tensor([seq[-ctx:]])
        seq.append(model.sample_next(x, gen, 0.8, 40))
    gen_s = time.perf_counter() - t0
    out = enc.decode(seq[len(p_tokens):])
    result["generation"] = {"prompt": prompt.decode("utf-8", "replace"), "tokens": gen_tokens, "bytes": len(out),
                            "seconds": round(gen_s, 2), "tokens_per_s": round(gen_tokens / gen_s, 1),
                            "bytes_per_s": round(len(out) / gen_s, 1), "note": "sliding context, no KV cache, CPU",
                            "sample": out[:600].decode("utf-8", "replace")}
    log(f"[{enc.name}] generated {len(out)} bytes in {gen_s:.1f}s ({len(out)/gen_s:.0f} B/s): {out[:160]!r}")
    result["machine"] = {"python": platform.python_version(), "torch": torch.__version__, "threads": torch.get_num_threads(),
                         "cpu": platform.processor() or platform.machine(), "cores": os.cpu_count()}
    if save_path and not ngrams:
        # Weights plus what chat.py needs to rebuild the encoder; a hashed table is left out when
        # it is large, since it is for the speed and capacity measurement, not for talking.
        state = {k: v for k, v in model.state_dict().items() if not (k.startswith("table") and table_rows > 1 << 16)}
        torch.save({"encoding": enc.describe(), "model": result["model"], "state_dict": state}, save_path)
    return result


# --------------------------------------------------------------------------- CLI

def eval_sets_from(data: str) -> dict[str, list[str]]:
    return {"heldout": sorted(glob.glob(f"{data}/heldout/*.txt")), "ood": sorted(glob.glob(f"{data}/ood/*.txt"))}


def load_training(enc: LE.Encoding, data: str, work: Path) -> tuple[np.ndarray, np.ndarray]:
    concat = work / "train_concat.txt"
    if not concat.exists():
        concat.write_bytes(b"".join(Path(p).read_bytes() for p in sorted(glob.glob(f"{data}/train/*.txt"))))
    return enc.encode_file(str(concat), work)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--encoding", choices=["bytes", "v0", "hash"], required=True)
    ap.add_argument("--dict"); ap.add_argument("--native"); ap.add_argument("--hash-map")
    ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=600); ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--ctx", type=int, default=256); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--table-rows", type=int, default=0)
    ap.add_argument("--table-orders", default="2", help="comma-separated n-gram orders for the hashed tables")
    ap.add_argument("--ngram-orders", default="", help="comma-separated orders of exact n-gram output tables, e.g. 2,3")
    ap.add_argument("--ngram-topk", type=int, default=16)
    ap.add_argument("--valid-share", type=float, default=0.02, help="share of the training stream held back for validation")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many evaluations without improvement (0: never)")
    ap.add_argument("--min-delta", type=float, default=0.002, help="improvement in validation bits/byte that counts")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--eval-every", type=float, default=120)
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args(argv)
    if a.threads:
        torch.set_num_threads(a.threads)
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    if a.encoding == "bytes":
        enc = LE.Bytes()
    elif a.encoding == "v0":
        enc = LE.V0Dict(a.dict, a.native)
    else:
        enc = LE.HashCodes.load(a.hash_map)
    tokens, lens = load_training(enc, a.data, work)
    result = run(enc, tokens, lens, eval_sets_from(a.data), work, a.budget, a.width, a.layers, a.heads, a.ctx,
                 a.batch, a.lr, a.table_rows, a.seed, a.eval_every,
                 table_orders=tuple(int(o) for o in a.table_orders.split(",")),
                 ngram_orders=tuple(int(o) for o in a.ngram_orders.split(",") if o), ngram_topk=a.ngram_topk,
                 valid_share=a.valid_share, patience=a.patience, min_delta=a.min_delta)
    result["args"] = vars(a)
    Path(a.out).write_text(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
