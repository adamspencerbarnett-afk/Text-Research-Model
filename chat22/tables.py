"""Exact n-gram follower tables mixed into the output by the trust head: NgramTable and table_features copied
unchanged from compiler/experiments/train_lm.py, plus save/load so a model's tables travel with its checkpoint
(the old engine rebuilt them from whatever data path it was given)."""
from __future__ import annotations

import numpy as np
import torch


class NgramTable:
    """Exact n-gram follower lists for a count-based component at the output.

    For every context of ``order - 1`` tokens seen in training, keeps the ``topk`` most frequent
    next tokens with their frequencies (renormalised over the kept followers) and the context's
    total count. The followers are stored ragged (``offsets`` into ``ids`` and ``counts``, most
    contexts have far fewer than ``topk``), so a table of a 250 MB corpus fits on the GPU, and
    the lookup is a binary search over the sorted context keys plus one gather, on whatever
    device the batch is on. Flat vocabularies only; tokens must be below 2^21.
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
        rank = np.arange(len(ctx)) - np.repeat(first, sizes)
        keep = rank < topk                       # sorted by context then by count: kept rows stay contiguous
        self.ids = nxt[keep].astype(np.int32)
        self.counts = counts[keep].astype(np.float32)
        self.offsets = np.concatenate([[0], np.cumsum(np.minimum(sizes, topk))]).astype(np.int64)
        self.totals = totals.astype(np.float32)
        self.contexts, self.entries = len(self.keys), int(keep.sum())
        self._on: dict = {}

    def row(self, context: int) -> tuple[list[int], list[float]]:
        """Kept followers and counts of one context row (for inspection and tests)."""
        a, b = self.offsets[context], self.offsets[context + 1]
        return self.ids[a:b].tolist(), self.counts[a:b].tolist()

    def tensors(self, device):
        """The table as tensors on ``device`` (moved once, then cached)."""
        key = str(device)
        if key not in self._on:
            self._on[key] = tuple(torch.from_numpy(a).to(device) for a in (self.keys, self.offsets, self.ids, self.counts, self.totals))
        return self._on[key]

    def features(self, x, device=None):
        """For a (B, T) token array or tensor: follower ids (B, T, K), their counts (B, T, K)
        and the context's total count (B, T); -1 / 0 where the context is unseen or incomplete.
        Computed on ``device`` (default: the tensor's device, or the CPU for an array)."""
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.ascontiguousarray(x))
        device = device if device is not None else x.device
        x = x.to(device, torch.int64)
        keys, offsets, ids, counts, totals = self.tensors(device)
        B, T = x.shape
        key = torch.zeros((B, T), dtype=torch.int64, device=device)
        for j in range(self.k):
            shift = self.k - 1 - j
            col = torch.zeros((B, T), dtype=torch.int64, device=device)
            col[:, shift:] = x[:, :T - shift]
            key = (key << 21) + col
        pos = torch.searchsorted(keys, key).clamp_max(len(keys) - 1)
        found = keys[pos] == key
        found[:, :self.k - 1] = False
        start, n = offsets[pos], offsets[pos + 1] - offsets[pos]
        ar = torch.arange(self.topk, device=device)
        valid = found[..., None] & (ar < n[..., None])
        at = (start[..., None] + ar).clamp_max(len(ids) - 1)
        return (torch.where(valid, ids[at].long(), -1), torch.where(valid, counts[at], 0.0),
                torch.where(found, totals[pos], 0.0))


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


def save_tables(tables: list, path) -> None:
    np.savez(path, **{f"{i}_{n}": getattr(t, n) for i, t in enumerate(tables)
                      for n in ("keys", "offsets", "ids", "counts", "totals")},
             meta=np.array([[t.order, t.topk] for t in tables]))


def load_tables(path) -> list:
    z = np.load(path)
    out = []
    for i, (order, topk) in enumerate(z["meta"]):
        t = NgramTable.__new__(NgramTable)
        t.order, t.topk, t.k = int(order), int(topk), int(order) - 1
        for n in ("keys", "offsets", "ids", "counts", "totals"):
            setattr(t, n, z[f"{i}_{n}"])
        t.contexts, t.entries, t._on = len(t.keys), len(t.ids), {}
        out.append(t)
    return out
