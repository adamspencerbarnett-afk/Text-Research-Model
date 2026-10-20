"""ChatLM: the core (tied embedding, Blocks), the copy head and the trust head, in one class used by training and
answering alike. Replaces m2_qa.V2LM + train_lm.LM for the chat path (V2LM.nll, train_chat.nll_and_aux,
V2LM.mixed_logits and V2LM.extra_parts merged; legacy branches removed). Parameter names match the old classes, so old
checkpoints load for the equivalence tests, and loading is strict: a missing or unexpected part is an error.

The output at each code is a mixture, weighted per code by the trust head, of:
  network     softmax of the tied output
  tables      one per n-gram table (counts of what followed this context in the table text)
  copy        the copy head: attention from the current state to earlier states whose FOLLOWER lies in the copyable
              part of the window (turns, memory block); the attended follower is the predicted code
Trust-head inputs: the hidden state, 3 features per table (log(1 + total), top share, entropy), and the copy head's
top weight and entropy."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import Block, RMSNorm
from .tables import table_features


class ChatLM(nn.Module):
    def __init__(self, vocab: int, width: int, layers: int, heads: int, ctx: int, tables: list, rope_scale: float = 1.0):
        super().__init__()
        self.vocab, self.width, self.ctx, self.tables = vocab, width, ctx, tables
        self.emb = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, heads, rope_scale) for _ in range(layers))
        self.ln_f = RMSNorm(width)
        self.copy_q, self.copy_k = nn.Linear(width, width, bias=False), nn.Linear(width, width, bias=False)
        self.mix_head = nn.Linear(width + 3 * len(tables) + 2, len(tables) + 2)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.normal_(self.emb.weight, std=0.02)
        nn.init.zeros_(self.mix_head.bias)

    # ------------------------------------------------------------------ loading
    CORE_PREFIXES = ("emb.", "blocks.", "ln_f.")

    def load(self, state: dict, core_only: bool = False) -> None:
        """Strict load. core_only: a core checkpoint (no copy/trust head yet), which loads only the core's parts and
        must supply all of them; the heads keep their fresh initialisation."""
        own = self.state_dict()
        if core_only:
            sd = {k: v for k, v in state.items() if k.startswith(self.CORE_PREFIXES)}
            missing = [k for k in own if k.startswith(self.CORE_PREFIXES) and k not in sd]
            if missing:
                raise KeyError(f"core checkpoint lacks {missing[:5]}")
            self.load_state_dict({**own, **sd}, strict=True)
        else:
            sd = {k: v for k, v in state.items() if k in own or not k.startswith("table")}
            sd.pop("pos.weight", None)
            self.load_state_dict(sd, strict=True)

    def param_counts(self) -> dict:
        n = lambda ms: sum(p.numel() for m in ms for p in m.parameters())
        core = n(self.blocks) + n([self.ln_f]); emb = n([self.emb]); heads = n([self.copy_q, self.copy_k, self.mix_head])
        return {"core": core, "embedding": emb, "heads": heads, "total": core + emb + heads}

    # ------------------------------------------------------------------ the network
    def hidden(self, x):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        return self.ln_f(h)

    def hidden_cached(self, x_new, start: int, cache: list | None):
        h = self.emb(x_new)
        cache = cache or [None] * len(self.blocks)
        new = []
        for b, past in zip(self.blocks, cache):
            h, kv = b.forward_cached(h, past, start)
            new.append(kv)
        return self.ln_f(h), new

    # ------------------------------------------------------------------ training / scoring
    def score(self, x, y, src, gold=None, point=None, loo=None):
        """Per-code nll of targets y (B, T) given inputs x (B, T) and the copyable mask src (B, T), plus the two
        intermediate targets: aux = -log(copy attention on gold positions whose follower is the target) where such
        a position exists; point = -log(copy attention on the point positions) elsewhere ('nothing here' / 'ask
        which'). ``loo`` (B,) in {0, 1}: leave-one-out for the table counts, 1 only for windows whose text is in
        the tables (default: 1 for all rows in training, 0 in eval, as the old code did).
        Returns dict(nll, aux, has, point, has_point)."""
        B, T = x.shape
        h = self.hidden(x)
        logits = h @ self.emb.weight.T
        p_parts = [F.log_softmax(logits, -1).gather(-1, y[..., None])[..., 0].exp()]
        if loo is None:
            loo = torch.full((B,), 1.0 if self.training else 0.0, device=x.device)
        lo = loo.to(x.device, torch.float32)[:, None]                       # (B, 1)
        confs = []
        for table in self.tables:
            ids, counts, totals = table.features(x)
            c_target = (counts * (ids == y[..., None])).sum(-1)
            kept = counts.sum(-1)
            c_target = (c_target - lo * (c_target > 0)).clamp_min(0.0)
            kept = (kept - lo * (totals > 0)).clamp_min(0.0)
            p_parts.append(torch.where(kept > 0, c_target / kept.clamp_min(1.0), torch.zeros_like(kept)))
            confs.extend(_table_feats_rows(counts, totals, lo, ids, y))
        # copy head: only positions before the end of the copyable prefix can be attended (keys sliced to it)
        K = int(src.sum(1).max()) if src.any() else 1
        follow = src & torch.cat([src[:, 1:], torch.zeros_like(src[:, :1])], 1)
        x_next = torch.cat([x[:, 1:], torch.full_like(x[:, :1], -1)], 1)
        follow, x_next = follow[:, :K], x_next[:, :K]
        causal = torch.ones(T, K, dtype=torch.bool, device=x.device).tril(-1)
        allowed = follow[:, None, :] & causal[None]
        scores = (self.copy_q(h) @ self.copy_k(h[:, :K]).transpose(1, 2)) / math.sqrt(self.width)
        attn = torch.softmax(scores.masked_fill(~allowed, float("-inf")), -1).nan_to_num(0.0)
        match = x_next[:, None, :] == y[..., None]
        p_parts.append((attn * match).sum(-1))
        confs += [attn.max(-1).values, -(attn * torch.log2(attn.clamp_min(1e-9))).sum(-1)]
        mix = F.softmax(self.mix_head(torch.cat([h] + [c[..., None] for c in confs], -1)), -1)
        out = {"nll": -torch.log((mix * torch.stack(p_parts, -1)).sum(-1).clamp_min(1e-9))}
        if gold is not None:
            hit = allowed & match & gold[:, None, :K]
            out["has"] = hit.any(-1)
            out["aux"] = -torch.log((attn * hit).sum(-1).float().clamp_min(1e-6))
            if point is not None:
                pt = allowed & point[:, None, :K]
                out["has_point"] = pt.any(-1) & ~out["has"]
                out["point"] = -torch.log((attn * pt).sum(-1).float().clamp_min(1e-6))
        return out

    # ------------------------------------------------------------------ answering: one decode step
    def next_logprobs(self, h_last, keys, x, copy_end: int):
        """Log-probabilities of the next code (B, vocab). h_last (B, D): state at the last position; keys (B, T, D):
        the states of every position so far (the copy head's keys); x (B, T): the codes so far; copy_end: length of
        the copyable prefix."""
        logits = h_last @ self.emb.weight.T
        p, confs = [F.softmax(logits, -1)], []
        for table in self.tables:
            ids, counts, totals = table.features(x[:, -(table.k + 1):], h_last.device)
            dense = torch.zeros_like(p[0])
            ok = ids[:, -1] >= 0
            probs = counts[:, -1] / counts[:, -1].sum(-1, keepdim=True).clamp_min(1.0)
            dense.scatter_add_(-1, ids[:, -1].clamp_min(0), probs * ok)
            p.append(dense)
            confs.extend(f[:, -1] for f in table_features(counts, totals, 0.0))
        B, T = x.shape
        K = min(copy_end, T)
        src = torch.zeros(B, T, dtype=torch.bool, device=x.device); src[:, :K] = True
        follow = (src & torch.cat([src[:, 1:], torch.zeros_like(src[:, :1])], 1))[:, :K]
        x_next = torch.cat([x[:, 1:], torch.full_like(x[:, :1], -1)], 1)[:, :K]
        if K > 0:
            sc = (self.copy_q(h_last)[:, None, :] @ self.copy_k(keys[:, :K]).transpose(1, 2))[:, 0] / math.sqrt(self.width)
            attn = torch.softmax(sc.masked_fill(~follow, float("-inf")), -1).nan_to_num(0.0)
            dense = torch.zeros_like(p[0])
            dense.scatter_add_(-1, x_next.clamp_min(0), attn * follow)
        else:
            attn = torch.zeros(B, 1, device=x.device); dense = torch.zeros_like(p[0])
        p.append(dense)
        confs += [attn.max(-1).values, -(attn * torch.log2(attn.clamp_min(1e-9))).sum(-1)]
        mix = F.softmax(self.mix_head(torch.cat([h_last] + [c[..., None] for c in confs], -1)), -1)
        return torch.log((mix[..., None] * torch.stack(p, 1)).sum(1).clamp_min(1e-9))


def _table_feats_rows(counts, totals, lo, ids, y):
    """table_features with a per-row leave-one-out (lo: (B, 1)); equal to the old function when lo is all 0 or 1."""
    c = (counts - lo[..., None] * (ids == y[..., None])).clamp_min(0.0)
    tot = (totals - lo * (totals > 0)).clamp_min(0.0)
    kept = c.sum(-1).clamp_min(1.0)
    pr = c / kept[..., None]
    top = pr.max(-1).values
    ent = -(pr * torch.log2(pr.clamp_min(1e-9))).sum(-1)
    return [torch.log1p(tot), top, ent]
