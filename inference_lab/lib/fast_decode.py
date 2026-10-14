"""CUDA-graph answer step for the Q&A model: one generated code = one graph replay.

The eager answer loop launches about 300 small GPU kernels per generated code (6-10 blocks, the two
count tables, the copy head, the trust head), and on Windows each launch costs tens of microseconds,
so the network's arithmetic is a small part of the time. A CUDA graph records the whole step once
with fixed-size buffers and replays it with a single launch; this is how current inference servers
remove launch overhead.

Fixed-size buffers, for a window of C = ctx codes:
    win (1, C)            the codes so far          K, V (1, H, C, d) per layer  the attention cache
    hid (1, C, D)         final hidden states       ck   (1, C, D)    the copy head's keys
    span (1, C)           where the copy head may point (the retrieved exchange, optionally the question)
A step writes the new code at position p, runs every block for that one position against the whole
cache with positions > p masked, and computes the trust-head mixture exactly as V2LM.mixed_logits does.
Positions past p hold stale values and are always masked, so the result equals the eager loop up to
floating-point summation order.

Supported: legacy and modern blocks, copy head, count tables, similarity feature. Not supported (the
engine falls back to the eager loop): the evidence table, hashed input tables, factored vocabularies,
answers that would run past the window.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

import train_lm


def supported(model) -> bool:
    # fp32 only: bitsandbytes int8 kernels cannot be captured in a graph (a failed capture poisons the
    # CUDA context), and bf16 autocast inside the graph diverged from eager bf16 more than rounding explains
    return (getattr(model, "_kernels", "fp32") == "fp32" and torch.cuda.is_available() and next(model.parameters()).is_cuda
            and not getattr(model, "evidence", False) and not model.coords and model.table is None and not len(model.tables))


def _rope_at(x, p, inv):
    """Rotary positions for a single position given as a tensor (graph-safe form of train_lm.rope)."""
    ang = p.to(torch.float32)[:, None] * inv[None, :]
    c, s = ang.cos()[None, None].to(x.dtype), ang.sin()[None, None].to(x.dtype)
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class GraphStep:
    def __init__(self, model):
        self.m = m = model
        dev = next(m.parameters()).device
        self.dev, self.C, self.D = dev, m.ctx, m.emb.weight.shape[1]
        self.H = m.blocks[0].heads
        self.dh = self.D // self.H
        L, C, H, dh, D = len(m.blocks), self.C, self.H, self.dh, self.D
        z = lambda *s, dt=torch.float32: torch.zeros(*s, device=dev, dtype=dt)
        self.K, self.V = [z(1, H, C, dh) for _ in range(L)], [z(1, H, C, dh) for _ in range(L)]
        self.win, self.hid, self.ck = z(1, C, dt=torch.long), z(1, C, D), z(1, C, D)
        self.span, self.sim = z(1, C, dt=torch.bool), z(1)
        self.tok, self.p = z(1, 1, dt=torch.long), z(1, dt=torch.long)
        self.ar = torch.arange(C, device=dev)
        self.inv = (1.0 / (10000 ** (torch.arange(0, dh, 2, device=dev, dtype=torch.float32) / dh))) if m.block == "modern" else None
        self.out = None
        self.graph = None

    # ------------------------------------------------------------------ the step (captured)
    def _block(self, b, l, h):
        p = self.p
        mask = (self.ar <= p)[None, None, None, :]
        if self.m.block == "modern":
            q, k, v = b.qkv(b.ln1(h)).split(self.D, dim=2)
            q, k, v = (t.view(1, 1, self.H, self.dh).transpose(1, 2) for t in (q, k, v))
            q, k = _rope_at(q, p, self.inv), _rope_at(k, p, self.inv)
        else:
            q, k, v = b.qkv(b.ln1(h)).split(self.D, dim=2)
            q, k, v = (t.view(1, 1, self.H, self.dh).transpose(1, 2) for t in (q, k, v))
        self.K[l].index_copy_(2, p, k.to(self.K[l].dtype)); self.V[l].index_copy_(2, p, v.to(self.V[l].dtype))
        y = F.scaled_dot_product_attention(q, self.K[l].to(q.dtype), self.V[l].to(q.dtype), attn_mask=mask)
        x = h + b.proj(y.transpose(1, 2).reshape(1, 1, self.D))
        if self.m.block == "modern":
            return b._ff(x)
        return x + b.fc2(F.gelu(b.fc1(b.ln2(x))))

    def _logits(self, h_last):
        """V2LM.mixed_logits at position p, over the fixed buffers."""
        m, p, C, D = self.m, self.p, self.C, self.D
        logits = h_last @ m.emb.weight.T
        if not m.ngrams:
            return logits
        probs, confs = [F.softmax(logits, -1)], []
        for table in m.ngrams:
            idx = (p - table.k + torch.arange(table.k + 1, device=self.dev)).clamp_min(0)
            tail = self.win[:, idx]
            ids, counts, totals = table.features(tail, self.dev)
            dense = torch.zeros_like(probs[0])
            ok = ids[:, -1] >= 0
            pr = counts[:, -1] / counts[:, -1].sum(-1, keepdim=True).clamp_min(1.0)
            dense.scatter_add_(-1, ids[:, -1].clamp_min(0), pr * ok)
            probs.append(dense)
            confs.extend(f[:, -1] for f in train_lm.table_features(counts, totals, 0.0))
        if m.copy_head:
            src = self.span
            follow = src & torch.cat([src[:, 1:], torch.zeros_like(src[:, :1])], 1)
            x_next = torch.cat([self.win[:, 1:], torch.full_like(self.win[:, :1], -1)], 1)
            x_next = torch.where(self.ar[None] < p, x_next, torch.full_like(x_next, -1))
            allowed = follow & (self.ar[None] < p)
            scores = (m.copy_q(h_last)[:, None, :].float() @ self.ck.transpose(1, 2))[:, 0] / math.sqrt(D)
            scores = scores.masked_fill(~allowed, float("-inf"))
            attn = torch.softmax(scores, -1).nan_to_num(0.0)
            dense = torch.zeros_like(probs[0])
            dense.scatter_add_(-1, x_next.clamp_min(0), attn * allowed)
            probs.append(dense)
            confs += [attn.max(-1).values, -(attn * torch.log2(attn.clamp_min(1e-9))).sum(-1)]
        if m.sim_feature:
            confs.append(self.sim.to(h_last.dtype))
        mix = F.softmax(m.mix_head(torch.cat([h_last] + [c[..., None] for c in confs], -1)), -1)
        return torch.log((mix[..., None] * torch.stack(probs, 1)).sum(1).clamp_min(1e-9))

    def _step(self):
        m, p = self.m, self.p
        self.win.index_copy_(1, p, self.tok)
        h = m.emb(self.tok)
        if m.pos is not None:
            h = h + m.pos(p)[None]
        for l, b in enumerate(m.blocks):
            h = self._block(b, l, h)
        h = m.ln_f(h)
        self.hid.index_copy_(1, p, h.float())
        self.ck.index_copy_(1, p, (m.copy_k(h) if m.copy_head else h).float())
        return self._logits(h[:, 0])

    # ------------------------------------------------------------------ host side
    def _capture(self):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):          # warm-up on a side stream, as capture requires
            for _ in range(2):
                self._step()
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.out = self._step()

    @torch.no_grad()
    def prefill(self, seq_dev, span, sim: float):
        """Process the prompt eagerly (one forward), fill the buffers, return the first logits."""
        m, T = self.m, seq_dev.shape[0]
        h, cache = m.hidden_cached(seq_dev[None], 0, None)
        for l, (k, v) in enumerate(cache):
            self.K[l][:, :, :T] = k; self.V[l][:, :, :T] = v
        self.win.zero_(); self.win[0, :T] = seq_dev
        self.hid[:, :T] = h
        self.ck[:, :T] = m.copy_k(h) if m.copy_head else h
        self.span.zero_(); self.span[0, :T] = span
        self.sim.fill_(sim)
        m._keys = h
        sim_t = torch.tensor([sim], device=self.dev)
        return m.mixed_logits(h[:, -1], seq_dev[None], span[None], sim_t)

    @torch.no_grad()
    def step(self, code: int, pos: int):
        """Append one code at position pos and return the next-code log-probabilities (1, V)."""
        if self.graph is None:
            self.tok.fill_(code); self.p.fill_(pos)
            saved = [(k.clone(), v.clone()) for k, v in zip(self.K, self.V)], self.win.clone(), self.hid.clone(), self.ck.clone()
            self._capture()                 # capture runs the step: restore the buffers it touched
            for (k, v), (k0, v0) in zip(zip(self.K, self.V), saved[0]):
                k.copy_(k0); v.copy_(v0)
            self.win.copy_(saved[1]); self.hid.copy_(saved[2]); self.ck.copy_(saved[3])
        self.tok.fill_(code); self.p.fill_(pos)
        self.graph.replay()
        return self.out
