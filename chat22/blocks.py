"""The transformer block of the adopted core (RoPE + RMSNorm + SwiGLU), as in compiler/experiments/train_lm.py
ModernBlock, with one addition: ``rope_scale`` (position interpolation, an S42 arm; 1.0 = the trained positions).
Parameter names match the old block, so old checkpoints load unchanged."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

_ROPE_CACHE: dict = {}


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight, self.eps = nn.Parameter(torch.ones(d)), eps

    def forward(self, x):
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype) * self.weight


def rope(x, start: int = 0, scale: float = 1.0):
    """Rotate query/key dimension pairs by an angle growing with the absolute position (divided by ``scale``).
    x: (B, H, T, D)."""
    T, D = x.shape[-2], x.shape[-1]
    key = (start, T, D, x.device, x.dtype, scale)
    cs = _ROPE_CACHE.get(key)
    if cs is None:
        inv = 1.0 / (10000 ** (torch.arange(0, D, 2, device=x.device, dtype=torch.float32) / D))
        pos = torch.arange(start, start + T, device=x.device, dtype=torch.float32)
        ang = (pos / scale if scale != 1.0 else pos)[:, None] * inv[None, :]
        cs = (ang.cos()[None, None].to(x.dtype), ang.sin()[None, None].to(x.dtype))
        if len(_ROPE_CACHE) > 4096:
            _ROPE_CACHE.clear()
        _ROPE_CACHE[key] = cs
    c, s = cs
    a, b = x[..., 0::2], x[..., 1::2]
    return torch.stack((a * c - b * s, a * s + b * c), dim=-1).flatten(-2)


class Block(nn.Module):
    def __init__(self, d: int, heads: int, rope_scale: float = 1.0):
        super().__init__()
        self.heads, self.rope_scale = heads, rope_scale
        self.ln1, self.ln2 = RMSNorm(d), RMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        inner = (8 * d) // 3
        self.gate, self.up, self.down = nn.Linear(d, inner, bias=False), nn.Linear(d, inner, bias=False), nn.Linear(inner, d, bias=False)

    def _qkv(self, x, start):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(D, dim=2)
        q, k, v = (t.view(B, T, self.heads, D // self.heads).transpose(1, 2) for t in (q, k, v))
        return rope(q, start, self.rope_scale), rope(k, start, self.rope_scale), v

    def _ff(self, x):
        h = self.ln2(x)
        return x + self.down(F.silu(self.gate(h)) * self.up(h))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self._qkv(x, 0)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, D)
        return self._ff(x + self.proj(y))

    def forward_cached(self, x, past, start: int = 0):
        """New positions start .. start+T-1 against the cached keys/values of the earlier ones."""
        B, T, D = x.shape
        q, k, v = self._qkv(x, start)
        if past is not None:
            k, v = torch.cat([past[0], k], 2), torch.cat([past[1], v], 2)
        P = k.shape[2] - T
        if T == 1:
            y = F.scaled_dot_product_attention(q, k, v)
        else:
            mask = torch.ones(T, P + T, dtype=torch.bool, device=x.device).tril(P)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self._ff(x + self.proj(y.transpose(1, 2).reshape(B, T, D))), (k, v)
