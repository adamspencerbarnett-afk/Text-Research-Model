"""Weight precision for a trained model: how low can the stored numbers go before the answers change.

    precision.apply(model, "fp32" | "bf16" | "int8" | "int4")

The weights are rounded to the named format and written back as float32, so the same code paths
run and only the information kept in the weights differs (the standard way to measure the accuracy
cost of a quantised deployment; speed is a separate question that needs real low-bit kernels).

  bf16  each weight rounded to bfloat16 (8 bits of mantissa)
  int8  symmetric per-row absmax, 8 bits (255 levels) for every matrix with two or more dimensions
  int4  symmetric absmax in groups of 32 along the last dimension, 4 bits (15 levels)

Vectors (norm scales, biases) stay in float32 in every mode, as deployments do.
"""
from __future__ import annotations

import torch

MODES = ("fp32", "bf16", "int8", "int4")


@torch.no_grad()
def _fake_quant(w: torch.Tensor, bits: int, group: int | None) -> torch.Tensor:
    qmax = 2 ** (bits - 1) - 1
    shape = w.shape
    x = w.float()
    if group:
        x = x.reshape(-1, group) if x.numel() % group == 0 else x.reshape(x.shape[0], -1)
    else:
        x = x.reshape(x.shape[0], -1)
    scale = x.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / qmax
    return ((x / scale).round().clamp(-qmax, qmax) * scale).reshape(shape).to(w.dtype)


@torch.no_grad()
def apply(model: torch.nn.Module, mode: str) -> dict:
    """Round the weights in place; returns what was touched."""
    if mode not in MODES:
        raise ValueError(f"precision must be one of {MODES}, not {mode!r}")
    touched, numel = 0, 0
    for name, p in model.named_parameters():
        if p.dim() < 2 or mode == "fp32":
            continue
        if mode == "bf16":
            p.copy_(p.to(torch.bfloat16).to(p.dtype))
        elif mode == "int8":
            p.copy_(_fake_quant(p, 8, None))
        else:
            p.copy_(_fake_quant(p, 4, 32))
        touched += 1; numel += p.numel()
    return {"precision": mode, "tensors": touched, "weights": numel}
