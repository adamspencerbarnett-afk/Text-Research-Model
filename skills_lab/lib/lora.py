"""LoRA adapters for the Transformer blocks (14 October 2026; docs/MULTITURN_PLAN.md arm L).

    lora.apply(model, rank=16)      # every nn.Linear inside model.blocks gets a trainable low-rank update;
                                    # the original weights are frozen
    lora.trainable(model)           # the adapter parameters
    lora.merge(model)               # fold each update into its weight and drop the adapters: the model is a
                                    # plain model again (same state_dict keys), so the engine and the CUDA-graph
                                    # decoder serve it unchanged

An adapted layer computes  W x + (alpha / rank) * B (A x),  with A small random and B zero, so training starts
exactly at the frozen model. A second skill is trained by merging the first adapter and applying a new one,
which is the same as stacking a new adapter on a frozen first one.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank, self.scale = rank, alpha / rank
        dev, dt = base.weight.device, base.weight.dtype
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=dev, dtype=dt))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=dev, dtype=dt))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))

    def forward(self, x):
        return self.base(x) + (x @ self.lora_a.t() @ self.lora_b.t()) * self.scale

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        self.base.weight += (self.lora_b @ self.lora_a) * self.scale
        for p in self.base.parameters():
            p.requires_grad_(True)
        return self.base


def apply(model: nn.Module, rank: int = 16, alpha: float | None = None) -> int:
    """Adapt every nn.Linear directly inside each block; freeze every block parameter. Returns the number of
    adapted layers. Embeddings, norms and heads are left as they are (the caller decides what else trains)."""
    alpha = float(rank if alpha is None else alpha)
    n = 0
    for block in model.blocks:
        for p in block.parameters():
            p.requires_grad_(False)
        for name, child in list(block.named_children()):
            if isinstance(child, nn.Linear):
                setattr(block, name, LoRALinear(child, rank, alpha)); n += 1
    return n


def trainable(model: nn.Module) -> list[nn.Parameter]:
    return [p for n, p in model.named_parameters() if "lora_" in n]


def merge(model: nn.Module) -> int:
    n = 0
    for block in model.blocks:
        for name, child in list(block.named_children()):
            if isinstance(child, LoRALinear):
                setattr(block, name, child.merged()); n += 1
    return n


def count(model: nn.Module) -> int:
    return sum(p.numel() for p in trainable(model))
