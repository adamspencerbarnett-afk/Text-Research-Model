"""The judge of search engine v2 (part 2, docs/SEARCH_ENGINE_V2.md): a small model of our own that reads a question and a
memory item together and scores "does this item answer the question?". Built from the 5M core's first blocks (our own
weights, no outside model), trained on data_chat22/search_v2/judge_train.jsonl (pairs labelled from what the real search
returned: the right item vs look-alikes, outdated values, same-topic stored facts, near misses), tested on judge_test.jsonl.
The bar (threshold) is set on held-out pairs for high precision, so look-alikes rarely pass.

    python -m chat22.judge [--layers 2] [--steps 0 (= one pass)] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import encoder as E
from .blocks import Block, RMSNorm

ROOT = Path(__file__).resolve().parents[1]
MAXLEN = 160


class Judge(nn.Module):
    def __init__(self, vocab, width, heads, layers):
        super().__init__()
        self.emb = nn.Embedding(vocab, width)
        self.blocks = nn.ModuleList(Block(width, heads) for _ in range(layers))
        self.ln_f = RMSNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x, lengths):
        h = self.emb(x)
        for b in self.blocks:
            h = b(h)
        h = self.ln_f(h)
        last = h[torch.arange(x.shape[0], device=x.device), lengths - 1]      # the state after the whole pair
        return self.head(last)[:, 0]


def encode_pair(enc, q: str, item: str) -> np.ndarray:
    c = enc.encode(b"Question: " + q.encode("utf-8", "replace")[:300] + b"\nMemory: " + item.encode("utf-8", "replace")[:600] + b"\nAnswers it:")[0]
    return c[-MAXLEN:]


def batches(enc, rows, bs, rng, dev):
    idx = rng.permutation(len(rows))
    for s in range(0, len(idx) - bs + 1, bs):
        chunk = [rows[i] for i in idx[s:s + bs]]
        codes = [encode_pair(enc, r["q"], r["item"]) for r in chunk]
        L = max(len(c) for c in codes)
        x = np.zeros((bs, L), np.int64); n = np.zeros(bs, np.int64)
        for j, c in enumerate(codes):
            x[j, :len(c)] = c; n[j] = len(c)
        yield torch.from_numpy(x).to(dev), torch.from_numpy(n).to(dev), torch.tensor([r["label"] for r in chunk], dtype=torch.float32, device=dev)


@torch.no_grad()
def scores(model, enc, rows, dev, bs=256):
    model.eval(); out = []
    for s in range(0, len(rows), bs):
        chunk = rows[s:s + bs]
        codes = [encode_pair(enc, r["q"], r["item"]) for r in chunk]
        L = max(len(c) for c in codes)
        x = np.zeros((len(chunk), L), np.int64); n = np.zeros(len(chunk), np.int64)
        for j, c in enumerate(codes):
            x[j, :len(c)] = c; n[j] = len(c)
        out.append(torch.sigmoid(model(torch.from_numpy(x).to(dev), torch.from_numpy(n).to(dev))).float().cpu())
    model.train()
    return torch.cat(out).numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=int, default=2); ap.add_argument("--steps", type=int, default=0); ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--device", default="cuda"); ap.add_argument("--out", default="runs/chat22/judge.pt")
    ap.add_argument("--max-train", type=int, default=0)
    a = ap.parse_args()
    dev = torch.device(a.device if torch.cuda.is_available() or a.device == "cpu" else "cpu")
    rng = np.random.default_rng(1); torch.manual_seed(1)
    core = torch.load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
    enc = E.load(ROOT / core["encoding"]["dictionary"])
    d = ROOT / "data_chat22" / "search_v2"
    train = [json.loads(l) for l in open(d / "judge_train.jsonl", encoding="utf-8")]
    test = [json.loads(l) for l in open(d / "judge_test.jsonl", encoding="utf-8")]
    if a.max_train:
        train = train[:a.max_train]
    sh = core["model"]
    model = Judge(enc.vocab, sh["width"], sh["heads"], a.layers)
    sd = {k: v for k, v in core["state_dict"].items() if k.startswith("emb.") or k.startswith("ln_f.") or
          (k.startswith("blocks.") and int(k.split(".")[1]) < a.layers)}
    missing = model.load_state_dict(sd, strict=False)
    model.to(dev).train()
    pos = sum(r["label"] for r in train) / len(train)
    steps = a.steps or len(train) // a.batch
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"judge: {a.layers} layers from the 5M core, {sum(p.numel() for p in model.parameters()):,} parameters; {len(train):,} train / "
          f"{len(test):,} test pairs; positives {pos:.1%}; {steps:,} steps of {a.batch}", flush=True)
    t0 = time.time(); step = 0; w = torch.tensor((1 - pos) / max(pos, 1e-6), device=dev)
    while step < steps:
        for x, n, y in batches(enc, train, a.batch, rng, dev):
            for g in opt.param_groups:
                g["lr"] = a.lr * min(1.0, (step + 1) / 300) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
                logit = model(x, n)
            loss = F.binary_cross_entropy_with_logits(logit.float(), y, pos_weight=w)
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            step += 1
            if step % 1000 == 0:
                print(f"step {step}/{steps} loss {float(loss):.4f} ({time.time() - t0:.0f} s)", flush=True)
            if step >= steps:
                break
    s = scores(model, enc, test, dev); y = np.array([r["label"] for r in test])
    # the bar: the lowest threshold whose precision on held-out pairs is at least 95% (look-alikes rarely pass)
    best = None
    for th in np.linspace(0.05, 0.95, 91):
        p = s >= th; tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        if prec >= 0.95 and (best is None or rec > best["recall"]):
            best = {"threshold": round(float(th), 2), "precision": round(prec, 3), "recall": round(rec, 3)}
    order = np.argsort(-s); auc_num = 0.0; npos = int(y.sum()); nneg = len(y) - npos; seen_neg = 0
    for i in order[::-1]:
        if y[i] == 0:
            seen_neg += 1
        else:
            auc_num += seen_neg
    auc = auc_num / max(1, npos * nneg)
    by = {}
    for r, sc in zip(test, s):
        k = r["from"].split(":")[0] + ":" + ("right" if r["label"] else "wrong")
        b = by.setdefault(k, [0, 0]); b[0] += int((sc >= (best or {"threshold": 0.5})["threshold"]) == bool(r["label"])); b[1] += 1
    res = {"layers": a.layers, "train_pairs": len(train), "test_pairs": len(test), "steps": steps, "seconds": round(time.time() - t0),
           "auc": round(auc, 4), "bar": best, "correct_by_source_at_bar": {k: f"{v[0]}/{v[1]}" for k, v in by.items()}}
    torch.save({"state_dict": {k: v.cpu() for k, v in model.state_dict().items()}, "config": {"layers": a.layers, "width": sh["width"],
                "heads": sh["heads"], "vocab": enc.vocab, "dict": core["encoding"]["dictionary"], "maxlen": MAXLEN,
                "threshold": (best or {"threshold": 0.5})["threshold"]}, "result": res}, ROOT / a.out)
    (ROOT / "results" / "chat22_judge.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
