"""K1: a learned question key for the exchange memory.

The core's own state finds a reworded question only half the time (core_eval: 50% at rank 1
against 74% for the literal word key). This trains a small key head on top of the frozen core so
that two wordings of the same question land close together: questions are embedded as the mean of
the core's final hidden states, passed through a two-layer head to a 128-dimensional unit vector,
and trained with a contrastive loss (each question must pick its rewording among the batch) on
the Quora duplicate pairs. Tested on held-out pairs: alone, and combined with the literal key.

    python compiler/experiments/k1_question_key.py runs/core_D3s512/scale_5M_v0_8k_ng64.pt --device cuda

Writes results/k1_question_key_<core>.json and the head to runs/<core dir>/key_head.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import core_eval as C  # noqa: E402
import m2_qa  # noqa: E402


@torch.no_grad()
def pooled(model, enc, texts, device, batch=256):
    out = []
    for i in range(0, len(texts), batch):
        codes = [enc.encode(t)[0][: model.ctx] for t in texts[i:i + batch]]
        L = max(len(c) for c in codes)
        x = np.zeros((len(codes), L), dtype=np.int64); m = np.zeros((len(codes), L), dtype=np.float32)
        for r, c in enumerate(codes):
            x[r, :len(c)] = c; m[r, :len(c)] = 1
        h = model.hidden(torch.from_numpy(x).to(device)).float()
        mt = torch.from_numpy(m).to(device)[..., None]
        out.append(((h * mt).sum(1) / mt.sum(1).clamp_min(1)).cpu())
    return torch.cat(out)


def recall(S: np.ndarray):
    r = (S > S.diagonal()[:, None]).sum(1)
    return round(float((r == 0).mean()), 4), round(float((r < 10).mean()), 4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("core"); ap.add_argument("--device", default="cuda"); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--batch", type=int, default=1024); ap.add_argument("--temp", type=float, default=0.05)
    ap.add_argument("--n-val", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = random.Random(a.seed)
    native = ROOT / "data" / "cv0.exe"
    enc, model = C.load_core(a.core, str(native) if native.exists() else None, a.device)
    t0 = time.time()
    pairs = []
    for d in (ROOT / "data_core" / "src" / "qqp.txt").read_text(encoding="utf-8").split("\n\f\n"):
        if "\nIn other words: " in d:
            q1, q2 = d.split("\nIn other words: ", 1)
            pairs.append((q1.removeprefix("Question: ").strip(), q2.strip()))
    val = [json.loads(l) for l in open(ROOT / "data_core" / "eval" / "qqp_val.jsonl", encoding="utf-8")]
    rng.shuffle(val); val = val[: a.n_val]
    fmt = lambda q: ("Question: " + q).encode()
    tr1 = pooled(model, enc, [fmt(p) for p, _ in pairs], a.device); tr2 = pooled(model, enc, [fmt(q) for _, q in pairs], a.device)
    v1 = pooled(model, enc, [fmt(v["q1"]) for v in val], a.device); v2 = pooled(model, enc, [fmt(v["q2"]) for v in val], a.device)
    print(f"embedded {len(pairs):,} training pairs and {len(val)} validation pairs ({time.time() - t0:.0f}s)", flush=True)
    D = tr1.shape[1]
    head = torch.nn.Sequential(torch.nn.Linear(D, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    mu, sd = tr1.mean(0, keepdim=True), tr1.std(0, keepdim=True) + 1e-6
    norm = lambda z: (z - mu) / sd
    X1, X2 = norm(tr1).to(a.device), norm(tr2).to(a.device)
    n = len(X1)
    for step in range(a.steps):
        idx = torch.randint(0, n, (a.batch,), device=a.device)
        z1 = F.normalize(head(X1[idx]), dim=-1); z2 = F.normalize(head(X2[idx]), dim=-1)
        logits = z1 @ z2.T / a.temp
        lab = torch.arange(a.batch, device=a.device)
        loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    head.eval()
    with torch.no_grad():
        k1 = F.normalize(head(norm(v1).to(a.device)), dim=-1).cpu().numpy(); k2 = F.normalize(head(norm(v2).to(a.device)), dim=-1).cpu().numpy()
    raw1 = F.normalize(v1, dim=-1).numpy(); raw2 = F.normalize(v2, dim=-1).numpy()
    mem = m2_qa.ExchangeMemory()
    for v in val:
        mem.add(v["q2"].encode(), b"")
    Lit = np.array([[mem.similarity_keys(mem.key_of(v["q1"].encode()), mem.keys[j]) for j in range(len(val))] for v in val])
    K = k1 @ k2.T
    res = {"core": a.core, "pairs": len(pairs), "steps": a.steps, "literal_key": recall(Lit), "core_state_raw": recall(raw1 @ raw2.T),
           "learned_key": recall(K)}
    best = None
    for w in (0.25, 0.5, 1.0, 2.0):
        comb = Lit / (Lit.std() + 1e-9) + w * K / (K.std() + 1e-9)
        r = recall(comb)
        if best is None or r > best[0]:
            best = (r, w)
    res["literal_plus_learned"] = {"recall_at_1_10": best[0], "weight": best[1]}
    out = ROOT / "results" / f"k1_question_key_{Path(a.core).parent.name}.json"
    out.write_text(json.dumps(res, indent=1))
    torch.save({"head": head.state_dict(), "mu": mu, "sd": sd, "core": a.core, "weight": best[1],
                "lit_std": float(Lit.std()), "key_std": float(K.std())}, Path(a.core).parent / "key_head.pt")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
