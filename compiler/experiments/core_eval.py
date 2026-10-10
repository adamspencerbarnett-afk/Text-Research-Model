"""Report card for a stage-1 core, measured on the three jobs the design gives it
(docs/CORE_TRAINING_PLAN.md section 1):

  1. fluency      bits per byte on held-out text of each kind (one file per source in the mix's
                  heldout/ folder, plus the corpus v2 held-out books for comparison with older cores)
  2. evidence     how much better the core predicts a paragraph when a related paragraph is placed
                  before it than when an unrelated one is: evidence gain = 1 - bpb(related)/bpb(unrelated),
                  on data_core/eval/evidence_val.jsonl (pairs never trained on)
  3. questions    whether the core's own state finds a reworded question: each Quora validation
                  question (q1) is embedded as the mean of the core's final hidden states over its codes,
                  and its duplicate (q2) must be the nearest of 2,000 candidates by cosine; recall@1 and
                  @10, beside the literal word key used by the exchange memory

    python compiler/experiments/core_eval.py runs/core_D3/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda

Writes results/core_eval_<checkpoint stem>_<data name>.json.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import ladder_encodings as LE  # noqa: E402
import m2_qa  # noqa: E402
import train_lm  # noqa: E402


def load_core(path: str, native: str | None, device: str):
    """A train_lm checkpoint with its dictionary; count tables are not rebuilt (the report card
    measures the network, so table-free scores are comparable across data mixes)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info, shape = ck["encoding"], ck["model"]
    dict_path = Path(info["dictionary"])
    if not dict_path.exists():
        dict_path = Path(path).parent / dict_path.name
    enc = LE.V0Dict(str(dict_path), native)
    model = train_lm.LM(enc, shape["width"], shape["layers"], shape["heads"], shape["ctx"], 0, None, (2,), [])
    sd = {k: v for k, v in ck["state_dict"].items() if not k.startswith("mix_head") and not k.startswith("table")}
    model.load_state_dict(sd, strict=False)
    return enc, model.to(device).eval()


@torch.no_grad()
def bits_of(model, enc, prefix: bytes, target: bytes, device) -> tuple[float, int]:
    """Bits for the target's codes given the prefix, in one window (prefix trimmed from the left)."""
    p, t = enc.encode(prefix)[0], enc.encode(target)[0]
    t = t[: model.ctx - 1]
    p = p[-(model.ctx - len(t)):] if len(p) + len(t) > model.ctx else p
    seq = np.concatenate([p, t]).astype(np.int64)
    if len(seq) < 2:
        return 0.0, 0
    x = torch.from_numpy(seq[None, :-1]).to(device); y = torch.from_numpy(seq[None, 1:]).to(device)
    nll = model.nll(x, y)[0, len(p) - 1:] if len(p) else model.nll(x, y)[0]
    return float(nll.sum()) / math.log(2), len(target)


@torch.no_grad()
def embed(model, enc, texts: list[bytes], device, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(texts), batch):
        codes = [enc.encode(t)[0][: model.ctx] for t in texts[i:i + batch]]
        L = max(len(c) for c in codes)
        x = np.zeros((len(codes), L), dtype=np.int64); m = np.zeros((len(codes), L), dtype=np.float32)
        for r, c in enumerate(codes):
            x[r, :len(c)] = c; m[r, :len(c)] = 1
        h = model.hidden(torch.from_numpy(x).to(device))
        mt = torch.from_numpy(m).to(device)[..., None]
        v = (h * mt).sum(1) / mt.sum(1).clamp_min(1)
        out.append(torch.nn.functional.normalize(v, dim=-1).float().cpu().numpy())
    return np.concatenate(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--data", required=True); ap.add_argument("--device", default="cpu")
    ap.add_argument("--native", default=str(ROOT / "data" / "cv0.exe")); ap.add_argument("--n-evidence", type=int, default=600)
    ap.add_argument("--n-questions", type=int, default=2000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    native = a.native if Path(a.native).exists() else None
    enc, model = load_core(a.checkpoint, native, a.device)
    rng = random.Random(a.seed)
    res = {"checkpoint": a.checkpoint, "data": a.data, "fluency": {}, "evidence": {}, "questions": {}}
    t0 = time.time()

    # 1. fluency: network-only bits per byte on each held-out file (first 400 KB of each)
    files = sorted(glob.glob(f"{a.data}/heldout/*.txt")) + sorted(glob.glob(str(ROOT / "data_v2" / "heldout" / "*.txt")))
    work = Path(a.checkpoint).parent
    for f in files:
        blob = Path(f).read_bytes()[:400_000]
        if len(blob) < 1000:
            continue
        tmp = work / f"_eval_{Path(f).parent.parent.name}_{Path(f).stem}.txt"
        tmp.write_bytes(blob)
        toks, _ = enc.encode_file(str(tmp), work)
        bits, _ = train_lm.total_bits(model, toks)
        name = ("v2_books/" if "data_v2" in f else "") + Path(f).stem
        res["fluency"][name] = round(bits / len(blob), 4)
    print("fluency", res["fluency"], f"({time.time() - t0:.0f}s)", flush=True)

    # 2. evidence gain
    ev = [json.loads(l) for l in open(ROOT / "data_core" / "eval" / "evidence_val.jsonl", encoding="utf-8")]
    rng.shuffle(ev); ev = ev[: a.n_evidence]
    rel = unr = nb = 0.0
    for i, e in enumerate(ev):
        other = ev[(i + len(ev) // 2) % len(ev)]["evidence"]
        b1, n = bits_of(model, enc, (e["evidence"] + "\n\n").encode(), e["target"].encode(), a.device)
        b2, _ = bits_of(model, enc, (other + "\n\n").encode(), e["target"].encode(), a.device)
        rel += b1; unr += b2; nb += n
    res["evidence"] = {"pairs": len(ev), "bpb_related": round(rel / nb, 4), "bpb_unrelated": round(unr / nb, 4),
                       "evidence_gain_pct": round(100 * (1 - rel / max(unr, 1e-9)), 2)}
    print("evidence", res["evidence"], f"({time.time() - t0:.0f}s)", flush=True)

    # 3. question representations: reworded-question retrieval among N candidates
    qs = [json.loads(l) for l in open(ROOT / "data_core" / "eval" / "qqp_val.jsonl", encoding="utf-8")]
    rng.shuffle(qs); qs = qs[: a.n_questions]
    e1 = embed(model, enc, [("Question: " + q["q1"]).encode() for q in qs], a.device)
    e2 = embed(model, enc, [("Question: " + q["q2"]).encode() for q in qs], a.device)
    sims = e1 @ e2.T
    ranks = (sims > sims.diagonal()[:, None]).sum(1)
    res["questions"]["core_state"] = {"n": len(qs), "recall_at_1": round(float((ranks == 0).mean()), 4), "recall_at_10": round(float((ranks < 10).mean()), 4)}
    mem = m2_qa.ExchangeMemory()
    for q in qs:
        mem.add(q["q2"].encode(), b"")
    hit1 = hit10 = 0
    for i, q in enumerate(qs):
        k = mem.key_of(q["q1"].encode())
        scored = sorted(((mem.similarity_keys(k, mem.keys[j]), j) for j in range(len(qs))), reverse=True)
        order = [j for _, j in scored]
        hit1 += order[0] == i; hit10 += i in order[:10]
    res["questions"]["literal_key"] = {"n": len(qs), "recall_at_1": round(hit1 / len(qs), 4), "recall_at_10": round(hit10 / len(qs), 4)}
    print("questions", res["questions"], f"({time.time() - t0:.0f}s)", flush=True)

    out = ROOT / "results" / f"core_eval_{Path(a.checkpoint).stem}_{Path(a.data).name}.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
