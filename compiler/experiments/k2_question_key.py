"""K2: our own learned question key, to replace the outside sentence encoder (MiniLM) in retrieval (14 October 2026).

K1 (11 October) put a small head on the frozen 5M core's mean state and trained it on Quora duplicates only: +3.5
points on Quora, no answer gain. K2 keeps the idea (and K1's file format, so m2_qa's --key-head and LearnedKeys load
it unchanged) and changes what it learns from and on:
  cores       the 100M core (quality) or the 5M core (speed); frozen, nothing outside the project
  positives   Quora duplicates (134k), our own reworded Alpaca questions (perturb_question, 51k), and invented-fact
              rewordings from gen_skills (stored question <-> the ways people ask it)
  negatives   in-batch, plus a hard negative per invented fact: the near miss (Anna's car / Anne's car, the same
              person's other field), which must land apart
Benchmarks, each for the literal key alone, literal + MiniLM (weight 2, the research setting) and literal + K2:
  recall@1 on 2,000 held-out Quora pairs; retrieval@1 on the skills test (the right stored fact first, in the full
  51k-exchange memory plus the test facts); milliseconds per query.

    python compiler/experiments/k2_question_key.py runs/core_P100/scale_custom_v0_8k.pt --tag 100m --device cuda

Writes runs/k2_<tag>/key_head.pt and results/k2_question_key_<tag>.json.
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
import gen_skills as G  # noqa: E402
import m2_qa  # noqa: E402
import testset as T  # noqa: E402

fmt = lambda q: b"Question: " + (q if isinstance(q, bytes) else q.encode())


@torch.no_grad()
def pooled(model, enc, texts, device, batch=256):
    """Mean of the core's final hidden states over each text (as LearnedKeys.embed)."""
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))      # similar lengths together: less padding
    for s in range(0, len(order), batch):
        ids = order[s:s + batch]
        codes = [enc.encode(texts[i])[0][: model.ctx] for i in ids]
        L = max(len(c) for c in codes)
        x = np.zeros((len(codes), L), dtype=np.int64); m = np.zeros((len(codes), L), dtype=np.float32)
        for r, c in enumerate(codes):
            x[r, :len(c)] = c; m[r, :len(c)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(device).startswith("cuda")):
            h = model.hidden(torch.from_numpy(x).to(device)).float()
        mt = torch.from_numpy(m).to(device)[..., None]
        out.append((ids, ((h * mt).sum(1) / mt.sum(1).clamp_min(1)).cpu()))
    res = torch.zeros(len(texts), out[0][1].shape[1])
    for ids, v in out:
        res[ids] = v
    return res


def recall(S: np.ndarray):
    r = (S > S.diagonal()[:, None]).sum(1)
    return round(float((r == 0).mean()), 4), round(float((r < 10).mean()), 4)


def synth_pairs(n, rng_np):
    """(stored question, a way of asking it, a near-miss stored question) for invented facts."""
    out = []
    while len(out) < n:
        f = G.Fact(rng_np)
        if f.kind == "p" and rng_np.random() < 0.5:
            g = G.Fact(rng_np, "p", f.field, G.person(rng_np), extra={"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]})
        elif f.kind == "p":
            g = G.Fact(rng_np, "p", G.pick(rng_np, [k for k in G.P if k != f.field]), dict(f.ent))
        else:
            g = G.Fact(rng_np, "t", f.field, {"t": G.pick(rng_np, [t for t in G.TH[f.field][0] if t != f.ent["t"]])})
        ask = G.pick(rng_np, f.spec[2]).format(**f.fmt, v=f.value, art="a")
        out.append((f.stored()[0], ask, g.stored()[0]))
    return out


def skills_targets():
    """(question, set of stored questions that count as the right first hit)."""
    out = [(ask, {q}) for q, _, ask, _ in T.TAUGHT]
    for (old, new, ask, *_), kind in zip(T.UPDATE, T.UPDATE_KIND):
        out.append((ask, {new[0]} if kind == "current" else {old[0]}))
    for f1, f2, (q1, _), (q2, _) in T.NEARMISS:
        out += [(q1, {f1[0]}), (q2, {f2[0]})]
    out += [(ask, {f1[0], f2[0]}) for f1, f2, ask, *_ in T.CLARIFY]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("core"); ap.add_argument("--tag", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--temp", type=float, default=0.05); ap.add_argument("--n-val", type=int, default=2000)
    ap.add_argument("--n-synth", type=int, default=60000); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--no-minilm", action="store_true", help="skip the MiniLM reference")
    ap.add_argument("--out-dir", default="", help="write the result and the head here instead of results/ and runs/k2_<tag>/ (side tests)")
    ap.add_argument("--smoke", action="store_true", help="a few thousand pairs and 50 steps: checks the script, not the key")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = random.Random(a.seed); rng_np = np.random.default_rng(a.seed)
    native = ROOT / "data" / "cv0.exe"
    enc, model = C.load_core(a.core, str(native) if native.exists() else None, a.device)
    t0 = time.time()
    # --- positives and hard negatives
    qqp = []
    for d in (ROOT / "data_core" / "src" / "qqp.txt").read_text(encoding="utf-8").split("\n\f\n"):
        if "\nIn other words: " in d:
            q1, q2 = d.split("\nIn other words: ", 1)
            qqp.append((q1.removeprefix("Question: ").strip().encode(), q2.strip().encode(), None))
    train_x = m2_qa.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
    alp = [(q, m2_qa.perturb_question(q, rng_np, train_x[int(rng_np.integers(len(train_x)))][0].split(b" ")), None) for q, _ in train_x]
    syn = [(s.encode(), k.encode(), n.encode()) for s, k, n in synth_pairs(3000 if a.smoke else a.n_synth, rng_np)]
    if a.smoke:
        qqp, alp, syn, a.steps, a.n_val, a.batch = qqp[:3000], alp[:3000], syn[:3000], 50, 300, 128
        train_x = train_x[:5000]
    triples = qqp + alp + syn
    texts = sorted({t for tr in triples for t in tr if t is not None})
    pos = {t: i for i, t in enumerate(texts)}
    V = pooled(model, enc, [fmt(t) for t in texts], a.device)
    A = torch.tensor([pos[t[0]] for t in triples]); P = torch.tensor([pos[t[1]] for t in triples])
    N = torch.tensor([pos[t[2]] if t[2] is not None else -1 for t in triples])
    print(f"{len(qqp):,} Quora + {len(alp):,} Alpaca + {len(syn):,} invented pairs, {len(texts):,} texts embedded ({time.time() - t0:.0f}s)", flush=True)
    D = V.shape[1]
    mu, sd = V.mean(0, keepdim=True), V.std(0, keepdim=True) + 1e-6
    X = ((V - mu) / sd).to(a.device)
    head = torch.nn.Sequential(torch.nn.Linear(D, 512), torch.nn.GELU(), torch.nn.Linear(512, 128)).to(a.device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(a.steps):
        idx = torch.randint(0, len(triples), (a.batch,))
        za = F.normalize(head(X[A[idx].to(a.device)]), dim=-1); zp = F.normalize(head(X[P[idx].to(a.device)]), dim=-1)
        n_idx = N[idx]; has = n_idx >= 0
        logits = za @ zp.T / a.temp
        if has.any():                                   # each anchor's own near miss as an extra wrong choice
            zn = F.normalize(head(X[n_idx.clamp_min(0).to(a.device)]), dim=-1)
            hard = ((za * zn).sum(-1) / a.temp).masked_fill(~has.to(a.device), float("-inf"))
            logits = torch.cat([logits, hard[:, None]], 1)
        lab = torch.arange(a.batch, device=a.device)
        loss = F.cross_entropy(logits, lab) + F.cross_entropy((za @ zp.T / a.temp).T, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
    head.eval()
    key = lambda vecs: F.normalize(head(((vecs - mu) / sd).to(a.device)), dim=-1).detach().cpu()
    res = {"core": a.core, "tag": a.tag, "pairs": {"quora": len(qqp), "alpaca": len(alp), "invented": len(syn)}, "steps": a.steps}
    # --- 1. Quora held-out pairs, as K1
    val = [json.loads(l) for l in open(ROOT / "data_core" / "eval" / "qqp_val.jsonl", encoding="utf-8")]
    rng.shuffle(val); val = val[: a.n_val]
    mem = m2_qa.ExchangeMemory()
    for v in val:
        mem.add(v["q2"].encode(), b"")
    Lit = np.array([[mem.similarity_keys(mem.key_of(v["q1"].encode()), mem.keys[j]) for j in range(len(val))] for v in val])
    with torch.no_grad():
        k1 = key(pooled(model, enc, [fmt(v["q1"]) for v in val], a.device)); k2 = key(pooled(model, enc, [fmt(v["q2"]) for v in val], a.device))
    K = (k1 @ k2.T).numpy()
    lit_std, key_std = float(Lit.std() + 1e-9), float(K.std() + 1e-9)
    best = max(((recall(Lit / lit_std + w * K / key_std), w) for w in (0.25, 0.5, 1.0, 2.0, 4.0)))
    res["quora"] = {"literal": recall(Lit), "k2_alone": recall(K), "literal_plus_k2": best[0], "weight": best[1]}
    dk = None
    if not a.no_minilm:
        import dense_keys
        dk = dense_keys.DenseKeys(a.device, weight=2.0)
        M = (dk.embed([v["q1"].encode() for v in val]) @ dk.embed([v["q2"].encode() for v in val]).T).cpu().numpy()
        res["quora"]["minilm_alone"] = recall(M); res["quora"]["literal_plus_minilm"] = recall(Lit + 2.0 * M)
    # --- 2. skills-test retrieval in the full memory (as the chat sees it)
    import chat_probe
    big = m2_qa.ExchangeMemory()
    for q, ans in train_x + [(q.encode(), x.encode()) for q, x in chat_probe.TEACH + T.facts_in_order()]:
        big.add(q, ans)
    with torch.no_grad():
        KV = key(pooled(model, enc, [fmt(q) for q, _ in big.exchanges], a.device))
    MV = dk.embed([q for q, _ in big.exchanges]) if dk is not None else None

    def first(q: bytes, mode: str) -> bytes:
        hit = big.exact_hit(q)
        if hit is not None:
            return big.exchanges[hit][0]
        r = big.retrieve(q, 50)
        lit = {i: s for s, i in r}
        if mode == "literal":
            return big.exchanges[r[0][1]][0] if r else b""
        if mode == "k2":
            with torch.no_grad():
                cos = (KV @ key(pooled(model, enc, [fmt(q)], a.device))[0])
            w, ls, ks = best[1], lit_std, key_std
        else:
            cos = (MV @ dk.embed([q])[0]).cpu(); w, ls, ks = 2.0, 1.0, 1.0
        cand = set(lit) | set(torch.topk(cos, 50).indices.tolist())
        top = max(cand, key=lambda i: lit.get(i, 0.0) / ls + w * float(cos[i]) / ks)
        return big.exchanges[top][0]

    tg = skills_targets()
    for mode in ["literal", "k2"] + (["minilm"] if dk is not None else []):
        t1 = time.perf_counter()
        hits = [first(q.encode(), mode).decode() in want for q, want in tg]
        res.setdefault("skills_retrieval_at_1", {})[mode] = f"{sum(hits)}/{len(hits)}"
        res.setdefault("ms_per_query", {})[mode] = round(1000 * (time.perf_counter() - t1) / len(tg), 1)
    base = Path(a.out_dir) if a.out_dir else None
    out = (base or ROOT / "results") / f"k2_question_key_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True); out.write_text(json.dumps(res, indent=1))
    d = (base / f"k2_{a.tag}") if base else ROOT / "runs" / f"k2_{a.tag}"; d.mkdir(parents=True, exist_ok=True)
    torch.save({"head": head.state_dict(), "mu": mu, "sd": sd, "core": a.core, "weight": best[1],
                "lit_std": lit_std, "key_std": key_std}, d / "key_head.pt")
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
