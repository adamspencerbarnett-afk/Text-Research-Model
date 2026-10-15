"""K2 version 2: our own retrieval key with the core's top layers trained and outcome targets (15 October 2026, Adam).

K2 (session 32) froze the core and trained a small head on question rewordings: close to MiniLM on Quora, behind it on
answers. Version 2 changes two things:
  1. the core's last --unfreeze blocks (and its final norm) train with the head, at a gentle rate, so the core itself
     learns to represent questions for lookup; the saved key carries its own copy of the core
  2. OUTCOME targets: for training questions, the Q&A model scores how much each stored exchange (found by our own word
     key, no outside model) lowers the bits of the true answer; the most helpful one is the positive, a high-ranked one
     that does not help is the hard negative. Invented facts (stored question, the ways people ask it, the near miss)
     and Quora rewordings are kept beside them.
Benchmarks as session 32: Quora recall@1, skills-test retrieval@1, ms per query (literal and K2 v2; session 32 holds
MiniLM's numbers on the same items). Saved in K1/K2's format (+ its own core), so LearnedKeys and --key-head load it.

    python compiler/experiments/k2v2_question_key.py runs/core_D3s512/scale_5M_v0_8k_ng64.pt --tag 5m --device cuda
Writes runs/k2v2_<tag>/{core.pt, key_head.pt} and results/k2v2_question_key_<tag>.json.
"""
from __future__ import annotations

import argparse
import json
import math
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
import k2_question_key as K1  # noqa: E402
import m2_qa  # noqa: E402

fmt = K1.fmt


def batch_codes(enc, texts, ctx, maxlen=256):
    codes = [enc.encode(t)[0][:maxlen] for t in texts]
    L = max(1, max(len(c) for c in codes))
    x = np.zeros((len(codes), L), dtype=np.int64); m = np.zeros((len(codes), L), dtype=np.float32)
    for r, c in enumerate(codes):
        x[r, :len(c)] = c; m[r, :len(c)] = 1
    return torch.from_numpy(x), torch.from_numpy(m)


class Key(torch.nn.Module):
    """Mean of the core's final states over the question, normalised, through a two-layer head (K1's design)."""

    def __init__(self, core, mu, sd):
        super().__init__()
        self.core = core
        D = mu.shape[1]
        self.register_buffer("mu", mu); self.register_buffer("sd", sd)
        self.head = torch.nn.Sequential(torch.nn.Linear(D, 512), torch.nn.GELU(), torch.nn.Linear(512, 128))

    def pooled(self, x, m):
        h = self.core.hidden(x).float()
        return (h * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp_min(1)

    def forward(self, x, m):
        return F.normalize(self.head((self.pooled(x, m) - self.mu) / self.sd), dim=-1)


@torch.no_grad()
def embed_all(key, enc, texts, dev, batch=128):
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    out = torch.zeros(len(texts), 128)
    for s in range(0, len(order), batch):
        ids = order[s:s + batch]
        x, m = batch_codes(enc, [texts[i] for i in ids], key.core.ctx)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(dev).startswith("cuda")):
            out[ids] = key(x.to(dev), m.to(dev)).float().cpu()
    return out


@torch.no_grad()
def mine_outcomes(qa_path, n, dev, rng, log):
    """(question, most helpful stored question, a high-ranked unhelpful one) from the Q&A model's own answer bits."""
    eng = m2_qa.QAEngine(str(ROOT / qa_path), native=str(ROOT / "data" / "cv0.exe"), device=dev, log=lambda *x: None)
    eng.mem.learned = None                       # mining uses our own word key only
    enc, model, a = eng.enc, eng.model, eng.a
    ex = eng.mem.exchanges
    out, t0 = [], time.time()
    for i in rng.choice(len(ex), size=min(n, len(ex)), replace=False):
        q, ans = ex[int(i)]
        cands = [c for _, c in eng.mem.retrieve(q, 9, exclude=int(i))][:8]
        if len(cands) < 2:
            continue
        own = np.append(enc.encode(m2_qa.render(q, ans))[0], m2_qa.EOS)[: a.ctx // 2]
        ql = min(len(enc.encode(m2_qa.render(q, None))[0]), len(own) - 1)
        seqs, srcs, starts = [], [], []
        for c in [None] + cands:
            nb = enc.encode(m2_qa.render(*ex[c]))[0][-a.neighbor_codes:] if c is not None else np.array([], dtype=np.int64)
            seq = np.concatenate([nb, eng.sep, own])[: a.ctx + 1]
            seqs.append(seq); starts.append(len(nb) + len(eng.sep) + ql); srcs.append(len(nb))
        L = max(len(s) for s in seqs)
        X = torch.zeros(len(seqs), L, dtype=torch.long); S = torch.zeros(len(seqs), L - 1, dtype=torch.bool)
        for r, s in enumerate(seqs):
            X[r, :len(s)] = torch.from_numpy(s.astype(np.int64)); S[r, :srcs[r]] = True
        X, S = X.to(dev), S.to(dev)
        nll = model.nll(X[:, :-1], X[:, 1:], S, torch.zeros(len(seqs), device=dev)).float()
        bits = [float(nll[r, starts[r] - 1:len(seqs[r]) - 1].mean()) for r in range(len(seqs))]
        gains = [bits[0] - b for b in bits[1:]]
        best = int(np.argmax(gains))
        if gains[best] > 0.05:
            worst = int(np.argmin(gains[:4]))      # among the word key's top four: ranked high but no help
            out.append((q, ex[cands[best]][0], ex[cands[worst]][0] if gains[worst] <= 0 and worst != best else None))
    log(f"outcome pairs: {len(out):,} of {n:,} questions had a stored exchange that helps the answer ({time.time() - t0:.0f}s)")
    del eng; torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("core"); ap.add_argument("--tag", required=True); ap.add_argument("--device", default="cuda")
    ap.add_argument("--qa-model", default="models/qa_best_5M_dense.pt", help="Q&A model that scores which stored exchange helps")
    ap.add_argument("--mine", type=int, default=20000, help="training questions scored for outcome pairs")
    ap.add_argument("--unfreeze", type=int, default=2, help="core blocks (from the top) trained with the head")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--core-lr", type=float, default=2e-5)
    ap.add_argument("--temp", type=float, default=0.05); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = random.Random(a.seed); rng_np = np.random.default_rng(a.seed)
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    dev = torch.device(a.device)
    if a.smoke:
        a.mine, a.steps, a.batch = 200, 30, 32
    outc = mine_outcomes(a.qa_model, a.mine, a.device, rng_np, say)
    enc, core = C.load_core(str(ROOT / a.core), str(ROOT / "data" / "cv0.exe"), a.device)
    for p in core.parameters():
        p.requires_grad_(False)
    trainable = list(core.blocks[-a.unfreeze:]) + [core.ln_f] if a.unfreeze else []
    for mod in trainable:
        for p in mod.parameters():
            p.requires_grad_(True)
    qqp = []
    for d in (ROOT / "data_core" / "src" / "qqp.txt").read_text(encoding="utf-8").split("\n\f\n"):
        if "\nIn other words: " in d:
            q1, q2 = d.split("\nIn other words: ", 1)
            qqp.append((q1.removeprefix("Question: ").strip().encode(), q2.strip().encode(), None))
    syn = [(k.encode(), s.encode(), n.encode()) for s, k, n in K1.synth_pairs(400 if a.smoke else 40000, rng_np)]
    if a.smoke:
        qqp = qqp[:400]
    triples = [(q, p, n) for q, p, n in outc] * 2 + syn + qqp[:60000]     # outcome pairs count twice
    say(f"{len(outc):,} outcome x2 + {len(syn):,} invented + {min(len(qqp), 60000):,} Quora triples")
    # normalisation of the pooled states from a sample (as K1)
    with torch.no_grad():
        x, m = batch_codes(enc, [fmt(t[0]) for t in triples[:2048]], core.ctx)
        h = core.hidden(x.to(dev)).float(); pool = ((h * m.to(dev)[..., None]).sum(1) / m.to(dev).sum(1, keepdim=True)).cpu()
    key = Key(core, pool.mean(0, keepdim=True), pool.std(0, keepdim=True) + 1e-6).to(dev)
    opt = torch.optim.AdamW([{"params": key.head.parameters(), "lr": a.lr},
                             {"params": [p for mod in trainable for p in mod.parameters()], "lr": a.core_lr}], weight_decay=0.01)
    t0 = time.time()
    for step in range(a.steps):
        for g, base in zip(opt.param_groups, (a.lr, a.core_lr)):
            g["lr"] = base * min(1.0, (step + 1) / 100) * 0.5 * (1 + math.cos(math.pi * step / a.steps))
        bt = [triples[rng.randrange(len(triples))] for _ in range(a.batch)]
        xa, ma = batch_codes(enc, [fmt(t[0]) for t in bt], core.ctx); xp, mp = batch_codes(enc, [fmt(t[1]) for t in bt], core.ctx)
        negs = [i for i, t in enumerate(bt) if t[2] is not None]
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=dev.type == "cuda"):
            za = key(xa.to(dev), ma.to(dev)).float(); zp = key(xp.to(dev), mp.to(dev)).float()
            logits = za @ zp.T / a.temp
            if negs:
                xn, mn = batch_codes(enc, [fmt(bt[i][2]) for i in negs], core.ctx)
                zn = key(xn.to(dev), mn.to(dev)).float()
                hard = torch.full((a.batch,), float("-inf"), device=dev)
                hard[negs] = (za[negs] * zn).sum(-1) / a.temp
                logits = torch.cat([logits, hard[:, None]], 1)
        lab = torch.arange(a.batch, device=dev)
        loss = F.cross_entropy(logits, lab) + F.cross_entropy((za @ zp.T / a.temp).T, lab)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 250 == 0 or step == a.steps - 1:
            say(f"step {step} loss {loss.item():.3f} ({time.time() - t0:.0f}s)")
    key.eval()
    # --- save: the core with its trained top layers, and the head in K1/K2's format
    d = ROOT / "runs" / f"k2v2_{a.tag}"; d.mkdir(parents=True, exist_ok=True)
    ck = torch.load(ROOT / a.core, map_location="cpu", weights_only=False)
    sd = dict(ck["state_dict"]); sd.update({k: v.detach().cpu() for k, v in core.state_dict().items()})
    ck["state_dict"] = sd; ck["k2v2"] = {"from": a.core, "unfreeze": a.unfreeze}
    core_path = d / "core.pt"; torch.save(ck, core_path)
    # --- benchmarks (same items as session 32)
    res = {"core": a.core, "tag": a.tag, "unfreeze": a.unfreeze, "outcome_pairs": len(outc), "steps": a.steps}
    val = [json.loads(l) for l in open(ROOT / "data_core" / "eval" / "qqp_val.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(val); val = val[:300 if a.smoke else 2000]
    mem = m2_qa.ExchangeMemory()
    for v in val:
        mem.add(v["q2"].encode(), b"")
    Lit = np.array([[mem.similarity_keys(mem.key_of(v["q1"].encode()), mem.keys[j]) for j in range(len(val))] for v in val])
    k1 = embed_all(key, enc, [fmt(v["q1"]) for v in val], dev); k2 = embed_all(key, enc, [fmt(v["q2"]) for v in val], dev)
    K = (k1 @ k2.T).numpy()
    lit_std, key_std = float(Lit.std() + 1e-9), float(K.std() + 1e-9)
    best = max(((K1.recall(Lit / lit_std + w * K / key_std), w) for w in (0.5, 1.0, 2.0, 4.0, 8.0)))
    res["quora"] = {"literal": K1.recall(Lit), "k2v2_alone": K1.recall(K), "literal_plus_k2v2": best[0], "weight": best[1]}
    torch.save({"head": key.head.state_dict(), "mu": key.mu.cpu(), "sd": key.sd.cpu(), "core": str(core_path.relative_to(ROOT)).replace("\\", "/"),
                "weight": best[1], "lit_std": lit_std, "key_std": key_std}, d / "key_head.pt")
    import chat_probe
    import testset as T
    big = m2_qa.ExchangeMemory()
    train_x = m2_qa.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
    for q, ans in train_x + [(q.encode(), x.encode()) for q, x in chat_probe.TEACH + T.facts_in_order()]:
        big.add(q, ans)
    KV = embed_all(key, enc, [fmt(q) for q, _ in big.exchanges], dev)
    hits, t1 = [], time.perf_counter()
    for q, want in K1.skills_targets():
        hit = big.exact_hit(q.encode())
        if hit is None:
            r = big.retrieve(q.encode(), 50); lit = {i: s for s, i in r}
            cos = KV @ embed_all(key, enc, [fmt(q)], dev)[0]
            cand = set(lit) | set(torch.topk(cos, 50).indices.tolist())
            hit = max(cand, key=lambda i: lit.get(i, 0.0) / lit_std + best[1] * float(cos[i]) / key_std)
        hits.append(big.exchanges[hit][0].decode() in want)
    res["skills_retrieval_at_1"] = {"k2v2": f"{sum(hits)}/{len(hits)}"}
    res["ms_per_query"] = {"k2v2": round(1000 * (time.perf_counter() - t1) / len(hits), 1)}
    res["note"] = "session 32 on the same items: literal 41/70, K2-5M 39/70, K2-100M 38/70, MiniLM 41/70; Quora literal+MiniLM 0.809"
    (ROOT / "results" / f"k2v2_question_key_{a.tag}.json").write_text(json.dumps(res, indent=1))
    say(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
