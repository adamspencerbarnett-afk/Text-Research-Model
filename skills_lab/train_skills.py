"""Continue a Q&A model's training with skill windows (14 October 2026, CPU, beside the GPU queue).

    .venv/Scripts/python.exe -B skills_lab/train_skills.py --init models/qa_best_5M_dense.pt --save skills_lab/models/NAME.pt \
        --steps 3000 [--skill-p 0.35] [--k-p 0.5] [--threads 16]

Window mix per batch:
  rehearse-p   core text (keeps the language), as in every research Q&A stage
  skill-p      synthetic skill windows (gen_skills.py): near misses, updates, unknowns, clarifying questions, follow-ups
  the rest     the research Q&A windows on Alpaca (own exchange 50% of the time, 30% of those reworded, else the nearest
               other exchange); a share k-p of them shows 2-3 candidates (the own or nearest exchange among the next
               dense neighbours, in stored order) instead of one, so choosing among candidates is also learned on real data
--skill-p 0 --k-p 0 is the control: the same steps of the research recipe only.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
import numpy as np  # noqa: E402
import torch  # noqa: E402

import engine as E  # noqa: E402
import gen_skills as G  # noqa: E402

m2_qa = E.m2_qa
LE, train_lm = m2_qa.LE, m2_qa.train_lm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", required=True); ap.add_argument("--save", required=True)
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5); ap.add_argument("--head-lr", type=float, default=5e-4)
    ap.add_argument("--skill-p", type=float, default=0.35); ap.add_argument("--k-p", type=float, default=0.5)
    ap.add_argument("--rehearse-p", type=float, default=0.25); ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--dataset", default="", help="read skill windows from this file (build_skills_data.py), in a shuffled order, instead of generating them")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--amp", action="store_true", help="bfloat16 autocast on the GPU")
    ap.add_argument("--gen-version", type=int, default=1, help="gen_skills mix: 1 = first test, 2 = more answerable and clarifying windows")
    ap.add_argument("--unrelated-p", type=float, default=0.0, help="share of research windows whose evidence is 1-3 unrelated exchanges and whose "
                    "target is still the real answer (general questions are answered from the network, not refused)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads); torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    G.set_version(a.gen_version)
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    ck = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)
    a0 = argparse.Namespace(**ck["args"])
    res = lambda p: str(p if Path(p).is_absolute() else ROOT / p)
    enc = LE.V0Dict(res(a0.dict), str(ROOT / "data" / "cv0.exe"))
    train_file = res(f"{a0.data}/train/qa_train.txt")
    train_x = m2_qa.parse_exchanges(Path(train_file).read_bytes())
    work = LAB / "work"
    all_codes = enc.encode_file(train_file, str(work))[0]
    codes = [enc.encode(m2_qa.render(q, x))[0] for q, x in train_x]
    say(f"{len(train_x):,} exchanges encoded")
    sep = enc.encode(b"\n\n")[0]
    # dense neighbours of every training question (the same sentence encoder as retrieval), top 4 excluding itself
    nb_file = work / "dense_top4.npy"
    if nb_file.exists():
        nbrs = np.load(nb_file)
    else:
        import dense_keys
        dk = dense_keys.DenseKeys("cpu", weight=2.0)
        V = dk.embed([q for q, _ in train_x])
        nbrs = np.zeros((len(train_x), 4), dtype=np.int64)
        for s in range(0, len(train_x), 2048):
            sims = V[s:s + 2048] @ V.T
            sims[torch.arange(sims.shape[0]), torch.arange(s, s + sims.shape[0])] = -2
            nbrs[s:s + 2048] = torch.topk(sims, 4).indices.numpy()
        np.save(nb_file, nbrs)
    say("dense neighbours ready")
    ngrams = [train_lm.NgramTable(all_codes, o, a0.topk) for o in (2, 3)]
    model = m2_qa.V2LM(enc, a0.width, a0.layers, a0.heads, a0.ctx, ngrams, a0.copy_head, getattr(a0, "evidence", False),
                       getattr(a0, "sim_feature", False), getattr(a0, "block", "legacy"))
    model.load_state_dict(ck["state_dict"], strict=False)
    dev = torch.device(a.device); model.to(dev)
    ctx = a0.ctx
    data, ptr = None, [0]
    if a.dataset:
        data = [json.loads(l) for l in open(ROOT / a.dataset, encoding="utf-8")]
        rng.shuffle(data)
        say(f"skills dataset: {len(data):,} windows from {a.dataset}")
    rehearse = None
    if a.rehearse_p and getattr(a0, "rehearse", None):
        import glob
        concat = work / "rehearse_concat.txt"
        if not concat.exists():
            concat.write_bytes(b"".join(Path(f).read_bytes() for f in sorted(glob.glob(res(a0.rehearse) + "/train/*.txt"))))
        rehearse = enc.encode_file(str(concat), str(work))[0]
        say(f"rehearsal text: {len(rehearse):,} codes")

    def pack(seq, ev_len, prompt_len):
        seq = seq[:ctx + 1]
        mask = np.zeros(ctx + 1, dtype=np.float32); mask[prompt_len:len(seq)] = 1.0
        src = np.zeros(ctx + 1, dtype=bool); src[:ev_len] = True
        return np.pad(seq, (0, ctx + 1 - len(seq))), mask, src

    def research(i):
        q, ans = train_x[i]
        if a.unrelated_p and rng.random() < a.unrelated_p:
            cands = [train_x[int(rng.integers(len(train_x)))] for _ in range(int(rng.integers(1, 4)))]
            seq, ev, pl = E.build_window(enc, sep, cands, [], q, ans)
            return pack(seq, ev, pl)
        self_ = rng.random() < 0.5
        own_ev = (q, ans)
        if self_ and rng.random() < 0.3:      # reworded own question (the copy-under-rewording recipe)
            own_ev = (m2_qa.perturb_question(q, rng, train_x[int(rng.integers(len(train_x)))][0].split(b" ")), ans)
        first = own_ev if self_ else train_x[int(nbrs[i, 0])]
        if rng.random() < a.k_p:
            n_more = int(rng.integers(1, 3))
            idx = [int(j) for j in nbrs[i, (0 if self_ else 1):][:n_more]]
            pool = [(i if self_ else int(nbrs[i, 0]), first)] + [(j, train_x[j]) for j in idx]
            cands = [c for _, c in sorted(pool, key=lambda t: t[0])]
            seq, ev, pl = E.build_window(enc, sep, cands, [], q, ans)
        else:
            ev_codes = codes[i][-E.NB:] if (self_ and own_ev[0] == q) else enc.encode(m2_qa.render(*first))[0][-E.NB:]
            own = np.append(codes[i], m2_qa.EOS)
            ql = len(enc.encode(m2_qa.render(q, None))[0])
            seq = np.concatenate([ev_codes, sep, own]); ev, pl = len(ev_codes), len(ev_codes) + len(sep) + ql
        return pack(seq, ev, pl)

    alpaca = lambda r: train_x[int(r.integers(len(train_x)))]
    B = lambda x: x if isinstance(x, bytes) else x.encode()
    kinds = {}

    def skill():
        if data is not None:
            w = data[ptr[0] % len(data)]; ptr[0] += 1
        else:
            w = G.make(rng, alpaca)
        kinds[w["kind"]] = kinds.get(w["kind"], 0) + 1
        seq, ev, pl = E.build_window(enc, sep, [(B(c[0]), B(c[1])) for c in w["cands"]],
                                     [(B(t[0]), B(t[1])) for t in w["turns"]], w["q"].encode(), w["target"].encode(),
                                     codes_of=lambda c: enc.encode(m2_qa.render(*c))[0])
        return pack(seq, ev, pl)

    def one():
        r = rng.random()
        if rehearse is not None and r < a.rehearse_p:
            j = int(rng.integers(0, len(rehearse) - ctx - 1))
            return rehearse[j:j + ctx + 1].astype(np.int64), np.ones(ctx + 1, np.float32), np.zeros(ctx + 1, bool)
        if r < a.rehearse_p + a.skill_p:
            return skill()
        return research(int(rng.integers(len(train_x))))

    heads = {n for n, _ in model.named_parameters() if n.split(".")[0] in ("copy_q", "copy_k", "mix_head")}
    groups = []
    for is_head in (False, True):
        ps = [(n, p) for n, p in model.named_parameters() if (n in heads) == is_head]
        lr = a.head_lr if is_head else a.lr
        groups += [{"params": [p for n, p in ps if p.dim() >= 2], "weight_decay": 0.1, "lr": lr, "peak": lr},
                   {"params": [p for n, p in ps if p.dim() < 2], "weight_decay": 0.0, "lr": lr, "peak": lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], betas=(0.9, 0.95))
    params = [p for g in groups for p in g["params"]]
    model.train(); t0 = time.time(); log = []
    for step in range(a.steps):
        for g in opt.param_groups:
            g["lr"] = g["peak"] * min(1.0, (step + 1) / 50) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        xs, ms, ss = zip(*(one() for _ in range(a.batch)))
        x = torch.from_numpy(np.stack(xs)).to(dev); m = torch.from_numpy(np.stack(ms))[:, 1:].to(dev); s = torch.from_numpy(np.stack(ss))[:, :-1].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
            nll = model.nll(x[:, :-1], x[:, 1:], s, torch.zeros(a.batch, device=dev))
        nll = nll.float()
        loss = (nll * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        if (step + 1) % 100 == 0 or step == 0:
            el = time.time() - t0
            log.append({"step": step + 1, "loss_bits": round(float(loss) / math.log(2), 3), "s": round(el)})
            say(f"step {step + 1}/{a.steps}  loss {float(loss) / math.log(2):.3f} bits/code  {el / (step + 1):.2f} s/step  kinds {kinds}")
    if data is not None:
        say(f"skill windows used: {ptr[0]:,} ({ptr[0] / len(data):.2f} passes over the dataset)")
    args = dict(ck["args"]); args["skills_lab"] = {"init": a.init, **{k: v for k, v in vars(a).items()}, "log": log, "train_s": round(time.time() - t0)}
    Path(a.save).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "args": args}, a.save)
    say(f"saved {a.save} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
