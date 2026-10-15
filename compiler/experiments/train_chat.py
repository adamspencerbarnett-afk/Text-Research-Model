"""Train the 5M chat model v2 (15 October 2026, Adam: "a new 5M model based on the best of everything we have learned").

From the best 5M core (runs/core_mt5m: modern block, Muon + WSD, P100 mix, clean Wikipedia 1.260 bits/byte), one stage
(or two, --stage a / b, when session 33 picks the sequential structure) of five window kinds, all in the chat layout of
chat_engine.build_window, at a 2,048-code window:

  qa        the adopted Q&A recipe on Alpaca: own exchange 50% (30% of those reworded: copy under rewording), else the
            nearest by MiniLM; half of them show 2-3 candidates; 20% answered beside unrelated evidence (no refusing
            general questions)                                                            [sessions 2, 21, 23; skills lab]
  skills    data_skills/train_v4.jsonl: near misses, updates, 'I don't know', clarify + follow-up, combined facts,
            rewriting (short, yes/no, lists), 'your' not 'my'                             [skills arm: 33 -> 58-65/100]
  convo     real multi-turn conversations (data_mt: UltraChat + OpenAssistant), as many earlier turns as fit; 30% with an
            unrelated stored exchange in the memory slot (conversation is not memory)      [session 31: history gain]
  recall    long generated conversations: facts stated early (some corrected later), real filler turns between, then a
            question; when the fact's turn no longer fits the window it is in the memory slot, as the chat's session
            memory retrieves it                                                           [new: remember the whole chat]
  rehearse  core text from the P100 mix at the full window (keeps the language, teaches the longer positions)

Batches hold one kind each and are padded to their longest window only.

    python compiler/experiments/train_chat.py --out runs/chat5m_v2 --name chat5m_v2 --steps 30000 [--stage mixed|a|b] [--init X.pt]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import chat_engine as CE  # noqa: E402
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import gen_skills_v4 as G4  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import m2_qa  # noqa: E402
import train_lm  # noqa: E402

SHARES = {  # window kind -> share of batches
    "mixed": {"qa": 0.30, "skills": 0.25, "convo": 0.15, "recall": 0.10, "rehearse": 0.20},
    "a": {"qa": 0.45, "skills": 0.35, "rehearse": 0.20},                                   # sequential, stage 1
    "b": {"qa": 0.10, "skills": 0.10, "convo": 0.35, "recall": 0.25, "rehearse": 0.20},    # sequential, stage 2
}
ACKS = ["Got it. I'll remember that.", "Thanks, noted.", "Okay, I'll keep that in mind.", "Thanks for telling me.", "Noted."]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--core", default="runs/core_mt5m/scale_custom_v0_8k.pt"); ap.add_argument("--init", default="")
    ap.add_argument("--out", default="runs/chat5m_v2"); ap.add_argument("--name", default="chat5m_v2")
    ap.add_argument("--stage", default="mixed", choices=list(SHARES)); ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--skills-data", default="data_skills/train_v4.jsonl"); ap.add_argument("--mt-data", default="data_mt")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--amp", action="store_true"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--smoke", action="store_true", help="a few steps on small data: checks the script")
    a = ap.parse_args()
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    G3.setup(test=False)          # invented facts never use the names and phrasings reserved for the held-out tests
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    native = str(ROOT / "data" / "cv0.exe")
    core_ck = torch.load(ROOT / a.core, map_location="cpu", weights_only=False)
    dict_path = str(ROOT / core_ck["encoding"]["dictionary"]).replace("\\", "/")
    enc = LE.V0Dict(dict_path, native)
    sep = enc.encode(CE.SEP_TEXT)[0]
    shape = core_ck["model"]
    ctx, nb = a.ctx, CE.EV_CODES

    # ---- Q&A data, its count tables (as the engine rebuilds them), dense neighbours
    train_file = ROOT / "data_qa" / "train" / "qa_train.txt"
    train_x = m2_qa.parse_exchanges(train_file.read_bytes())
    all_codes = enc.encode_file(str(train_file), str(out))[0]
    if a.smoke:
        train_x = train_x[:3000]
    ngrams = [train_lm.NgramTable(all_codes, o, 64) for o in (2, 3)]
    nb_file = out / "dense_top4.npy"
    if nb_file.exists() and not a.smoke:
        nbrs = np.load(nb_file)
    else:
        import dense_keys
        dk = dense_keys.DenseKeys(a.device, weight=2.0)
        V = dk.embed([q for q, _ in train_x])
        nbrs = np.zeros((len(train_x), 4), dtype=np.int64)
        for s in range(0, len(train_x), 2048):
            sims = V[s:s + 2048] @ V.T
            sims[torch.arange(sims.shape[0]), torch.arange(s, s + sims.shape[0])] = -2
            nbrs[s:s + 2048] = torch.topk(sims, 4).indices.cpu().numpy()
        if not a.smoke:
            np.save(nb_file, nbrs)
        del dk, V; torch.cuda.empty_cache()
    say(f"{len(train_x):,} Q&A exchanges, dense neighbours ready")

    # ---- skills, conversations, rehearsal text
    shares = SHARES[a.stage]
    skills = []
    if "skills" in shares:
        with open(ROOT / a.skills_data, encoding="utf-8") as f:
            for i, l in enumerate(f):
                if a.smoke and i >= 2000:
                    break
                skills.append(json.loads(l))
        rng.shuffle(skills)
    dialogues = []
    if "convo" in shares or "recall" in shares:
        with open(ROOT / a.mt_data / "train" / "dialogues.jsonl", encoding="utf-8") as f:
            for i, l in enumerate(f):
                if a.smoke and i >= 3000:
                    break
                dialogues.append(json.loads(l)["turns"])
    say(f"{len(skills):,} skill windows, {len(dialogues):,} conversations")
    sample = out / "rehearse_sample.txt"
    if not sample.exists():
        parts = []
        for name, mb in (("wiki", 70), ("fineweb", 90), ("books", 40), ("ultrachat", 60), ("evidence", 40)):
            with open(ROOT / "data_core" / "mix_P100" / "train" / f"{name}.txt", "rb") as f:
                parts.append(f.read((3 if a.smoke else mb) * 1_000_000))
        sample.write_bytes(b"\f".join(parts))
    rehearse = enc.encode_file(str(sample), str(out))[0]
    say(f"rehearsal text: {len(rehearse):,} codes")

    # ---- model: the core, a fresh copy head and trust head; or a stage-1 model (--init)
    model = m2_qa.V2LM(enc, shape["width"], shape["layers"], shape["heads"], ctx, ngrams, True, False, False, shape.get("block", "modern")).to(dev)
    init = torch.load(ROOT / a.init, map_location="cpu", weights_only=False)["state_dict"] if a.init else core_ck["state_dict"]
    own = model.state_dict()
    sd = {k: v for k, v in init.items() if k in own and own[k].shape == v.shape and not k.startswith("table")}
    model.load_state_dict(sd, strict=False)
    say(f"model: {model.param_counts()['total']:,} learned parameters, window {ctx}; {len(sd)} tensors from {a.init or a.core}")

    B = lambda x: x if isinstance(x, bytes) else x.encode("utf-8", "replace")

    def pack(seq, ev, pl):
        seq = seq[:ctx + 1]
        mask = np.zeros(len(seq), np.float32); mask[pl:] = 1.0
        src = np.zeros(len(seq), bool); src[:ev] = True
        return seq, mask, src

    def w_qa():
        i = int(rng.integers(len(train_x))); q, ans = train_x[i]
        if rng.random() < 0.2:          # general question beside unrelated evidence: still answered
            cands = [train_x[int(rng.integers(len(train_x)))] for _ in range(int(rng.integers(1, 4)))]
            return pack(*CE.build_window(enc, sep, cands, [], q, ans, ctx, nb)[:3])
        self_ = rng.random() < 0.5
        first = (q, ans)
        if self_ and rng.random() < 0.3:
            first = (m2_qa.perturb_question(q, rng, train_x[int(rng.integers(len(train_x)))][0].split(b" ")), ans)
        if not self_:
            first = train_x[int(nbrs[i, 0])]
        cands = [first]
        if rng.random() < 0.5:
            pool = [(i if self_ else int(nbrs[i, 0]), first)] + [(int(j), train_x[int(j)]) for j in nbrs[i, (0 if self_ else 1):][:int(rng.integers(1, 3))]]
            cands = [c for _, c in sorted(pool, key=lambda t: t[0])]
        return pack(*CE.build_window(enc, sep, cands, [], q, ans, ctx, nb)[:3])

    sk = [0]

    def w_skills():
        w = skills[sk[0] % len(skills)]; sk[0] += 1
        return pack(*CE.build_window(enc, sep, [(B(c[0]), B(c[1])) for c in w["cands"]], [(B(t[0]), B(t[1])) for t in w["turns"]],
                                      B(w["q"]), B(w["target"]), ctx, nb)[:3])

    def turn_pairs(t):
        return [(t[j][1].encode("utf-8", "replace"), t[j + 1][1].encode("utf-8", "replace")) for j in range(0, len(t) - 1, 2)
                if t[j][0] == "user" and t[j + 1][0] == "assistant"]

    def w_convo():
        for _ in range(20):
            pairs = turn_pairs(dialogues[int(rng.integers(len(dialogues)))])
            if len(pairs) >= 2:
                break
        k = int(rng.integers(1, len(pairs)))
        q, ans = pairs[k][0][-1200:], pairs[k][1]
        cands = [train_x[int(rng.integers(len(train_x)))]] if rng.random() < 0.3 else []
        return pack(*CE.build_window(enc, sep, cands, pairs[:k], q, ans, ctx, nb)[:3])

    def filler(n):
        out_ = []
        while len(out_) < n:
            p = turn_pairs(dialogues[int(rng.integers(len(dialogues)))])
            if p:
                u, r = p[int(rng.integers(len(p)))]
                out_.append((u[:600], r[:700]))
        return out_

    def w_recall():
        facts = [G.Fact(rng) for _ in range(int(rng.integers(1, 4)))]
        turns = filler(int(rng.integers(1, 14)))
        slots = sorted(int(rng.integers(0, len(turns) + 1)) for _ in facts)
        stated = []
        for f, pos in sorted(zip(facts, slots), key=lambda t: -t[1]):
            ack = G.pick(rng, ACKS)
            turns.insert(pos, (f.stored()[1].encode(), ack.encode()))
        target_f = facts[int(rng.integers(len(facts)))]
        q, ans = target_f.ask(), G4.your(target_f.stored()[1])
        if target_f.can_update() and rng.random() < 0.3:          # a correction later in the conversation
            (_, new_ans), _ = target_f.update()
            turns.insert(int(rng.integers(len(turns) // 2, len(turns) + 1)), (("Actually, " + new_ans[0].lower() + new_ans[1:]).encode(), G.pick(rng, ACKS).encode()))
            ans = G4.your(new_ans)
        q, ans = q.encode(), ans.encode()
        codes_, ev, pl, kept = CE.build_window(enc, sep, [], turns, q, ans, ctx, nb)
        lost = turns[:len(turns) - kept]
        stmt = [t for t in lost if target_f.value.encode() in t[0] or (hasattr(target_f, "new_value") and target_f.new_value.encode() in t[0])]
        if stmt:                     # the fact scrolled out of the window: the session memory brings it back
            codes_, ev, pl, kept = CE.build_window(enc, sep, stmt[-2:], turns, q, ans, ctx, nb)
        return pack(codes_, ev, pl)

    def w_rehearse():
        j = int(rng.integers(0, len(rehearse) - ctx - 1))
        seq = rehearse[j:j + ctx + 1].astype(np.int64)
        return seq, np.ones(len(seq), np.float32), np.zeros(len(seq), bool)

    makers = {"qa": w_qa, "skills": w_skills, "convo": w_convo, "recall": w_recall, "rehearse": w_rehearse}
    kinds, probs = list(shares), np.array(list(shares.values())); probs = probs / probs.sum()

    heads = {n for n, _ in model.named_parameters() if n.split(".")[0] in ("copy_q", "copy_k", "mix_head")}
    groups = []
    for is_head in (False, True):
        ps = [(n, p) for n, p in model.named_parameters() if (n in heads) == is_head]
        lr = a.head_lr if is_head else a.lr
        groups += [{"params": [p for n, p in ps if p.dim() >= 2], "weight_decay": 0.1, "peak": lr},
                   {"params": [p for n, p in ps if p.dim() < 2], "weight_decay": 0.0, "peak": lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, 0.95))
    params = [p for g in groups for p in g["params"]]
    steps = 30 if a.smoke else a.steps
    stats = {k: [0.0, 0] for k in kinds}; log = []; t0 = time.time(); model.train()
    for step in range(steps):
        for g in opt.param_groups:
            g["lr"] = g["peak"] * min(1.0, (step + 1) / 200) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / steps)))
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        items = [makers[kind]() for _ in range(a.batch)]
        L = max(len(s) for s, _, _ in items)
        X = np.zeros((a.batch, L), np.int64); M = np.zeros((a.batch, L), np.float32); S = np.zeros((a.batch, L), bool)
        for r, (s, m, sr) in enumerate(items):
            X[r, :len(s)] = s; M[r, :len(m)] = m; S[r, :len(sr)] = sr
        x = torch.from_numpy(X).to(dev); m = torch.from_numpy(M[:, 1:]).to(dev); s = torch.from_numpy(S[:, :-1]).to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp and dev.type == "cuda"):
            nll = model.nll(x[:, :-1], x[:, 1:], s, torch.zeros(a.batch, device=dev))
        loss = (nll.float() * m).sum() / m.sum().clamp_min(1.0)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        st = stats[kind]; st[0] += float(loss); st[1] += 1
        if (step + 1) % 500 == 0 or step + 1 == steps:
            row = {"step": step + 1, "s": round(time.time() - t0), **{k: round(v[0] / max(1, v[1]) / math.log(2), 3) for k, v in stats.items() if v[1]}}
            log.append(row); say(f"step {step + 1}/{steps} ({(time.time() - t0) / (step + 1):.2f} s/step) bits/code by kind: "
                                 + ", ".join(f"{k} {v}" for k, v in row.items() if k in kinds))
            stats = {k: [0.0, 0] for k in kinds}
    args = {"data": "data_qa", "work": a.out, "dict": dict_path.replace(str(ROOT).replace("\\", "/") + "/", ""), "width": shape["width"],
            "layers": shape["layers"], "heads": shape["heads"], "ctx": ctx, "block": shape.get("block", "modern"), "copy_head": True,
            "evidence": False, "sim_feature": False, "eos": True, "topk": 64, "neighbor_codes": nb, "history_codes": ctx, "dense": 2.0,
            "dense_train": True, "max_train": 0, "fold_keys": False, "calculator": False, "copy_question": False, "rerank_core": 0,
            "greedy": True, "chat_v2": {"ev_codes": nb, "stage": a.stage, "shares": shares, "steps": steps, "core": a.core, "init": a.init,
                                        "skills_data": a.skills_data, "train_s": round(time.time() - t0), "log": log}}
    torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "args": args}, out / f"{a.name}.pt")
    say(f"saved {out / (a.name + '.pt')} ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
