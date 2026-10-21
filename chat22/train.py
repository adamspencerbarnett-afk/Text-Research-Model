"""chat22 training: one mixed stage from a core checkpoint, stopped by held-out measurements, not a fixed step count
(docs/V22_CLEAN_DESIGN.md section 9).

  held-out windows   built once from held-out sources (held-out Q&A, the last 2,000 skill records, held-out dialogues,
                     clean wiki/fineweb/books, generated recall/story windows of another seed); every 500 steps: loss per
                     kind, evidence-target accuracy, and 'reply_sure' (every reply code given more than half the probability) per kind
  combined score     the mean over kinds of held-out reply loss relative to the first check (lower is better), so no
                     kind can hide another's loss
  schedule           warmup (200 steps), then a STABLE rate while the combined score improves; plateau = no gain of more
                     than 0.5% over the last 3 checks (patience 1,500 steps) after min_steps; then DECAY over 20% of the
                     steps so far (cosine to 10%) and stop; never past max_steps
  best checkpoint    the one with the best combined score is the model (with its tables beside it)
  overfitting alarm  a kind whose training loss falls while its held-out loss rises over 2 checks is logged
  resume             state saved every check (model, optimiser, rng, step, log); rerunning the command resumes

    python -m chat22.train --name v22_s1 [--seed 1] [--ctx 2048] [--min-steps 10000] [--max-steps 60000] [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from . import encoder as E
from .model import ChatLM
from .tables import NgramTable, save_tables
from .windows_train import Windows

ROOT = Path(__file__).resolve().parents[1]
SHARES = {"qa": 0.20, "skills": 0.15, "convo": 0.13, "recall": 0.12, "buried": 0.10, "docqa": 0.07, "story": 0.10, "rehearse": 0.13}
# v2.3 (docs/MIX_STUDY_V23.md): any-length conversations with the real search; buried evidence back to 15; more plain text
SHARES_V23 = {"longconv": 0.18, "buried": 0.15, "rehearse": 0.17, "qa": 0.12, "skills": 0.12, "convo": 0.12, "story": 0.08,
              "docqa": 0.04, "recall": 0.02}


def batch_tensors(ws, dev):
    L = max(len(w.codes) for w in ws)
    B = len(ws)
    X = np.zeros((B, L), np.int64); M = np.zeros((B, L), np.float32); S = np.zeros((B, L), bool)
    Gd = np.zeros((B, L), bool); Pt = np.zeros((B, L), bool); lo = np.zeros(B, np.float32)
    for r, w in enumerate(ws):
        n = len(w.codes); X[r, :n] = w.codes
        M[r, w.prompt_len:n] = 1.0 if w.prompt_len > 0 else 0.0
        if w.prompt_len == 0:
            M[r, :n] = 1.0                               # plain text: every code is a target
        S[r, :w.copy_end] = True
        for a, b in w.gold:
            Gd[r, a:b] = True
        for a, b in w.point:
            Pt[r, a:b] = True
        lo[r] = w.loo
    t = lambda a: torch.from_numpy(a).to(dev)
    x = t(X)
    return x[:, :-1], x[:, 1:], t(M[:, 1:]), t(S[:, :-1]), t(Gd[:, :-1]), t(Pt[:, :-1]), t(lo)


def losses(model, ws, dev, aux_w, point_w, amp):
    x, y, m, s, g, p, lo = batch_tensors(ws, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
        r = model.score(x, y, s, g, p, lo if model.training else torch.zeros_like(lo))
    nll = (r["nll"].float() * m).sum() / m.sum().clamp_min(1.0)
    am = r["has"].float() * m; pm = r["has_point"].float() * m
    aux = (r["aux"] * am).sum() / am.sum() if am.sum() > 0 else torch.zeros((), device=dev)
    pt = (r["point"] * pm).sum() / pm.sum() if pm.sum() > 0 else torch.zeros((), device=dev)
    return nll, aux, pt, r, (x, y, m)


@torch.no_grad()
def evaluate(model, held: dict, dev, amp, bs: int = 8) -> dict:
    model.eval()
    out = {}
    for kind, ws in held.items():
        tot, n, ex, exn, aux_hit, aux_n = 0.0, 0, 0, 0, 0, 0
        for i in range(0, len(ws), bs):
            chunk = ws[i:i + bs]
            nll, aux, pt, r, (x, y, m) = losses(model, chunk, dev, 0, 0, amp)
            tot += float(nll) * len(chunk); n += len(chunk)
            # reply_sure: every target code of the reply gets more than half the probability (so it is also the top
            # choice): a cheap stand-in for "the reply would be written exactly", computed without decoding
            for row in range(len(chunk)):
                mm = m[row] > 0
                if mm.any() and kind != "rehearse":
                    ex += bool((r["nll"][row][mm] < math.log(2)).all()); exn += 1
            if "has" in r:
                hm = r["has"] & (m > 0)
                aux_hit += int((r["aux"][hm] < math.log(2)).sum()); aux_n += int(hm.sum())
        out[kind] = {"loss": round(tot / max(1, n) / math.log(2), 4), "reply_sure": round(ex / exn, 3) if exn else None,
                     "evidence_right": round(aux_hit / aux_n, 3) if aux_n else None}
    model.train()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True); ap.add_argument("--out", default="runs/chat22")
    ap.add_argument("--core", default="runs/core_mt5m/scale_custom_v0_8k.pt"); ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--rope-scale", type=float, default=1.0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--batch", type=int, default=16); ap.add_argument("--lr", type=float, default=1e-4); ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation (20 Oct, for the 100M): the batch is run in this many "
                    "parts before one update; same windows and update, less GPU memory (1 = unchanged)")
    ap.add_argument("--aux", type=float, default=0.5); ap.add_argument("--point", type=float, default=0.5)
    ap.add_argument("--min-steps", type=int, default=10000); ap.add_argument("--max-steps", type=int, default=60000)
    ap.add_argument("--eval-every", type=int, default=500); ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--held-per-kind", type=int, default=96); ap.add_argument("--shares", default="")
    ap.add_argument("--mix", default="v22", choices=["v22", "v23"], help="v23: the any-length kind, look-alike facts, facts told and stored, varied memory budget")
    ap.add_argument("--parts", default=None, help="v2.3 changes to use, comma list of longconv,crowd,stored,budget,longdocs,mem1024,memqa,realev,alpacabury,alpacaqa,afterno,twomem,twomem1,twomem2,reason,reason15,reasoncr,multi,multi2 (default: all for --mix v23, none for v22)")
    ap.add_argument("--data-dir", default=None, help="a size folder such as data_real/5m (files there, then in its shared/ folder); default: the folders used so far")
    ap.add_argument("--memory-opts", default="", help="engine memory settings saved with the model (JSON), e.g. step 1's winner")
    ap.add_argument("--gpu-frac", type=float, default=0.0, help="cap this run's GPU memory (share of the card); over it = an error, never paging")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--no-amp", action="store_true"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.gpu_frac and dev.type == "cuda":                            # 17 Oct: two runs side by side must not page the card
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    amp = not a.no_amp
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    if a.smoke:                       # tiny settings, unless given explicitly (the plateau test passes its own)
        dflt = ap.parse_args(["--name", "x"])
        for k, v in {"min_steps": 20, "max_steps": 40, "eval_every": 10, "held_per_kind": 8}.items():
            if getattr(a, k) == getattr(dflt, k):
                setattr(a, k, v)
    core = torch.load(ROOT / a.core, map_location="cpu", weights_only=False)
    dict_path = core["encoding"]["dictionary"].replace("\\", "/")   # saved on Windows; also opens on Linux and macOS
    enc = E.load(ROOT / dict_path)
    v23 = a.mix == "v23"
    parts = set(a.parts.split(",")) - {""} if a.parts is not None else ({"longconv", "crowd", "stored", "budget", "mem1024"} if v23 else set())
    mem_codes, reply_room = (a.ctx // 2 if "mem1024" in parts else a.ctx // 4), min(300, a.ctx // 3)
    shares = dict(SHARES_V23 if v23 else SHARES); shares.update(json.loads(a.shares) if a.shares else {})
    shares = {k: v for k, v in shares.items() if v > 0}
    kinds = list(shares); probs = np.array([shares[k] for k in kinds]); probs /= probs.sum()
    say("effective shares: " + ", ".join(f"{k} {p:.3f}" for k, p in zip(kinds, probs)))   # 17 Oct: kinds left out of --shares keep their defaults
    train_w = Windows(enc, a.ctx, mem_codes, reply_room, rng, "train", small=a.smoke, v23=v23, parts=parts, data_dir=a.data_dir)
    # the held-out windows are the same for every seed (16 Oct: they depended on the seed, so curves were not comparable)
    held_w = Windows(enc, a.ctx, mem_codes, reply_room, np.random.default_rng(10_000), "heldout", small=a.smoke, v23=v23, parts=parts, data_dir=a.data_dir)
    held = {k: [held_w.make(k) for _ in range(a.held_per_kind)] for k in kinds}
    say(f"windows ready: {len(train_w.qa):,} Q&A, {len(train_w.skills):,} skills, {len(train_w.dialogues):,} dialogues, "
        f"{len(train_w.rehearse_codes):,} rehearsal codes; held-out {a.held_per_kind} per kind")
    # the n-gram tables come from the Q&A training text (as before), and are saved beside the model
    qa_file = (ROOT / "data_qa" / "train" / "qa_train.txt" if "alpacaqa" in parts
               else train_w.paths.find("qa/qa_train.txt", ROOT / "data_qa" / "train" / "qa_train.txt"))
    qa_codes = enc.encode_file(str(qa_file), str(ROOT / "runs" / "chat22_cache"))[0]
    tables = [NgramTable(qa_codes, o, 64) for o in (2, 3)]
    model = ChatLM(enc.vocab, core["model"]["width"], core["model"]["layers"], core["model"]["heads"], a.ctx, tables, a.rope_scale)
    model.load(core["state_dict"], core_only=True); model.to(dev).train()
    heads = {n for n, _ in model.named_parameters() if n.split(".")[0] in ("copy_q", "copy_k", "mix_head")}
    groups = []
    for is_head in (False, True):
        ps = [(n, p) for n, p in model.named_parameters() if (n in heads) == is_head]
        lr = a.head_lr if is_head else a.lr
        groups += [{"params": [p for n, p in ps if p.dim() >= 2], "weight_decay": 0.1, "peak": lr},
                   {"params": [p for n, p in ps if p.dim() < 2], "weight_decay": 0.0, "peak": lr}]
    opt = torch.optim.AdamW([g for g in groups if g["params"]], lr=a.lr, betas=(0.9, 0.95))
    params = [p for g in groups for p in g["params"]]
    cfg = {"vocab": enc.vocab, "width": core["model"]["width"], "layers": core["model"]["layers"], "heads": core["model"]["heads"],
           "ctx": a.ctx, "rope_scale": a.rope_scale, "dict": dict_path,
           "facts": (str(train_w.paths.find("knowledge/knowledge_real.jsonl", ROOT / "data_chat22" / "knowledge_wiki.jsonl").relative_to(ROOT)).replace("\\", "/") if a.data_dir
                     else "data_chat22/facts_pool.jsonl" if v23 else "data_qa/train/qa_train.txt"),
           "dense": 2.0, "parts": sorted(parts), "data_dir": a.data_dir,
           "memory_opts": json.loads(a.memory_opts) if a.memory_opts else None, "mix": a.mix, "lr": a.lr, "batch": a.batch,
           "mem_codes": mem_codes, "reply_room": reply_room, "layout": "chat22", "core": a.core, "shares": shares,
           "aux": a.aux, "point": a.point, "seed": a.seed}
    state_p = out / f"{a.name}.state.pt"
    st = {"step": 0, "log": [], "best": None, "decay_from": None, "decay_len": None, "base": None, "train_loss": {}}
    if state_p.exists() and not a.smoke:                              # resume
        s = torch.load(state_p, map_location="cpu", weights_only=False)
        model.load(s["model"]); opt.load_state_dict(s["opt"]); st = s["st"]; rng.bit_generator.state = s["rng"]
        say(f"resumed at step {st['step']}")
    t0 = time.time(); run = {k: [0.0, 0] for k in kinds}
    while True:
        step = st["step"]
        # learning rate: warmup, stable, then decay once the plateau is found
        warm = min(1.0, (step + 1) / 200)
        if st["decay_from"] is None:
            f = warm
        else:
            prog = min(1.0, (step - st["decay_from"]) / max(1, st["decay_len"]))
            f = 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * prog))
        for g in opt.param_groups:
            g["lr"] = g["peak"] * f
        kind = kinds[int(rng.choice(len(kinds), p=probs))]
        ws = [train_w.make(kind) for _ in range(a.batch)]
        if a.accum <= 1:
            nll, aux, pt, _, _ = losses(model, ws, dev, a.aux, a.point, amp)
            loss = nll + a.aux * aux + a.point * pt
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            run[kind][0] += float(nll.detach()); run[kind][1] += 1
        else:                                             # the same windows in a.accum parts, one update
            opt.zero_grad(set_to_none=True); part_nll = 0.0
            n = len(ws)
            for j in range(a.accum):
                part = ws[j * n // a.accum:(j + 1) * n // a.accum]
                if not part:
                    continue
                nll, aux, pt, _, _ = losses(model, part, dev, a.aux, a.point, amp)
                ((nll + a.aux * aux + a.point * pt) * (len(part) / n)).backward()
                part_nll += float(nll.detach()) * len(part) / n
            torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
            run[kind][0] += part_nll; run[kind][1] += 1
        st["step"] = step = step + 1
        done = st["decay_from"] is not None and step >= st["decay_from"] + st["decay_len"]
        if step % a.eval_every == 0 or done or step >= a.max_steps:
            ev = evaluate(model, held, dev, amp)
            if st["base"] is None:
                st["base"] = {k: v["loss"] for k, v in ev.items()}
            comb = float(np.mean([ev[k]["loss"] / st["base"][k] for k in ev]))
            tr = {k: round(v[0] / v[1] / math.log(2), 4) for k, v in run.items() if v[1]}
            alarms = []
            for k in ev:                                  # training loss down, held-out loss up, twice running
                hist = [r for r in st["log"][-2:]]
                if len(hist) == 2 and k in tr and all(k in h["train"] for h in hist):
                    if tr[k] < hist[-1]["train"][k] < hist[-2]["train"][k] and ev[k]["loss"] > hist[-1]["held"][k]["loss"] > hist[-2]["held"][k]["loss"]:
                        alarms.append(k)
            row = {"step": step, "s": round(time.time() - t0), "gpu_peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if dev.type == "cuda" else 0, "lr_factor": round(f, 3), "combined": round(comb, 4), "train": tr, "held": ev,
                   "overfit_alarm": alarms}
            st["log"].append(row); run = {k: [0.0, 0] for k in kinds}
            improved = st["best"] is None or comb < st["best"]["combined"] * (1 - 1e-4)
            if improved:
                st["best"] = {"step": step, "combined": comb}
                torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "config": cfg,
                            "train": {"best": st["best"], "log": st["log"]}}, out / f"{a.name}.pt")
                save_tables(tables, out / f"{a.name}.tables.npz")
            # plateau: after min_steps, no gain over 0.5% across the last `patience` checks -> decay over 20% of steps so far
            if st["decay_from"] is None and step >= a.min_steps and len(st["log"]) > a.patience:
                ref = st["log"][-a.patience - 1]["combined"]
                if min(r["combined"] for r in st["log"][-a.patience:]) > ref * (1 - 0.005):
                    st["decay_from"], st["decay_len"] = step, max(a.eval_every, int(0.2 * step))
                    say(f"plateau at step {step}: decaying over {st['decay_len']} steps")
            if st["decay_from"] is None and step >= a.max_steps * 0.8:
                st["decay_from"], st["decay_len"] = step, a.max_steps - step
            say(f"step {step} combined {comb:.4f}{' best' if improved else ''} | " +
                " ".join(f"{k} {v['loss']:.3f}/{v['reply_sure']}" for k, v in ev.items()) + (f" | ALARM {alarms}" if alarms else ""))
            if not a.smoke:
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "st": st, "rng": rng.bit_generator.state}, state_p)
            if done or step >= a.max_steps:
                break
    ck = torch.load(out / f"{a.name}.pt", map_location="cpu", weights_only=False)
    ck["train"].update({"steps_run": st["step"], "train_s": round(time.time() - t0), "decay_from": st["decay_from"]})
    torch.save(ck, out / f"{a.name}.pt")
    (ROOT / "results" / f"chat22_train_{a.name}.json").write_text(json.dumps({"name": a.name, "config": cfg, **ck["train"]}, indent=1))
    say(f"done: best step {st['best']['step']} (combined {st['best']['combined']:.4f}), {st['step']} steps, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
