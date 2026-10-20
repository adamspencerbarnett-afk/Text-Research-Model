"""Stage 3 of the merged model: grounding (docs/CHAT5M_V2_DESIGN.html#v24; Adam, 17 Oct 2026). On top of a stage-2 chat
model, preference pairs graded automatically by our generators teach the balance that example answers alone did not:

  told       a fact said earlier (in the window, or retrieved as a [turn] item)   answer          over  "I don't have that"
  never      not told, a look-alike told                                          "I don't know"  over  the look-alike's value
  corrected  a fact, then "Actually, ..."                                         the new value   over  the old value
  evidence   real Q&A with its paragraph (SQuAD, HotpotQA, SciQ, BoolQ)           the answer      over  a neighbour's answer or a refusal

Loss: direct preference optimisation against a frozen copy of the stage-2 model (beta 0.1), plus the stage-2 mix replayed
with its own loss so nothing fades. Kept: the best held-out preference accuracy whose replay loss stays within 2% of the
start. The suite and the guard decide whether stage 3 is kept at all.

    python -m chat22.ground --source runs/chat22/real_5m_s1.pt --name real_5m_s1_g [--steps 3000] [--device cuda]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from . import encoder as E
from . import window as W
from .model import ChatLM
from .tables import load_tables, save_tables
from .train import batch_tensors, losses
from .windows_train import ACKS, B, G, G4, Window, Windows

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("told", "never", "corrected", "evidence")


class Pairs:
    def __init__(self, ws: Windows, rng):
        self.ws, self.rng = ws, rng
        self.passage = [i for i in ws.usable if ws.qa[i][0].startswith(b"Passage:")]

    def _win(self, items, turns, q, ans):
        w = self.ws
        codes, ce, pl, kept = W.build(w.enc, w.sep, items, turns, B(q), B(ans), w.ctx, w.mem_codes, w.reply_room)
        return Window(codes[:w.ctx + 1], pl, ce), kept

    def _pair(self, items, turns, q, good, bad):
        if not good or not bad or good == bad:
            return None
        a, k1 = self._win(items, turns, q, good)
        b, k2 = self._win(items, turns, q, bad)
        if k1 != k2 or a.prompt_len != b.prompt_len or not np.array_equal(a.codes[:a.prompt_len], b.codes[:b.prompt_len]):
            return None                                          # the two answers must share one context exactly
        return a, b

    def _turns_with(self, statements):
        rng = self.rng
        turns = self.ws._filler(int(rng.integers(1, 12)))
        for st in statements:
            turns.insert(int(rng.integers(0, len(turns) + 1)), (B(st), B(G.pick(rng, ACKS))))
        return turns

    def told(self):
        rng = self.rng
        f = G.Fact(rng)
        st = f.stored()[1]
        turns = self._turns_with([st])
        items = []
        if rng.random() < 0.5:                                   # retrieved from conversation memory
            j = next(i for i, t in enumerate(turns) if t[0] == B(st))
            items = [W.Item(turns[j][0], turns[j][1], "turn", j + 1)]
            turns = turns[j + 1:]
        return self._pair(items, turns, f.ask(), G4.your(st), G.pick(rng, G.ABSTAIN))

    def never(self):
        rng = self.rng
        f = G.Fact(rng)
        d = G.Fact(rng, f.kind, f.field)
        if d.ent["id"] in f.ent["id"] or f.ent["id"] in d.ent["id"] or d.value == f.value:
            return None
        turns = self._turns_with([d.stored()[1]])
        return self._pair([], turns, f.ask(), G.pick(rng, G.ABSTAIN), G4.your(f._f(f.spec[1], d.value)))

    def corrected(self):
        rng = self.rng
        f = G.Fact(rng)
        if not f.can_update():
            return None
        st = f.stored()[1]
        (_, new_s), _ = f.update()
        turns = self._turns_with([st])
        j = next(i for i, t in enumerate(turns) if t[0] == B(st))
        turns.insert(int(rng.integers(j + 1, len(turns) + 1)), (B("Actually, " + new_s[0].lower() + new_s[1:]), B(G.pick(rng, ACKS))))
        return self._pair([], turns, f.ask(), G4.your(new_s), G4.your(st))

    def evidence(self):
        rng, w = self.rng, self.ws
        if not self.passage:
            return None
        i = self.passage[int(rng.integers(len(self.passage)))]
        q, a = w.qa[i]
        bad = None
        if w.nbrs is not None and rng.random() < 0.5:
            bad = w.qa[int(w.nbrs[i, 0])][1]
        bad = bad if (bad and bad.strip() and bad != a) else G.pick(rng, G.ABSTAIN).encode()
        return self._pair([], [], q, a, bad)

    def make(self, kind):
        for _ in range(8):
            p = getattr(self, kind)()
            if p:
                return p
        return None


def answer_logp(model, wins, dev, amp):
    x, y, m, s, g, p, lo = batch_tensors(wins, dev)
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
        r = model.score(x, y, s, None, None, torch.zeros_like(lo))
    return -(r["nll"].float() * m).sum(1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", default="runs/chat22")
    ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--pairs", type=int, default=6); ap.add_argument("--replay", type=int, default=6)
    ap.add_argument("--beta", type=float, default=0.1); ap.add_argument("--lr", type=float, default=2e-5); ap.add_argument("--replay-w", type=float, default=1.0)
    ap.add_argument("--eval-every", type=int, default=250); ap.add_argument("--eval-pairs", type=int, default=50)
    ap.add_argument("--device", default="cuda"); ap.add_argument("--gpu-frac", type=float, default=0.0); ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    dev = torch.device(a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu")
    if a.gpu_frac and dev.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(a.gpu_frac)
    amp = dev.type == "cuda"
    t0 = time.time()
    say = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    src = ROOT / a.source
    ck = torch.load(src, map_location="cpu", weights_only=False)
    cfg = dict(ck["config"])
    enc = E.load(ROOT / cfg["dict"])
    tables = load_tables(src.with_suffix(".tables.npz"))
    def new_model():
        m = ChatLM(cfg["vocab"], cfg["width"], cfg["layers"], cfg["heads"], cfg["ctx"], tables, cfg.get("rope_scale", 1.0))
        m.load(ck["state_dict"]); return m.to(dev)
    policy, ref = new_model().train(), new_model().eval()
    for p_ in ref.parameters():
        p_.requires_grad_(False)
    parts = set(cfg.get("parts") or [])
    rng = np.random.default_rng(a.seed)
    mk = lambda split, r: Windows(enc, cfg["ctx"], cfg["mem_codes"], cfg["reply_room"], r, split, small=a.smoke,
                                  v23=cfg.get("mix") == "v23", parts=parts, data_dir=cfg.get("data_dir"))
    train_w, held_w = mk("train", rng), mk("heldout", np.random.default_rng(10_000))
    pairs, held_pairs = Pairs(train_w, rng), Pairs(held_w, np.random.default_rng(20_000))
    shares = cfg["shares"]; kinds = list(shares); probs = np.array([shares[k] for k in kinds]); probs /= probs.sum()
    held_pairs_fixed = {k: [p for p in (held_pairs.make(k) for _ in range(a.eval_pairs)) if p] for k in KINDS}
    held_replay = [held_w.make(k) for k in kinds for _ in range(4)]

    @torch.no_grad()
    def evaluate():
        policy.eval()
        acc = {}
        for k, ps in held_pairs_fixed.items():
            ok = 0
            for i in range(0, len(ps), 8):
                ch = ps[i:i + 8]
                lc = answer_logp(policy, [x for x, _ in ch], dev, amp); lr_ = answer_logp(policy, [y for _, y in ch], dev, amp)
                ok += int((lc > lr_).sum())
            acc[k] = round(ok / len(ps), 3) if ps else None      # a kind with no pairs (no data for it) is left out
        tot = n = 0
        for i in range(0, len(held_replay), 8):
            nll, *_ = losses(policy, held_replay[i:i + 8], dev, 0, 0, amp); tot += float(nll) * len(held_replay[i:i + 8]); n += len(held_replay[i:i + 8])
        policy.train()
        return acc, tot / max(1, n)

    opt = torch.optim.AdamW([p_ for p_ in policy.parameters() if p_.requires_grad], lr=a.lr, betas=(0.9, 0.95), weight_decay=0.0)
    acc0, replay0 = evaluate()
    say(f"grounding from {src.name}: start pair accuracy {acc0}, replay loss {replay0:.4f}")
    log = [{"step": 0, "acc": acc0, "replay": round(replay0, 4)}]
    mean_of = lambda acc: float(np.mean([v for v in acc.values() if v is not None]))
    best = {"step": 0, "mean_acc": mean_of(acc0), "replay": replay0}
    out = ROOT / a.out; out.mkdir(parents=True, exist_ok=True)
    def save(step):
        c2 = dict(cfg); c2["stage"] = "grounding"; c2["grounding"] = {"source": a.source, "step": step, "beta": a.beta, "lr": a.lr}
        torch.save({"state_dict": {k: v.detach().cpu() for k, v in policy.state_dict().items()}, "config": c2,
                    "train": {"grounding_log": log}}, out / f"{a.name}.pt")
        save_tables(tables, out / f"{a.name}.tables.npz")
    save(0)
    for step in range(1, a.steps + 1):
        for g in opt.param_groups:
            g["lr"] = a.lr * min(1.0, step / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / a.steps)))
        batch = [p for p in (pairs.make(KINDS[int(rng.integers(len(KINDS)))]) for _ in range(a.pairs)) if p]
        lc = answer_logp(policy, [x for x, _ in batch], dev, amp); lr_ = answer_logp(policy, [y for _, y in batch], dev, amp)
        with torch.no_grad():
            rc = answer_logp(ref, [x for x, _ in batch], dev, amp); rr = answer_logp(ref, [y for _, y in batch], dev, amp)
        dpo = -F.logsigmoid(a.beta * ((lc - rc) - (lr_ - rr))).mean()
        rw = [train_w.make(kinds[int(rng.choice(len(kinds), p=probs))]) for _ in range(a.replay)]
        nll, aux, pt, *_ = losses(policy, rw, dev, 0.5, 0.5, amp)
        loss = dpo + a.replay_w * (nll + 0.5 * aux + 0.5 * pt)
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0); opt.step()
        if step % a.eval_every == 0 or step == a.steps:
            acc, rep = evaluate()
            mean_acc = mean_of(acc)
            keep = mean_acc > best["mean_acc"] and rep <= replay0 * 1.02
            log.append({"step": step, "acc": acc, "replay": round(rep, 4), "dpo": round(dpo.item(), 4), "kept": keep})
            say(f"step {step} pair accuracy {acc} replay {rep:.4f} (start {replay0:.4f}) dpo {dpo.item():.4f}{' best' if keep else ''}")
            if keep:
                best = {"step": step, "mean_acc": mean_acc, "replay": rep}; save(step)
    res = {"name": a.name, "source": a.source, "steps": a.steps, "best": best, "start": {"acc": acc0, "replay": replay0},
           "log": log, "seconds": round(time.time() - t0)}
    (ROOT / "results" / f"chat22_ground_{a.name}.json").write_text(json.dumps(res, indent=1))
    say(f"done: best step {best['step']} (mean pair accuracy {best['mean_acc']:.3f}, replay {best['replay']:.4f}), {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
