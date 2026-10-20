"""Equivalence test (docs/V22_CLEAN_DESIGN.md section 5, step 1): the clean ChatLM must compute exactly what the old
code computes, on the same weights and windows. Any difference is a bug in chat22.

  1. training loss: ChatLM.score vs train_chat.nll_and_aux (nll, aux / has, point / has_point), train mode (loo = 1)
  2. eval loss: same with loo = 0
  3. answering: ChatLM.next_logprobs vs V2LM.mixed_logits at several positions of real windows

    python -m chat22.tests.test_equivalence [--model runs/chat5m_v2/chat5m_v2_A.pt]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments")); sys.path.insert(0, str(ROOT / "compiler")); sys.path.insert(0, str(ROOT))
import chat_engine as CE  # noqa: E402
import m2_qa  # noqa: E402
import train_chat as TC  # noqa: E402

from chat22.model import ChatLM  # noqa: E402


def windows(eng, n, rng):
    """Real chat windows (old v2 layout): stored exchanges as evidence, held-out-style turns, a question and answer."""
    ex = eng.mem.exchanges
    out = []
    for _ in range(n):
        cands = [ex[int(rng.integers(len(ex)))] for _ in range(int(rng.integers(1, 4)))]
        turns = [ex[int(rng.integers(len(ex)))] for _ in range(int(rng.integers(0, 4)))]
        q, a = ex[int(rng.integers(len(ex)))]
        sp = {}
        codes, copy_end, pl, kept = CE.build_window(eng.enc, eng.sep, cands, turns, q, a, eng.a.ctx, eng.ev_codes, spans=sp)
        out.append((codes[:eng.a.ctx + 1], copy_end, sp))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/chat5m_v2/chat5m_v2_A.pt"); ap.add_argument("--n", type=int, default=6)
    a = ap.parse_args()
    torch.manual_seed(0); rng = np.random.default_rng(0)
    eng = CE.ChatEngine(str(ROOT / a.model), native=str(ROOT / "data" / "cv0.exe"), device="cpu", log=lambda *x: None)
    old = eng.model.float()
    new = ChatLM(old.vocab, old.emb.weight.shape[1], len(old.blocks), old.blocks[0].heads, eng.a.ctx, old.ngrams)
    new.load(old.state_dict())
    worst = {}

    def note(k, d):
        worst[k] = max(worst.get(k, 0.0), float(d))

    for train in (True, False):
        old.train(train); new.train(train)
        for codes, copy_end, sp in windows(eng, a.n, rng):
            x = torch.from_numpy(codes[None, :-1].astype(np.int64)); y = torch.from_numpy(codes[None, 1:].astype(np.int64))
            src = torch.zeros_like(x, dtype=torch.bool); src[:, :copy_end] = True
            gold = torch.zeros_like(src); point = torch.zeros_like(src)
            if sp.get("cands"):
                s0, e0 = sp["cands"][0]; gold[:, s0:e0] = True
                s1, e1 = sp["cands"][-1]; point[:, s1:e1] = True
            with torch.no_grad():
                nll, aux, has, aux_p, has_p = TC.nll_and_aux(old, x, y, src, gold, True, point)
                r = new.score(x, y, src, gold, point)
            tag = "train" if train else "eval"
            note(f"{tag} nll", (nll - r["nll"]).abs().max())
            note(f"{tag} aux (where defined)", ((aux - r["aux"]).abs() * has).max())
            note(f"{tag} point (where defined)", ((aux_p - r["point"]).abs() * has_p).max())
            note(f"{tag} has / has_point flags differ", (has != r["has"]).sum() + (has_p != r["has_point"]).sum())
    old.eval(); new.eval()
    for codes, copy_end, _ in windows(eng, a.n, rng):
        for cut in (copy_end + 3, len(codes) - 1):
            seq = torch.from_numpy(codes[None, :cut].astype(np.int64))
            with torch.no_grad():
                keys = old.hidden(seq); old._keys = keys
                span = torch.arange(cut)[None] < copy_end
                ref = old.mixed_logits(keys[:, -1], seq, span, torch.zeros(1))
                got = new.next_logprobs(keys[:, -1], keys, seq, copy_end)
            note("answer next-code log-probs", (ref - got).abs().max())
            note("answer greedy code differs", int(ref.argmax(-1) != got.argmax(-1)))
    print(f"model {a.model}: {new.param_counts()}")
    ok = True
    for k, v in worst.items():
        good = v < (1e-4 if "log" in k or "nll" in k or "aux" in k or "point" in k else 0.5)
        ok &= good
        print(f"{'SAME' if good else 'DIFF'}  {k:40s} max difference {v:.2e}")
    print("EQUIVALENT" if ok else "NOT EQUIVALENT")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
