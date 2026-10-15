"""Self-training round for the 5M chat model v2 (arm D; 15 October 2026): the small, verifiable form of the outcome-
rewarded training behind recent reasoning models (o1, R1): the model answers training questions whose right answer is
known, several times; answers that are both CORRECT (the target's key words) and SUPPORTED (every content word found in
the evidence, the earlier turns or the question) are kept as new targets; questions with no good answer keep their
original target. train_chat.py --extra-data then fine-tunes on them beside the normal mix.

    python compiler/experiments/selftrain_chat.py runs/chat5m_v2/chat5m_v2_B.pt --n 2500 --k 4 --out runs/chat5m_v2/selftrain.jsonl

Reports how often the greedy answer is correct, and how often any of the k answers is correct and supported.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import chat_engine as CE  # noqa: E402

ABSTAIN = ("don't have", "do not have", "not in my memory", "could you tell me")
VERIFIABLE = {"single", "nearmiss", "update", "update_earlier", "combo", "short", "yesno", "list", "unknown", "followup"}


def correct(kind, target: str, ans: str, q: str) -> bool:
    lo = ans.lower()
    if kind == "unknown":
        return any(p in lo for p in ABSTAIN)
    if kind == "yesno" and not lo.startswith(target[:2].lower()):
        return False
    need = CE.content_words(target.encode()) - CE.content_words(q.encode())
    return bool(need) and need <= CE.content_words(ans.encode())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--n", type=int, default=2500); ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--data", default="data_skills/train_v4c.jsonl"); ap.add_argument("--out", default="runs/chat5m_v2/selftrain.jsonl")
    ap.add_argument("--device", default="cuda"); ap.add_argument("--seed", type=int, default=9)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    eng = CE.ChatEngine(str(ROOT / a.model), native=str(ROOT / "data" / "cv0.exe"), device=a.device, log=lambda *x: None)
    lines = (ROOT / a.data).read_text(encoding="utf-8").splitlines()
    rnd.shuffle(lines)
    items = [w for w in (json.loads(l) for l in lines) if w["kind"] in VERIFIABLE][:a.n]
    gen = torch.Generator().manual_seed(a.seed)
    B = lambda x: x.encode("utf-8", "replace") if isinstance(x, str) else x
    out, p1, pk, t0, kept_own = [], 0, 0, time.time(), 0
    for w in items:
        cands = [(B(c[0]), B(c[1])) for c in w["cands"]]
        turns = [(B(t[0]), B(t[1])) for t in w["turns"]]
        prompt, copy_end, _, _ = CE.build_window(eng.enc, eng.sep, cands, turns, B(w["q"]), None, eng.a.ctx, eng.ev_codes, 80)
        pool = set().union(*[CE.content_words(q + b" " + r) for q, r in cands + turns]) if cands or turns else set()
        pool |= CE.content_words(B(w["q"]))
        good = []
        for j in range(a.k):
            o = eng._generate(prompt, copy_end, 0.0, 80, 3, None, 0.0 if j == 0 else 0.8, 20, gen)
            ans = o[0].decode("utf-8", "replace")
            ok = correct(w["kind"], w["target"], ans, w["q"])
            aw = CE.content_words(o[0])
            sup = w["kind"] == "unknown" or (aw <= pool if aw else False)
            if j == 0:
                p1 += ok
            if ok and sup:
                good.append(ans)
        pk += bool(good)
        if good:                      # the model's own correct, supported phrasing becomes the target
            tgt = good[0]
        else:                         # no good attempt: keep the original target
            tgt = w["target"]; kept_own += 1
        out.append({**w, "target": tgt, "kind": w["kind"], "self": bool(good)})
    Path(ROOT / a.out).write_text("".join(json.dumps(x) + "\n" for x in out), encoding="utf-8")
    rep = {"model": a.model, "questions": len(items), "k": a.k, "greedy_answer_correct": round(p1 / len(items), 3), "any_answer_correct_and_supported": round(pk / len(items), 3),
           "original_target_kept": kept_own, "seconds": round(time.time() - t0)}
    (ROOT / "results" / f"selftrain_{Path(a.model).stem}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
