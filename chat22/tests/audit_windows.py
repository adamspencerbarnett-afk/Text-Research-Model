"""Audit of the chat22 training windows: for every kind, N windows: lengths (none may exceed ctx + 1), reply codes,
how many have an evidence span (gold) and whether it holds the reply's content words, how many have a pointer
target; two decoded examples per kind.   python -m chat22.tests.audit_windows [--n 300] [--split train|heldout]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from chat22 import encoder as E
from chat22.windows_train import Windows, words

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300); ap.add_argument("--split", default="train"); ap.add_argument("--ctx", type=int, default=2048)
    ap.add_argument("--v23", action="store_true", help="the v2.3 windows (any-length kind, look-alike facts, told and stored facts)")
    a = ap.parse_args()
    core = torch.load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
    enc = E.load(ROOT / core["encoding"]["dictionary"])
    ws = Windows(enc, a.ctx, a.ctx // 2 if a.v23 else a.ctx // 4, min(300, a.ctx // 3), np.random.default_rng(3), a.split, small=True, v23=a.v23)
    rep = {}
    for kind in [k for k in Windows.MAKERS if a.v23 or k != "longconv"]:
        lens, reply, gold_n, gold_ok, point_n = [], [], 0, 0, 0
        for i in range(a.n):
            w = ws.make(kind)
            lens.append(len(w.codes)); reply.append(len(w.codes) - w.prompt_len)
            if i < 2 and kind != "rehearse":
                dec = lambda c: enc.decode(c.tolist()).decode("utf-8", "replace").replace("\n", " / ")
                print(f"--- {kind} #{i + 1}: {len(w.codes)} codes, copyable {w.copy_end}")
                print("  MEMORY+MSG:", dec(w.codes[max(0, w.copy_end - 160):w.prompt_len])[-300:])
                print("  REPLY:     ", dec(w.codes[w.prompt_len:])[:160])
                print("  GOLD:      ", " | ".join(dec(w.codes[s:e])[:90] for s, e in w.gold)[:220] or "(none)",
                      "  POINT:", " | ".join(dec(w.codes[s:e])[:40] for s, e in w.point)[:80] or "(none)")
            if w.gold:
                gold_n += 1
                need = words(enc.decode(w.codes[w.prompt_len:].tolist())) - {"remember", "got", "noted", "continue", "story"}
                g = set().union(*[words(enc.decode(w.codes[s:e].tolist())) for s, e in w.gold])
                gold_ok += bool(need) and len(need & g) >= max(1, len(need) // 2)
            point_n += bool(w.point)
        rep[kind] = {"codes_p50_p95_max": [int(np.percentile(lens, 50)), int(np.percentile(lens, 95)), int(max(lens))],
                     "over_ctx": sum(x > a.ctx + 1 for x in lens), "reply_p50": int(np.percentile(reply, 50)),
                     "no_reply": sum(x <= 0 for x in reply), "with_gold": gold_n, "gold_holds_reply": gold_ok, "with_point": point_n}
        print("audit", kind, rep[kind], flush=True)
    (ROOT / "results" / f"chat22_window_audit_{a.split}{'_v23' if a.v23 else ''}.json").write_text(json.dumps(rep, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
