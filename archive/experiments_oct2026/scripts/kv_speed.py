"""KV-cache speed check (Traditional Hybrid T1): the chat probe's 32 questions answered with and
without the cache; replies must be identical, time is reported. Writes results/kv_speed_<model>.json.

    python compiler/experiments/kv_speed.py runs/s2c_50m/qa_s2c_50m.pt --device cuda
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import chat_probe  # noqa: E402
import m2_qa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("model"); ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    native = ROOT / "data" / "cv0.exe"
    eng = m2_qa.QAEngine(a.model, native=str(native) if native.exists() else None, device=a.device, log=lambda m: None)
    qs = [q for _, q, _ in chat_probe.PROBES]
    times, same = {False: 0.0, True: 0.0}, 0
    for q in qs:
        outs = {}
        for fast in (False, True):
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            t = time.perf_counter()
            outs[fast] = eng.ask(q.encode(), 120, 3, fast=fast)[0]
            if a.device.startswith("cuda"):
                torch.cuda.synchronize()
            times[fast] += time.perf_counter() - t
        same += outs[False] == outs[True]
    res = {"model": a.model, "device": a.device, "questions": len(qs), "identical": same,
           "seconds_without_cache": round(times[False], 2), "seconds_with_cache": round(times[True], 2),
           "speedup": round(times[False] / max(times[True], 1e-9), 2)}
    (ROOT / "results" / f"kv_speed_{Path(a.model).stem}.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
