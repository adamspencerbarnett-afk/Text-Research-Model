"""Talk to a ladder model through its encoder: the prompt is compiled to IDs, the model
continues the IDs, and the same compiler turns them back into text.

    python compiler/experiments/chat.py WORK/ladder1_v0_8k_plain.pt --prompt "We went to the park"
    python compiler/experiments/chat.py WORK/ladder1_bytes.pt            # interactive

The checkpoint stores the encoding description and model shape beside the weights, so the
right encoder is rebuilt from the work directory (dictionary file or hash reverse map).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder_encodings as LE  # noqa: E402
import train_lm  # noqa: E402


def load(path: str, native: str | None = None):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info, shape, work = ck["encoding"], ck["model"], Path(path).parent
    if info["name"] == "bytes":
        enc = LE.Bytes()
    elif info["groups"]:
        enc = LE.HashCodes.load(str(work / f"{info['name']}.json"))
    else:
        enc = LE.V0Dict(info["dictionary"], native)
    valid = np.fromiter(enc.reverse.keys(), dtype=np.int64, count=len(enc.reverse)) if isinstance(enc, LE.HashCodes) else None
    model = train_lm.LM(enc, shape["width"], shape["layers"], shape["heads"], shape["ctx"], shape["table_rows"], valid)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or any(not k.startswith("table") for k in missing):
        raise ValueError(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    model.eval()
    return enc, model


def reply(enc, model, prompt: bytes, n_tokens: int, temperature: float, top_k: int, seed: int) -> tuple[bytes, dict]:
    gen = torch.Generator().manual_seed(seed)
    p_tokens, _ = enc.encode(prompt)
    seq = [int(t) for t in p_tokens]
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        seq.append(model.sample_next(torch.tensor([seq[-model.ctx:]]), gen, temperature, top_k))
    dt = time.perf_counter() - t0
    out = enc.decode(seq[len(p_tokens):])
    return out, {"prompt_tokens": len(p_tokens), "tokens": n_tokens, "seconds": round(dt, 2),
                 "tokens_per_s": round(n_tokens / dt, 1), "bytes_per_s": round(len(out) / dt, 1)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--native"); ap.add_argument("--prompt")
    ap.add_argument("--tokens", type=int, default=120); ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-k", type=int, default=40); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    enc, model = load(a.checkpoint, a.native)
    print(f"encoder {enc.name}: {len(enc.encode(b'We went to the park')[0])} IDs for 'We went to the park'")
    prompts = [a.prompt] if a.prompt else iter(lambda: input("you> "), "")
    for p in prompts:
        out, stats = reply(enc, model, p.encode("utf-8"), a.tokens, a.temperature, a.top_k, a.seed)
        print(f"model> {out.decode('utf-8', 'replace')}\n       [{stats}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
