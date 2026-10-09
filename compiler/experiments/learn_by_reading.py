"""Learning by reading: does adding a book to the count tables, with no retraining, make the
model better on another book by the same author?

    python compiler/experiments/learn_by_reading.py DATA WORK --author Austen --test Emma_158 \
        --add Pride-and-Prejudice_1342 --control <some non-Austen book not in training> --budget 240

Steps: build a training set with every book of the author removed; train the best
configuration (v0 8k + exact n-gram tables, 64 followers) on it; score the test book. Then
recount the tables with the added book included (the network untouched) and score again. A
control adds an unrelated book of similar size instead, to show that it is the author's
text, not extra counts in general, that helps.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder1  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import train_lm  # noqa: E402


def retune_head(model, tokens: np.ndarray, seconds: float, device, amp: bool, log, batch: int = 32, lr: float = 1e-3) -> None:
    """Train the trust head alone (every other parameter frozen) on the new stream with the new
    tables, leave-one-out as in training: the cheap way to absorb an addition."""
    import torch
    for p in model.parameters():
        p.requires_grad_(False)
    for p in model.mix_head.parameters():
        p.requires_grad_(True)
    opt = torch.optim.AdamW(model.mix_head.parameters(), lr=lr, weight_decay=0.0)
    rng = np.random.default_rng(0)
    ctx, n, offsets = model.ctx, len(tokens), np.arange(model.ctx + 1)
    model.train()
    t_end, steps = time.perf_counter() + seconds, 0
    while time.perf_counter() < t_end:
        idx = rng.integers(0, n - ctx - 1, size=batch)
        win = tokens[idx[:, None] + offsets]
        x = torch.from_numpy(win[:, :-1]).to(device); y = torch.from_numpy(win[:, 1:]).to(device)
        with torch.autocast(device_type="cuda" if str(device).startswith("cuda") else "cpu", dtype=torch.bfloat16, enabled=amp):
            loss = model.nll(x, y).mean()
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); steps += 1
    model.eval()
    for p in model.parameters():
        p.requires_grad_(True)
    log(f"trust head re-tuned: {steps} steps in {seconds:.0f}s, last loss {float(loss):.3f} nats")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("work"); ap.add_argument("--native", default=None)
    ap.add_argument("--author", required=True, help="substring that marks the author's books, e.g. a title list file or a name")
    ap.add_argument("--books", required=True, help="comma-separated stems of the author's books in DATA/train")
    ap.add_argument("--test", required=True); ap.add_argument("--add", required=True); ap.add_argument("--control", required=True)
    ap.add_argument("--budget", type=float, default=240); ap.add_argument("--topk", type=int, default=64)
    ap.add_argument("--out", default="results/learn_by_reading.json")
    ap.add_argument("--device", default="cpu"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--retune-s", type=float, default=0, help="after each addition, re-tune the trust head alone for this many seconds on the new stream")
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    author_books = set(a.books.split(","))
    # Training set without the author; the test book is scored as held-out.
    sub = work / "data_no_author"
    for split in ("train", "heldout", "ood"):
        (sub / split).mkdir(parents=True, exist_ok=True)
    train_paths = []
    for p in sorted(glob.glob(f"{a.data}/train/*.txt")):
        if Path(p).stem not in author_books:
            shutil.copy(p, sub / "train" / Path(p).name); train_paths.append(str(sub / "train" / Path(p).name))
    test_path = f"{a.data}/train/{a.test}.txt"
    shutil.copy(test_path, sub / "heldout" / f"{a.test}.txt")
    for p in glob.glob(f"{a.data}/ood/*.txt"):
        shutil.copy(p, sub / "ood" / Path(p).name)
    print(f"training set: {len(train_paths)} books without {sorted(author_books)}; test book {a.test}", flush=True)
    enc = LE.V0Dict(str(ladder1.fit_dictionary("v0_8k_plain", train_paths, work)), a.native)
    tokens, lens = train_lm.load_training(enc, str(sub), work)
    log = lambda m: print(time.strftime("%H:%M:%S"), m, flush=True)
    r = train_lm.run(enc, tokens, lens, train_lm.eval_sets_from(str(sub)), work, a.budget, ngram_orders=(2, 3),
                     ngram_topk=a.topk, eval_every_s=60, log=log, keep_model=True, save_path=str(work / "learn_by_reading.pt"),
                     width=a.width, layers=a.layers, heads=a.heads, device=a.device, amp=a.amp)
    model = r.pop("_model")
    n_train = int(len(tokens) * 0.98)
    base_tokens = tokens[:n_train]
    out = {"setup": {"author_books_removed": sorted(author_books), "test": a.test, "add": a.add, "control": a.control,
                     "budget_s": a.budget, "topk": a.topk, "train_books": len(train_paths)},
           "before": r["eval"]["heldout"][a.test]["bits_per_byte"], "training": r["training"], "model": r["model"]}
    print(f"before: {out['before']:.4f} bits/byte on {a.test}", flush=True)
    head_state = {k: v.clone() for k, v in model.mix_head.state_dict().items()}
    for label, stems in (("after_adding_author_book", a.add), ("after_adding_control_book", a.control)):
        # one stem or several, comma-separated: stems in DATA/train, or paths
        paths = [s if s.endswith(".txt") else f"{a.data}/train/{s}.txt" for s in stems.split(",")]
        extra = np.concatenate([enc.encode_file(p, work)[0] for p in paths])
        stream = np.concatenate([base_tokens, extra])
        model.mix_head.load_state_dict(head_state)          # every addition starts from the trained head
        model.ngrams = [train_lm.NgramTable(stream, o, a.topk) for o in (2, 3)]
        row = train_lm.eval_file(model, enc, test_path, work)
        out[label] = {"books": [Path(p).stem for p in paths], "bytes_added": int(sum(Path(p).stat().st_size for p in paths)),
                      "bits_per_byte": row["bits_per_byte"]}
        print(f"{label} ({stems}): {row['bits_per_byte']:.4f}", flush=True)
        if a.retune_s:
            retune_head(model, stream, a.retune_s, a.device, a.amp, log)
            row = train_lm.eval_file(model, enc, test_path, work)
            out[label]["bits_per_byte_after_retune"] = row["bits_per_byte"]
            print(f"{label} + trust head re-tuned {a.retune_s:.0f}s: {row['bits_per_byte']:.4f}", flush=True)
    out["change_author_pct"] = round(100 * (out["after_adding_author_book"]["bits_per_byte"] - out["before"]) / out["before"], 2)
    out["change_control_pct"] = round(100 * (out["after_adding_control_book"]["bits_per_byte"] - out["before"]) / out["before"], 2)
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k in ("before", "change_author_pct", "change_control_pct")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
