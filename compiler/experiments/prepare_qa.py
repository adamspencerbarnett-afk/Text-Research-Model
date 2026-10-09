"""Turn an instruction dataset into plain chat text the compiler can encode, so a model can be
trained (or continued) on it like any other text and talked to through chat.py.

    python -I compiler/experiments/prepare_qa.py ALPACA.json OUT_DIR [--max 52000]

Format, one exchange per block:
    User: <instruction, plus the input on a new line when present>
    Assistant: <output>
    <blank line>
Writes OUT_DIR/train/qa_train.txt (98%), OUT_DIR/heldout/qa_heldout.txt (2%) and a small OOD
file, so the directory works with ladder1.py and train_lm.py unchanged.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def render(r: dict) -> str:
    q = r["instruction"].strip()
    if r.get("input", "").strip():
        q += "\n" + r["input"].strip()
    return f"User: {q}\nAssistant: {r['output'].strip()}\n\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("out"); ap.add_argument("--max", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rows = json.load(open(a.source, encoding="utf-8"))
    random.Random(a.seed).shuffle(rows)
    if a.max:
        rows = rows[:a.max]
    out = Path(a.out)
    for split in ("train", "heldout", "ood"):
        (out / split).mkdir(parents=True, exist_ok=True)
    cut = int(len(rows) * 0.98)
    (out / "train" / "qa_train.txt").write_text("".join(render(r) for r in rows[:cut]), encoding="utf-8")
    (out / "heldout" / "qa_heldout.txt").write_text("".join(render(r) for r in rows[cut:]), encoding="utf-8")
    (out / "ood" / "qa_sample.txt").write_text("".join(render(r) for r in rows[cut:cut + 50]), encoding="utf-8")
    sizes = {p.name: p.stat().st_size for p in out.glob("*/*.txt")}
    print(f"{len(rows)} exchanges -> {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
