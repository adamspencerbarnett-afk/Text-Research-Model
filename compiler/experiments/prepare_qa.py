"""Turn an instruction dataset into plain chat text the compiler can encode, so a model can be
trained (or continued) on it like any other text and talked to through chat.py.

    python -I compiler/experiments/prepare_qa.py ALPACA.json OUT_DIR [--max 52000]
    python -I compiler/experiments/prepare_qa.py GSM8K_DIR OUT_DIR --format gsm8k   # train.jsonl + test.jsonl

GSM8K (grade-school arithmetic, MIT licence) has a checkable numeric answer: the reasoning is
kept, the calculator annotations <<...>> are dropped, and the final line becomes "Answer: N",
so a check loop can read the target. Its official test split is the held-out file.

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
import re
from pathlib import Path


def render(r: dict) -> str:
    q = r["instruction"].strip()
    if r.get("input", "").strip():
        q += "\n" + r["input"].strip()
    return f"User: {q}\nAssistant: {r['output'].strip()}\n\n"


def render_gsm(r: dict) -> str:
    body, _, final = r["answer"].rpartition("####")
    body = re.sub(r"<<[^>]*>>", "", body).strip()
    return "User: " + r["question"].strip() + "\nAssistant: " + body + "\nAnswer: " + final.strip() + "\n\n"


def squad_rows(path: Path, max_words: int) -> tuple[list[str], int]:
    """SQuAD 1.1: one exchange per question; the passage is given in the question, the answer is
    the first reference span. Only passages that fit the window are kept."""
    out, total = [], 0
    for art in json.loads(path.read_text(encoding="utf-8"))["data"]:
        for para in art["paragraphs"]:
            ctx = " ".join(para["context"].split())
            for qa in para["qas"]:
                total += 1
                q = " ".join(qa["question"].split())
                if len(ctx.split()) + len(q.split()) > max_words:
                    continue
                ans = qa["answers"][0]["text"].strip()
                out.append("User: Passage: " + ctx + chr(10) + "Question: " + q + chr(10) + "Assistant: " + ans + chr(10) + chr(10))
    return out, total


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("source"); ap.add_argument("out"); ap.add_argument("--max", type=int, default=0); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--format", choices=["alpaca", "gsm8k", "squad"], default="alpaca")
    ap.add_argument("--max-words", type=int, default=150, help="squad: keep passage + question up to this many words (fits a 256-code window)")
    a = ap.parse_args()
    out = Path(a.out)
    for split in ("train", "heldout", "ood"):
        (out / split).mkdir(parents=True, exist_ok=True)
    if a.format == "squad":
        tr, n_tr = squad_rows(Path(a.source) / "train-v1.1.json", a.max_words)
        te, n_te = squad_rows(Path(a.source) / "dev-v1.1.json", a.max_words)
        random.Random(a.seed).shuffle(tr)
        (out / "train" / "qa_train.txt").write_bytes("".join(tr).encode("utf-8"))
        (out / "heldout" / "qa_heldout.txt").write_bytes("".join(te).encode("utf-8"))
        (out / "ood" / "qa_sample.txt").write_bytes("".join(te[:50]).encode("utf-8"))
        print(f"squad: kept {len(tr)}/{n_tr} train, {len(te)}/{n_te} dev (passage + question <= {a.max_words} words)")
        return 0
    if a.format == "gsm8k":
        train, test = read_jsonl(Path(a.source) / "train.jsonl"), read_jsonl(Path(a.source) / "test.jsonl")
        random.Random(a.seed).shuffle(train)
        (out / "train" / "qa_train.txt").write_bytes("".join(render_gsm(r) for r in train).encode("utf-8"))
        (out / "heldout" / "qa_heldout.txt").write_bytes("".join(render_gsm(r) for r in test).encode("utf-8"))
        (out / "ood" / "qa_sample.txt").write_bytes("".join(render_gsm(r) for r in test[:50]).encode("utf-8"))
        print(f"gsm8k: {len(train)} train, {len(test)} held-out -> {[p.stat().st_size for p in out.glob('*/*.txt')]}")
        return 0
    rows = json.load(open(a.source, encoding="utf-8"))
    random.Random(a.seed).shuffle(rows)
    if a.max:
        rows = rows[:a.max]
    cut = int(len(rows) * 0.98)
    (out / "train" / "qa_train.txt").write_bytes(("".join(render(r) for r in rows[:cut])).encode("utf-8"))
    (out / "heldout" / "qa_heldout.txt").write_bytes(("".join(render(r) for r in rows[cut:])).encode("utf-8"))
    (out / "ood" / "qa_sample.txt").write_bytes(("".join(render(r) for r in rows[cut:cut + 50])).encode("utf-8"))
    sizes = {p.name: p.stat().st_size for p in out.glob("*/*.txt")}
    print(f"{len(rows)} exchanges -> {sizes}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
