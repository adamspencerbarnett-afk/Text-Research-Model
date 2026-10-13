"""Verified self-training (STaR / expert iteration) for the arithmetic set.

The model writes several solutions to each training problem with its own stored solution hidden
from memory (so it must solve, not copy); the calculator computes every "a op b ="; a solution is
kept only if its final "Answer: N" equals the reference. The kept solutions, the model's own
correct reasoning, are added to the training text and the model is fine-tuned on them (m2_qa.py).

    # generate, in parallel shards (the 5M model leaves the GPU mostly idle at batch one)
    python compiler/experiments/star.py gen  MODEL.pt --shard 0 --of 8 --k 4 --out runs/star_r1/gen_0.jsonl
    # merge the kept solutions into a new training folder (GSM8K train + kept, held-out unchanged)
    python compiler/experiments/star.py merge runs/star_r1 data_qa_gsm data_qa_gsm_star_r1

Each kept line: {"i": problem index, "q": question, "a": the model's solution, "ref": reference answer}.
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import m2_qa  # noqa: E402


def cmd_gen(a) -> None:
    native = ROOT / "data" / "cv0.exe"
    eng = m2_qa.QAEngine(a.model, native=str(native) if native.exists() else None, device=a.device, log=lambda m: None)
    # the original problems only: a later round's memory also holds earlier kept self-solutions
    n_orig = len(m2_qa.parse_exchanges((ROOT / "data_qa_gsm" / "train" / "qa_train.txt").read_bytes()))
    train = eng.mem.exchanges[:n_orig]
    mine = [i for i in range(len(train)) if i % a.of == a.shard]
    gen = torch.Generator().manual_seed(1000 + a.shard)
    out = open(a.out, "w", encoding="utf-8", newline="\n")
    kept = tried = solved = 0
    t0 = time.time()
    for n, i in enumerate(mine):
        q, ref = train[i]
        target = m2_qa.final_number(ref)
        if target is None:
            continue
        hit = False
        for _ in range(a.k):
            reply, _info = eng.ask(q, a.max_codes, 0, calculator=True, temperature=a.temperature, top_k=a.top_k, gen=gen, exclude=i)
            tried += 1
            if m2_qa.final_number(reply) == target:
                out.write(json.dumps({"i": i, "q": q.decode("utf-8", "replace"), "a": reply.decode("utf-8", "replace"),
                                      "ref": target.decode()}) + "\n")
                kept += 1; hit = True
        solved += hit
        if n % 100 == 0:
            print(f"shard {a.shard}: {n}/{len(mine)} problems, {solved} solved at least once, {kept}/{tried} solutions kept ({time.time() - t0:.0f}s)", flush=True)
    out.close()
    print(f"shard {a.shard} done: {solved}/{len(mine)} problems solved at least once, {kept}/{tried} solutions kept", flush=True)


def cmd_merge(a) -> None:
    rows, seen = [], set()
    for f in sorted(glob.glob(f"{a.gen_dir}/gen_*.jsonl")):
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            key = (r["i"], r["a"].strip())
            if key not in seen and "Answer:" in r["a"]:
                seen.add(key); rows.append(r)
    src, dst = Path(a.src), Path(a.dst)
    for split in ("train", "heldout", "ood"):
        (dst / split).mkdir(parents=True, exist_ok=True)
    shutil.copy(src / "heldout" / "qa_heldout.txt", dst / "heldout" / "qa_heldout.txt")
    shutil.copy(src / "ood" / "qa_sample.txt", dst / "ood" / "qa_sample.txt")
    base = (src / "train" / "qa_train.txt").read_bytes()
    extra = "".join(m2_qa.render(r["q"].encode(), r["a"].strip().encode()).decode("utf-8", "replace") for r in rows).encode("utf-8")
    (dst / "train" / "qa_train.txt").write_bytes(base + extra)
    problems = len({r["i"] for r in rows})
    (dst / "star_manifest.json").write_text(json.dumps({"kept_solutions": len(rows), "problems_with_a_kept_solution": problems,
                                                        "from": a.gen_dir}, indent=1))
    print(f"merged {len(rows):,} distinct kept solutions for {problems:,} problems into {dst}")


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen"); g.add_argument("model"); g.add_argument("--shard", type=int, default=0); g.add_argument("--of", type=int, default=1)
    g.add_argument("--k", type=int, default=4); g.add_argument("--temperature", type=float, default=0.7); g.add_argument("--top-k", type=int, default=20)
    g.add_argument("--max-codes", type=int, default=200); g.add_argument("--device", default="cuda"); g.add_argument("--out", required=True)
    m = sub.add_parser("merge"); m.add_argument("gen_dir"); m.add_argument("src"); m.add_argument("dst")
    a = ap.parse_args()
    torch.set_num_threads(2)
    cmd_gen(a) if a.cmd == "gen" else cmd_merge(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
