"""chat22 data fixes (16 October 2026; docs/V22_CLEAN_DESIGN.md section 3). Writes data_chat22/ and changes nothing else.

  rehearse   random WHOLE documents from the P100 training mix (wiki, fineweb, books, ultrachat, evidence) in the old
             recipe's proportions, ~555 MB (~150M codes). Fixes the 15 MB sample bug: the file name carries the recipe
             and size, and the trainer refuses a file whose size does not match its name.
  long       every training document above 8k codes (28,000 characters; ~3.6 characters per code) and the subset above
             32k (112,000 characters), for the S42/S43 windows and the story continuation windows.
  dialogues  the training dialogues with duplicates and held-out overlap removed ON DISK (first-turn key, 300 chars),
             instead of filtered at load time.
  eval       the clean held-out text re-filtered against the chat training data too (data_mt dialogues, data_qa
             answers, the new rehearsal sample): 32% of eval_clean's ultrachat paragraphs were in data_mt.

    python chat22/data/build_data.py [--only rehearse,long,dialogues,eval] [--seed 1]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data_core" / "mix_P100" / "train"
OUT = ROOT / "data_chat22"
SHARES_MB = {"wiki": 130, "fineweb": 166, "books": 74, "ultrachat": 111, "evidence": 74}     # old 70/90/40/60/40 x 1.85
CHARS_8K, CHARS_32K = 28_000, 112_000


def docs_of(name: str) -> list[bytes]:
    return [d for d in (SRC / f"{name}.txt").read_bytes().split(b"\f") if d.strip()]


def rehearse(seed: int) -> dict:
    rng = random.Random(seed)
    parts, rep = [], {}
    for name, mb in SHARES_MB.items():
        docs = docs_of(name); rng.shuffle(docs)
        got, n, quota = [], 0, mb * 1_000_000
        for d in docs:
            if n >= quota:
                break
            got.append(d.strip()); n += len(d)
        parts += got
        rep[name] = {"documents": len(got), "mb": round(n / 1e6, 1), "of_documents": len(docs)}
    rng.shuffle(parts)                       # documents of all sources interleaved, each whole, \f between them
    data = b"\f".join(parts)
    name = f"rehearse_P100_whole_{len(data) // 1_000_000}MB_seed{seed}.txt"
    (OUT / name).write_bytes(data)
    return {"file": name, "bytes": len(data), "sources": rep}


def long_pool() -> dict:
    rep, l8, l32 = {}, [], []
    for name in ("books", "fineweb", "wiki"):
        docs = docs_of(name)
        a = [d.strip() for d in docs if len(d) > CHARS_8K]
        b = [d for d in a if len(d) > CHARS_32K]
        l8 += a; l32 += b
        rep[name] = {"over_8k": len(a), "over_32k": len(b), "mb_over_8k": round(sum(map(len, a)) / 1e6, 1)}
    (OUT / "long_over8k.txt").write_bytes(b"\f".join(l8))
    (OUT / "long_over32k.txt").write_bytes(b"\f".join(l32))
    return {"over_8k_docs": len(l8), "over_32k_docs": len(l32), "sources": rep}


def key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()[:300]


def dialogues() -> dict:
    held = {key(json.loads(l)["turns"][0][1]) for l in open(ROOT / "data_mt" / "heldout" / "dialogues.jsonl", encoding="utf-8")}
    seen, kept, dup, over = set(), 0, 0, 0
    with open(ROOT / "data_mt" / "train" / "dialogues.jsonl", encoding="utf-8") as f, \
         open(OUT / "dialogues_train.jsonl", "w", encoding="utf-8") as g:
        for l in f:
            k = key(json.loads(l)["turns"][0][1])
            if k in held:
                over += 1; continue
            if k in seen:
                dup += 1; continue
            seen.add(k); g.write(l); kept += 1
    return {"kept": kept, "duplicates_removed": dup, "heldout_overlap_removed": over}


def para_hashes_of_text(b: bytes, into: set):
    for p in re.split(rb"\n\s*\n|\f", b):
        p = re.sub(rb"\s+", b" ", p.strip().lower())
        if len(p) >= 40:
            into.add(hashlib.blake2b(p, digest_size=8).digest())


def eval_clean(rehearse_file: str) -> dict:
    seen: set = set()
    with open(ROOT / "data_mt" / "train" / "dialogues.jsonl", encoding="utf-8") as f:
        for l in f:
            for _, t in json.loads(l)["turns"]:
                para_hashes_of_text(t.encode("utf-8", "replace"), seen)
    para_hashes_of_text((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes(), seen)
    para_hashes_of_text((OUT / rehearse_file).read_bytes(), seen)
    out = OUT / "eval_clean"; out.mkdir(exist_ok=True)
    rep = {}
    for f in sorted((ROOT / "data_core" / "eval_clean" / "heldout").glob("*.txt")):
        paras = [p for p in re.split(rb"\n\s*\n", f.read_bytes()) if p.strip()]
        keep = []
        for p in paras:
            q = re.sub(rb"\s+", b" ", p.strip().lower())
            if len(q) < 40 or hashlib.blake2b(q, digest_size=8).digest() not in seen:
                keep.append(p.strip())
        (out / f.name).write_bytes(b"\n\n".join(keep))
        rep[f.stem] = {"paragraphs": len(paras), "kept": len(keep)}
    return rep


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="rehearse,long,dialogues,eval"); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    man_p = OUT / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for step in a.only.split(","):
        t0 = time.time()
        if step == "rehearse":
            man["rehearse"] = rehearse(a.seed)
        elif step == "long":
            man["long"] = long_pool()
        elif step == "dialogues":
            man["dialogues"] = dialogues()
        elif step == "eval":
            man["eval_clean"] = eval_clean(man["rehearse"]["file"])
        man.setdefault("seconds", {})[step] = round(time.time() - t0)
        man["date"] = time.strftime("%Y-%m-%d %H:%M")
        man_p.write_text(json.dumps(man, indent=1))
        print(step, json.dumps(man.get(step if step != "eval" else "eval_clean"))[:600], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
