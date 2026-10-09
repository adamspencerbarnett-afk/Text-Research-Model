"""Corpus v1: the 45 MB book corpus grown to a target size from the same source (GITenberg),
with a manifest, cleaned exactly as prepare_books.py cleans, plus one more held-out book by an
author absent from training.

    python compiler/experiments/prepare_corpus_v1.py DATA_V0 DATA_V1 --target-mb 200

The v0 books are copied as they are (so v0 is a subset of v1). New books come from the list of
GITenberg repositories that ships with the gitberg tool (``--repo-list``, 72,553 names), a
curated list first (``--curated``) and then a seeded random sample of the rest, fetching one
text file per book over HTTPS from raw.githubusercontent.com. A book is kept if it is 100 KB to
2 MB after cleaning, looks like English prose, and its metadata does not name a held-out
author. Every file's size and SHA-256 go into DATA_V1/manifest.json.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import random
import re
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_books import clean, HELD_OUT  # noqa: E402

EXTRA_HELD_OUT = {"Walden-and-On-The-Duty-Of-Civil-Disobedience_205"}   # Thoreau: no other Thoreau in training
HELD_OUT_AUTHORS = ("Conrad", "Doyle", "Machiavelli", "Thoreau")
RAW = "https://raw.githubusercontent.com/GITenberg/{name}/{branch}/{file}"


def get(url: str, timeout: float = 30) -> bytes | None:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "text-research-model corpus"}), timeout=timeout) as r:
            return r.read()
    except Exception:
        return None


def looks_english(text: str) -> bool:
    n = max(1, len(text))
    letters = sum(c.isascii() and c.isalpha() for c in text[:200_000]) / min(n, 200_000)
    the = text[:200_000].count(" the ") / max(1, min(n, 200_000) / 1000)
    return letters > 0.70 and the > 4   # at least ~4 " the " per 1,000 characters


def held_out_author(name: str) -> bool:
    meta = get(RAW.format(name=name, branch="master", file="metadata.yaml"))
    return bool(meta) and any(a in meta.decode("utf-8", "replace") for a in HELD_OUT_AUTHORS)


def fetch_book(name: str) -> str | None:
    book_id = name.rsplit("_", 1)[-1]
    for branch in ("master",):
        for file in (f"{book_id}-0.txt", f"{book_id}-8.txt", f"{book_id}.txt"):
            raw = get(RAW.format(name=name, branch=branch, file=file))
            if raw and len(raw) > 100_000:
                text = raw.decode("latin-1") if file.endswith("-8.txt") else raw.decode("utf-8", errors="replace")
                text = text.lstrip("﻿").replace("\r\n", "\n")
                a = re.search(r"\*\*\* ?START OF[^\n]*\n", text)
                b = re.search(r"\n\*\*\* ?END OF", text)
                text = text[a.end() if a else 0: b.start() if b else len(text)].strip("\n") + "\n"
                return text
    return None


def consider(name: str):
    """Fetch, clean and screen one book; returns (name, text) or (name, None)."""
    text = fetch_book(name)
    if not text or not (100_000 <= len(text.encode("utf-8")) <= 2_000_000) or not looks_english(text):
        return name, None
    if held_out_author(name):
        return name, None
    return name, text


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("v0"); ap.add_argument("v1"); ap.add_argument("--target-mb", type=float, default=200)
    ap.add_argument("--repo-list", required=True); ap.add_argument("--curated", default=None)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--max-id", type=int, default=30000)
    a = ap.parse_args()
    v0, v1 = Path(a.v0), Path(a.v1)
    for split in ("train", "heldout", "ood"):
        (v1 / split).mkdir(parents=True, exist_ok=True)
        for p in sorted((v0 / split).glob("*.txt")):
            shutil.copy(p, v1 / split / p.name)
    have = {p.stem for split in ("train", "heldout") for p in (v1 / split).glob("*.txt")}
    total = sum(p.stat().st_size for p in (v1 / "train").glob("*.txt"))
    target = a.target_mb * 1e6
    print(f"starting from {len(have)} books, {total/1e6:.1f} MB of training text; target {a.target_mb:.0f} MB", flush=True)
    for name in sorted(EXTRA_HELD_OUT):
        text = fetch_book(name)
        if text:
            (v1 / "heldout" / f"{name}.txt").write_bytes((text).encode("utf-8")); have.add(name)
            print(f"held-out: {name} ({len(text)/1e3:.0f} KB)", flush=True)
    curated = [l.strip() for l in open(a.curated)] if a.curated else []
    rows = [l.rstrip("\n").split("\t") for l in open(a.repo_list, encoding="utf-8", errors="replace")]
    pool = [n for i, n in rows if i.isdigit() and int(i) <= a.max_id and re.search(r"_\d+$", n)]
    random.Random(a.seed).shuffle(pool)
    candidates = [n for n in curated + pool if n not in have and n not in HELD_OUT and n not in EXTRA_HELD_OUT]
    kept = skipped = 0
    with cf.ThreadPoolExecutor(8) as ex:
        for i in range(0, len(candidates), 32):
            if total >= target:
                break
            for name, text in ex.map(consider, candidates[i:i + 32]):
                if text is None:
                    skipped += 1
                    continue
                if name in have or total >= target:
                    continue
                data = text.encode("utf-8")
                (v1 / "train" / f"{name}.txt").write_bytes(data)
                have.add(name); kept += 1; total += len(data)
            print(f"  {kept} books added, {skipped} skipped, {total/1e6:.1f} MB", flush=True)
    manifest = {"books": {}, "train_bytes": 0, "heldout_bytes": 0}
    for split in ("train", "heldout", "ood"):
        for p in sorted((v1 / split).glob("*.txt")):
            data = p.read_bytes()
            manifest["books"][f"{split}/{p.name}"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if split in ("train", "heldout"):
                manifest[f"{split}_bytes"] += len(data)
    manifest["source"] = ("GITenberg via raw.githubusercontent.com, cleaned as prepare_books.py; v0 corpus included unchanged; "
                          f"curated list then seeded sample (seed {a.seed}) of repository ids <= {a.max_id}; "
                          f"books whose metadata names {HELD_OUT_AUTHORS} excluded from training")
    (v1 / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"done: {kept} books added ({skipped} skipped); train {manifest['train_bytes']/1e6:.1f} MB in "
          f"{sum(k.startswith('train/') for k in manifest['books'])} books; held-out {manifest['heldout_bytes']/1e6:.2f} MB", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
