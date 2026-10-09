"""Corpus v2 = corpus v1 plus two files supplied by Adam: a combined file of 100 Gutenberg books
(split on its FILE markers, books already in the corpus or held out dropped by Gutenberg id)
and the first 50 MiB of the enwik Wikipedia benchmark text, of which the first 2 MiB become
an out-of-domain evaluation file and the rest a training file kept in its own split
(``train_wiki``), so books-only and books-plus-wiki runs can both be made from v2.

    python -I compiler/experiments/prepare_corpus_v2.py DATA_V1 BOOKS100.txt ENWIK.txt DATA_V2
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_books import clean  # noqa: E402

MARK = re.compile(r"=+\r?\n\s*FILE: ([^\r\n]+?)\s*\r?\n=+\r?\n")


def main(v1: str, books: str, enwik: str, v2: str) -> int:
    v1p, v2p = Path(v1), Path(v2)
    for split in ("train", "heldout", "ood"):
        (v2p / split).mkdir(parents=True, exist_ok=True)
        for p in sorted((v1p / split).glob("*.txt")):
            shutil.copy(p, v2p / split / p.name)
    (v2p / "train_wiki").mkdir(exist_ok=True)
    known_ids = {p.stem.rsplit("_", 1)[-1] for split in ("train", "heldout") for p in (v2p / split).glob("*.txt")}
    text = Path(books).read_text(encoding="utf-8", errors="replace")
    parts = MARK.split(text)
    added, dropped = [], []
    for i in range(1, len(parts) - 1, 2):
        fname, body = parts[i].strip(), parts[i + 1]
        m = re.match(r"(\d+)_(.+?)\.txt$", fname)
        book_id, title = (m.group(1), m.group(2)) if m else ("", fname)
        if book_id in known_ids:
            dropped.append(fname); continue
        body = body.replace("\r\n", "\n")
        a = re.search(r"\*\*\* ?START OF[^\n]*\n", body); b = re.search(r"\n\*\*\* ?END OF", body)
        body = body[a.end() if a else 0: b.start() if b else len(body)].strip("\n") + "\n"
        if len(body) < 100_000:
            dropped.append(fname); continue
        name = re.sub(r"[^A-Za-z0-9.]+", "-", title).strip("-") + f"_{book_id or 'x'}"
        (v2p / "train" / f"{name}.txt").write_bytes((body).encode("utf-8"))
        known_ids.add(book_id); added.append(name)
    wiki = Path(enwik).read_bytes()
    (v2p / "ood" / "enwik_head_2MiB.txt").write_bytes(wiki[:2 << 20])
    (v2p / "train_wiki" / "enwik_rest.txt").write_bytes(wiki[2 << 20:])
    manifest = {"books": {}, "train_bytes": 0, "heldout_bytes": 0, "train_wiki_bytes": 0}
    for split in ("train", "heldout", "ood", "train_wiki"):
        for p in sorted((v2p / split).glob("*.txt")):
            data = p.read_bytes()
            manifest["books"][f"{split}/{p.name}"] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
            if f"{split}_bytes" in manifest:
                manifest[f"{split}_bytes"] += len(data)
    manifest["source"] = "corpus v1 plus Adam's 100-book file (deduplicated by Gutenberg id) and the enwik 50 MiB prefix"
    manifest["books100_added"], manifest["books100_dropped"] = added, dropped
    (v2p / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"books100: {len(added)} added, {len(dropped)} dropped as duplicates/held-out/short: {dropped[:12]}{'...' if len(dropped) > 12 else ''}")
    print(f"v2: train {manifest['train_bytes']/1e6:.1f} MB in {sum(k.startswith('train/') for k in manifest['books'])} books; "
          f"held-out {manifest['heldout_bytes']/1e6:.2f} MB; wiki train {manifest['train_wiki_bytes']/1e6:.1f} MB; ood files {sum(k.startswith('ood/') for k in manifest['books'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
