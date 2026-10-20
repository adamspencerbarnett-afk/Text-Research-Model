"""The permanent fact store (Adam, 16 Oct 2026): real facts only, nothing made up. Personal and fictional things belong
in conversation memory; this store is what the engine looks up as lasting knowledge. Training data is NOT changed.

Source: Wikipedia articles of the P100 training split (data_core/mix_P100/train/wiki.txt; title on the first line).
  lead   "What is <title>?" -> the article's first sentence ("X is a ...")
  piece  clean ~200-byte pieces of the article body, labelled with the article title
Web junk, markup, lists and number tables are filtered out. Writes data_chat22/knowledge_wiki.jsonl (rows: q, a, key, source).

    python -m chat22.data.build_knowledge [--articles 25000] [--pieces 4] [--seed 1]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ..memory import split_text

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data_core" / "mix_P100" / "train" / "wiki.txt"
OUT = ROOT / "data_chat22" / "knowledge_wiki.jsonl"
LEAD = re.compile(r"^(.{3,200}?\b(?:is|was|are|were)\b.{10,300}?[.])(?:\s|$)", re.S)


def clean(t: str) -> bool:
    if len(t) < 60 or any(x in t for x in ("http", "|", "{{", "[[", "==", "<", ">", "\t")):
        return False
    letters = sum(c.isalpha() for c in t)
    return letters / len(t) > 0.7 and t[:1].isupper() and t.rstrip()[-1:] in ".!?\")"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--articles", type=int, default=25000); ap.add_argument("--pieces", type=int, default=4)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    docs = [d for d in SRC.read_bytes().split(b"\f") if d.strip()]
    rng = np.random.default_rng(a.seed)
    rows, n_lead, n_art = [], 0, 0
    for j in rng.permutation(len(docs)):
        if n_art >= a.articles:
            break
        d = docs[int(j)].decode("utf-8", "replace").strip()
        title, _, body = d.partition("\n")
        title, body = title.strip(), body.strip()
        if not (2 <= len(title) <= 80) or len(body) < 400 or title.lower().startswith(("list of", "index of")):
            continue
        got = 0
        m = LEAD.match(re.sub(r"\s+", " ", body[:600]))
        if m and clean(m.group(1)):
            rows.append({"q": f"What is {title}?", "a": m.group(1), "key": f"What is {title}?", "source": "wiki_lead"})
            n_lead += 1; got += 1
        for p in split_text(body[200:8000].encode("utf-8"))[1:]:
            t = re.sub(r"\s+", " ", p.decode("utf-8", "replace")).strip()
            if clean(t):
                rows.append({"q": f"From the Wikipedia article \"{title}\"", "a": t, "key": f"{title}. {t}", "source": "wiki_piece"})
                got += 1
                if got > a.pieces:
                    break
        n_art += got > 0
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rep = {"file": str(OUT.relative_to(ROOT)), "rows": len(rows), "articles": n_art, "leads": n_lead, "pieces": len(rows) - n_lead}
    print(json.dumps(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
