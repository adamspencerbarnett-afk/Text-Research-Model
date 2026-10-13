"""A report-card evaluation set that no training mix has seen (13 October 2026).

    python -I compiler/experiments/clean_eval.py data_core/eval_clean --held data_core/mix_D3_5M,data_core/mix_P100 \
        --train data_core/mix_D1_5M,data_core/mix_D2_5M,data_core/mix_D3_5M,data_core/mix_D3s_5M,data_core/mix_D3e_5M,data_core/mix_P50,data_core/mix_P100

Why: the evidence pairs in every mix were drawn from the whole Wikipedia and web sources before the mixes
held out their documents, so held-out paragraphs reach training inside evidence pairs (P100: 22.9% of the
report card's held-out Wikipedia, 10.5% of its web text, 94.6% of its held-out evidence text), and the
evidence-gain pairs (data_core/eval/evidence_val.jsonl) are 89-97% made of paragraphs whose documents are
trained on. This builds, from the held-out folders of --held:
  OUT/heldout/<source>.txt   every held-out paragraph that occurs in no --train mix's training text
                             (normalised-text hash, the mixes' own deduplication rule)
  OUT/evidence_clean.jsonl   evidence-gain pairs built only from those clean Wikipedia and web paragraphs,
                             with the builder's pairing rule (idf-weighted shared distinctive words,
                             0.10 <= similarity <= 0.8, paragraphs at least 20 apart in the file, each
                             paragraph used as evidence once)
  OUT/manifest.json          counts kept and dropped per source
Use with core_eval.py --data OUT --evidence-file OUT/evidence_clean.jsonl --tag _clean.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import random
import re
from pathlib import Path

WORD = re.compile(r"[a-z]{3,}")
COMMON = set("the and for that with this from have were which their there been they will would about into more "
             "other some than then them these only also when what where while after before over such most many "
             "like just your each very make made used using first time year years people part between".split())


def paras(text: str, min_len: int = 60) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) >= min_len]


def norm_hash(p: str) -> bytes:
    return hashlib.md5(re.sub(r"\W+", " ", p.lower()).strip().encode()).digest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out"); ap.add_argument("--held", required=True); ap.add_argument("--train", required=True)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--max-pairs", type=int, default=600)
    a = ap.parse_args()
    out = Path(a.out); (out / "heldout").mkdir(parents=True, exist_ok=True)
    trained: set[bytes] = set()
    for mix in a.train.split(","):
        for f in sorted(Path(mix, "train").glob("*.txt")):
            for line_block in re.split(r"\n\s*\n", f.read_text(encoding="utf-8", errors="replace")):
                for p in line_block.split("\n") + [line_block]:   # paragraphs, and lines inside evidence pairs
                    p = p.strip()
                    if len(p) >= 60:
                        trained.add(norm_hash(p))
        print(f"{mix}: {len(trained):,} training paragraph hashes so far", flush=True)
    man = {"held": a.held, "train": a.train, "sources": {}}
    clean: dict[str, list[str]] = collections.defaultdict(list)
    seen: set[bytes] = set()
    for mix in a.held.split(","):
        for f in sorted(Path(mix, "heldout").glob("*.txt")):
            src = f.stem
            ps = paras(f.read_text(encoding="utf-8", errors="replace"))
            kept = 0
            for p in ps:
                h = norm_hash(p)
                # a paragraph is clean only if neither it nor any line inside it was trained on (an evidence
                # pair is one paragraph of two lines, each of which may occur alone in the training text)
                lines = [x.strip() for x in p.split("\n") if len(x.strip()) >= 60]
                if h in trained or h in seen or any(norm_hash(x) in trained for x in lines):
                    continue
                seen.add(h); clean[src].append(p); kept += 1
            m = man["sources"].setdefault(src, {"paragraphs": 0, "kept": 0})
            m["paragraphs"] += len(ps); m["kept"] += kept
    for src, ps in clean.items():
        (out / "heldout" / f"{src}.txt").write_bytes(("\n\n".join(ps) + "\n").encode("utf-8"))
        print(f"  {src:10s} kept {man['sources'][src]['kept']:,} of {man['sources'][src]['paragraphs']:,} held-out paragraphs", flush=True)
    # evidence pairs from clean Wikipedia and web paragraphs only
    pool = [p for src in ("wiki", "fineweb") for p in clean.get(src, []) if 200 <= len(p) <= 900]
    words = [set(WORD.findall(p.lower())) - COMMON for p in pool]
    df = collections.Counter(w for ws in words for w in ws)
    idf = {w: math.log((1 + len(pool)) / (1 + c)) for w, c in df.items()}
    distinctive = {w for w, c in df.items() if c <= max(2, 0.02 * len(pool))}
    index: dict[str, list[int]] = collections.defaultdict(list)
    for i, ws in enumerate(words):
        for w in ws & distinctive:
            index[w].append(i)
    rng = random.Random(a.seed)
    order = list(range(len(pool))); rng.shuffle(order)
    used, pairs = set(), []
    for t in order:
        cand = collections.Counter(j for w in words[t] & distinctive for j in index[w] if abs(j - t) >= 20 and j not in used)
        best, best_s = None, 0.0
        for j, _ in cand.most_common(20):
            shared, union = words[t] & words[j], words[t] | words[j]
            s = sum(idf[w] for w in shared) / max(1e-9, sum(idf[w] for w in union))
            if 0.10 <= s <= 0.8 and s > best_s:
                best, best_s = j, s
        if best is not None:
            used.add(best); pairs.append({"evidence": pool[best], "target": pool[t], "similarity": round(best_s, 3)})
        if len(pairs) >= a.max_pairs:
            break
    with open(out / "evidence_clean.jsonl", "w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p) + "\n")
    man["evidence_pairs"] = len(pairs); man["evidence_pool"] = len(pool)
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(f"evidence pairs: {len(pairs)} from {len(pool):,} clean Wikipedia and web paragraphs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
