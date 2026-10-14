"""Multi-turn dialogue data for the conversation stage (14 October 2026; docs/MULTITURN_PLAN.md).

    python -I compiler/experiments/prepare_dialogue.py data_uploads data_mt --exclude-mix data_core/mix_P100

Reads whole conversations (not the window-sized exchanges the core mixes were cut into):
  UltraChat 200k   data_uploads/ultrachat/train_sft-*.parquet  -> training threads
                   data_uploads/ultrachat/test_sft-*.parquet   -> held-out candidates
  OpenAssistant    data_uploads/oasst1/*trees.jsonl.gz         -> English, top-ranked reply at each turn;
                                                                 1% by thread hash held out
Writes
  OUT/train/dialogues.jsonl     {"source", "turns": [[role, text], ...]} with at least two user turns
  OUT/heldout/dialogues.jsonl   held-out threads that are clean: no assistant reply of theirs occurs in the
                                --exclude-mix training text (the core mixes were built from every UltraChat
                                file, test split included, so most test conversations were trained on)
  OUT/manifest.json             counts, turn-depth histogram, the cleanliness check
Text is normalised as the core mixes are (typographic punctuation to ASCII).
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_core_mix as P  # noqa: E402


def reply_key(text: str) -> bytes:
    """A reply's identity for the cleanliness check: its first 160 characters, normalised. The core mixes
    keep each reply's opening (window_exchanges cuts long replies at a sentence end)."""
    return hashlib.md5(re.sub(r"\W+", " ", P.normalise(text).lower())[:160].strip().encode()).digest()


def ultrachat(path_glob: str) -> list[list[tuple[str, str]]]:
    import pyarrow.parquet as pq
    out = []
    for p in sorted(glob.glob(path_glob)):
        for batch in pq.ParquetFile(p).iter_batches(columns=["messages"], batch_size=4096):
            for msgs in batch.column(0).to_pylist():
                turns = [(m["role"], P.normalise(m["content"].strip())) for m in msgs
                         if m["role"] in ("user", "assistant") and m["content"].strip()]
                out.append(turns)
    return out


def oasst(path: str) -> list[list[tuple[str, str]]]:
    out = []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for line in fh:
            tree = json.loads(line)
            node = tree.get("prompt", {})
            if node.get("lang") != "en":
                continue
            turns = []
            while node:
                turns.append(("user" if node["role"] == "prompter" else "assistant", P.normalise(node["text"].strip())))
                replies = [r for r in node.get("replies", []) if r.get("text")]
                node = min(replies, key=lambda r: r.get("rank", 99) if r.get("rank") is not None else 99) if replies else None
            out.append(turns)
    return out


def well_formed(turns) -> bool:
    """Alternating user/assistant, starting with the user, at least two user turns each answered."""
    if len(turns) < 4 or turns[0][0] != "user":
        return False
    return all(turns[k][0] == ("user" if k % 2 == 0 else "assistant") for k in range(len(turns)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--exclude-mix", default="data_core/mix_P100", help="core mix whose training text held-out threads must not appear in")
    ap.add_argument("--heldout", type=int, default=1000, help="held-out threads to keep (after the cleanliness filter)")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "train").mkdir(parents=True, exist_ok=True); (out / "heldout").mkdir(parents=True, exist_ok=True)
    trained: set[bytes] = set()
    for src in ("ultrachat", "oasst"):
        f = Path(a.exclude_mix, "train", f"{src}.txt")
        if f.exists():
            for m in re.finditer(r"Assistant: (.{20,400})", f.read_text(encoding="utf-8", errors="replace")):
                trained.add(reply_key(m.group(1)))
    print(f"{len(trained):,} assistant replies in {a.exclude_mix} training text", flush=True)

    train = [("ultrachat", t) for t in ultrachat(f"{a.src}/ultrachat/train_sft-*.parquet") if well_formed(t)]
    cand = [("ultrachat", t) for t in ultrachat(f"{a.src}/ultrachat/test_sft-*.parquet") if well_formed(t)]
    for t in oasst(glob.glob(f"{a.src}/oasst1/*trees.jsonl.gz")[0]):
        if not well_formed(t):
            continue
        h = int.from_bytes(hashlib.sha256(t[0][1].encode()).digest()[:4], "little")
        (cand if h % 100 == 0 else train).append(("oasst", t))
    clean, dirty = [], 0
    for s, t in cand:
        if any(reply_key(x) in trained for r, x in t if r == "assistant"):
            dirty += 1
            continue
        clean.append((s, t))
    held = clean[: a.heldout]
    held_keys = {reply_key(x) for _, t in held for r, x in t if r == "assistant"}
    train = [(s, t) for s, t in train if not any(reply_key(x) in held_keys for r, x in t if r == "assistant")]
    for name, rows in (("train", train), ("heldout", held)):
        with open(out / name / "dialogues.jsonl", "w", encoding="utf-8") as fh:
            for s, t in rows:
                fh.write(json.dumps({"source": s, "turns": [list(x) for x in t]}) + "\n")
    depth = lambda rows: dict(sorted(collections.Counter(min(sum(r == "user" for r, _ in t), 6) for _, t in rows).items()))
    man = {"train_threads": len(train), "heldout_threads": len(held), "heldout_candidates": len(cand),
           "heldout_candidates_dropped_as_trained": dirty, "train_sources": dict(collections.Counter(s for s, _ in train)),
           "heldout_sources": dict(collections.Counter(s for s, _ in held)),
           "train_user_turns": depth(train), "heldout_user_turns": depth(held), "exclude_mix": a.exclude_mix}
    (out / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
