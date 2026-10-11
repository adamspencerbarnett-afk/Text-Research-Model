"""Evidence pairs, built in parallel (the same pairing rule as prepare_core_mix.cmd_evidence, about
10-50x faster): numpy word index shared from one file, vectorised scoring, W worker processes, merge.

    python -I compiler/experiments/evidence_parallel.py prep  data_core/src/wiki.txt,data_core/src/fineweb.txt runs/evp --min-len 200 --max-len 900
    python -I compiler/experiments/evidence_parallel.py work  runs/evp --worker 0 --of 10        # x10, in parallel
    python -I compiler/experiments/evidence_parallel.py merge runs/evp data_core/src_p50/evidence.txt data_core/eval/evidence_val_p50.jsonl --max-pairs 450000

Pairing rule (unchanged): content words of 3+ letters minus a common-word list; only distinctive words
(in at most 0.2% of paragraphs) vote; similarity = idf of shared voting words / idf of all words of the
two paragraphs; kept if 0.10 <= similarity <= 0.8, from different documents; each paragraph is used as
evidence at most once (enforced in the merge, in target order).
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_core_mix as P  # noqa: E402


def cmd_prep(srcs: str, out: str, min_len: int, max_len: int, seed: int = 1) -> None:
    t0 = time.time()
    rng = random.Random(seed)
    paras, docs = [], []
    for s_i, src in enumerate(srcs.split(",")):
        for d_i, doc in enumerate(P.read_docs(src)):
            for p in doc.split("\n\n"):
                p = P.normalise(p)
                if min_len <= len(p) <= max_len:
                    paras.append(p); docs.append(s_i * 10_000_000 + d_i)
    order = list(range(len(paras))); rng.shuffle(order)
    paras = [paras[i] for i in order]; docs = np.array([docs[i] for i in order], dtype=np.int64)
    vocab: dict[str, int] = {}
    key_ids, offsets = [], [0]
    for p in paras:
        ws = {w for w in (x.lower() for x in P.WORD.findall(p)) if w not in P.COMMON}
        ids = sorted(vocab.setdefault(w, len(vocab)) for w in ws)
        key_ids.extend(ids); offsets.append(len(key_ids))
    key_ids = np.array(key_ids, dtype=np.int32); offsets = np.array(offsets, dtype=np.int64)
    n = len(paras)
    df = np.bincount(key_ids, minlength=len(vocab)).astype(np.int64)
    idf = np.log((1 + n) / (1 + df)).astype(np.float32)
    idf_sum = np.add.reduceat(idf[key_ids], offsets[:-1]) if len(key_ids) else np.zeros(n, np.float32)
    idf_sum[np.diff(offsets) == 0] = 0
    # postings for distinctive words only: word -> paragraphs (CSR)
    para_of = np.repeat(np.arange(n, dtype=np.int32), np.diff(offsets))
    distinct = df <= max(1, int(0.002 * n))
    sel = distinct[key_ids]
    w_sel, p_sel = key_ids[sel], para_of[sel]
    o = np.argsort(w_sel, kind="stable")
    post_para = p_sel[o]
    post_off = np.concatenate([[0], np.cumsum(np.bincount(w_sel, minlength=len(vocab)))]).astype(np.int64)
    outp = Path(out); outp.mkdir(parents=True, exist_ok=True)
    np.savez(outp / "index.npz", key_ids=key_ids, offsets=offsets, idf=idf, idf_sum=idf_sum, docs=docs,
             distinct=distinct, post_para=post_para, post_off=post_off)
    (outp / "paras.txt").write_bytes("\n\f\n".join(p.replace("\f", " ") for p in paras).encode("utf-8"))
    print(f"prep: {n:,} paragraphs, {len(vocab):,} words, {int(distinct.sum()):,} distinctive ({time.time() - t0:.0f}s)", flush=True)


def cmd_work(dir_: str, worker: int, of: int, min_sim: float = 0.10) -> None:
    t0 = time.time()
    z = np.load(Path(dir_) / "index.npz")
    key_ids, offsets, idf, idf_sum, docs = z["key_ids"], z["offsets"], z["idf"], z["idf_sum"], z["docs"]
    distinct, post_para, post_off = z["distinct"], z["post_para"], z["post_off"]
    n = len(offsets) - 1
    out = open(Path(dir_) / f"pairs_{worker}.tsv", "w", encoding="ascii", newline="\n")
    found = 0
    for i in range(worker, n, of):
        ks = key_ids[offsets[i]:offsets[i + 1]]
        dk = ks[distinct[ks]]
        if len(dk) == 0:
            continue
        cand = np.concatenate([post_para[post_off[w]:post_off[w + 1]] for w in dk])
        wts = np.repeat(idf[dk], post_off[dk + 1] - post_off[dk])
        u, inv = np.unique(cand, return_inverse=True)
        score = np.bincount(inv, weights=wts)
        bad = (u == i) | (docs[u] == docs[i])
        score[bad] = -1
        # the three best partners: the merge takes the best one not already used as evidence,
        # as the serial builder does by skipping used paragraphs during the search
        top = np.argsort(-score)[:3]
        hit = False
        for rank, b in enumerate(top):
            if score[b] <= 0:
                break
            j = int(u[b])
            kj = key_ids[offsets[j]:offsets[j + 1]]
            shared_all = np.intersect1d(ks, kj, assume_unique=True)
            union = float(idf_sum[i] + idf_sum[j] - idf[shared_all].sum())
            sim = float(score[b]) / max(union, 1e-9)
            if min_sim <= sim <= 0.8:
                out.write(f"{i}\t{rank}\t{j}\t{sim:.4f}\n"); hit = True
        found += hit
        if (i // of) % 50_000 == 0:
            print(f"worker {worker}: {i // of:,}/{n // of:,} targets, {found:,} pairs ({time.time() - t0:.0f}s)", flush=True)
    out.close()
    print(f"worker {worker} done: {found:,} pairs ({time.time() - t0:.0f}s)", flush=True)


def cmd_merge(dir_: str, out: str, val_out: str, max_pairs: int) -> None:
    rows = []
    for f in sorted(Path(dir_).glob("pairs_*.tsv")):
        for line in open(f, encoding="ascii"):
            i, r, j, s = line.split("\t"); rows.append((int(i), int(r), int(j), float(s)))
    rows.sort()                                         # target order (the shuffled order, as the serial builder), then rank
    paras = (Path(dir_) / "paras.txt").read_text(encoding="utf-8").split("\n\f\n")
    used, done, pairs = set(), set(), []
    for i, _r, j, s in rows:
        if i in done or j in used:
            continue
        used.add(j); done.add(i); pairs.append((paras[j], paras[i], s))
        if len(pairs) >= max_pairs:
            break
    cut = int(len(pairs) * 0.99)
    P.write_docs(out, [f"{e}\n\n{t}" for e, t, _ in pairs[:cut]])
    Path(val_out).parent.mkdir(parents=True, exist_ok=True)
    with open(val_out, "w", encoding="utf-8", newline="\n") as f:
        for e, t, s in pairs[cut:]:
            f.write(json.dumps({"evidence": e, "target": t, "sim": s}) + "\n")
    med = sorted(s for _, _, s in pairs)[len(pairs) // 2] if pairs else 0
    print(f"evidence: {cut:,} training pairs, {len(pairs) - cut:,} held out; median similarity {med}; {len(rows):,} candidates before the once-only rule")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("a"); ap.add_argument("b", nargs="?"); ap.add_argument("c", nargs="?")
    ap.add_argument("--min-len", type=int, default=200); ap.add_argument("--max-len", type=int, default=900)
    ap.add_argument("--worker", type=int, default=0); ap.add_argument("--of", type=int, default=1)
    ap.add_argument("--max-pairs", type=int, default=450_000)
    a = ap.parse_args()
    if a.cmd == "prep": cmd_prep(a.a, a.b, a.min_len, a.max_len)
    elif a.cmd == "work": cmd_work(a.a, a.worker, a.of)
    elif a.cmd == "merge": cmd_merge(a.a, a.b, a.c, a.max_pairs)
    else: raise SystemExit("cmd: prep | work | merge")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
