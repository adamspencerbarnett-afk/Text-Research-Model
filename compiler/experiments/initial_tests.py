"""Initial Compiler v0 experiments: sequence length, learnable structure and speed.

    python compiler/experiments/initial_tests.py DATA_DIR WORK_DIR [--native PATH_TO_cv0]

DATA_DIR comes from prepare_books.py. Dictionaries, ID files and results are written to
WORK_DIR; the summary JSON is also printed. Only the training split is used for fitting.

Measures, per encoder configuration:
  * IDs per 1,000 bytes on held-out books and on out-of-domain Markdown and Python code;
  * the share of held-out words encoded as a single ID, and IDs per word;
  * held-out bits per byte of interpolated Witten-Bell n-gram models (orders 1-3) trained on
    the encoded training split: a cheap proxy for how much learnable structure the encoding
    keeps (not a neural result);
  * native encode/decode speed (best of 5 on the 45 MB training split).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_v0 as cv  # noqa: E402

CONFIGS = [
    # name, kind, vocab, phrases, max_phrase_units, case_flags
    ("bytes", "bytes", 256, 0, 1, False),
    ("bpe_32k", "bpe", 32768, 0, 1, False),
    ("whole_words_32k", "whole", 32768, 0, 1, True),
    ("v0_32k_plain", "v0", 32768, 0, 1, False),
    ("v0_8k", "v0", 8192, 0, 1, True),
    ("v0_16k", "v0", 16384, 0, 1, True),
    ("v0_32k", "v0", 32768, 0, 1, True),
    ("v0_64k", "v0", 65536, 0, 1, True),
    ("v0_32k_phrases", "v0", 32768, 4096, 3, True),
    ("v0_64k_phrases", "v0", 65536, 8192, 3, True),
    ("v0_32k_plain_phrases", "v0", 32768, 4096, 3, False),
    ("v0_64k_plain_phrases", "v0", 65536, 8192, 3, False),
]


def read(paths):
    return [open(p, "rb").read() for p in paths]


def wb_bits(train: np.ndarray, test: np.ndarray, vocab: int, max_order: int = 3) -> dict[int, float]:
    """Total held-out bits of interpolated Witten-Bell n-grams, vectorised with numpy."""
    train = train.astype(np.int64)
    test = test.astype(np.int64)
    p = np.full(len(test), 1.0 / vocab)
    out = {}
    for order in range(1, max_order + 1):
        ctx = order - 1

        def keys(a):
            h = np.zeros(len(a) - ctx, dtype=np.int64)
            for j in range(ctx):
                h = h * vocab + a[j:len(a) - ctx + j]
            return h, a[ctx:]

        h_tr, w_tr = keys(train)
        uniq, counts = np.unique(h_tr * vocab + w_tr, return_counts=True)
        hs = uniq // vocab
        h_vals, first = np.unique(hs, return_index=True)
        totals = np.add.reduceat(counts, first)
        types = np.diff(np.append(first, len(hs)))
        h_te, w_te = keys(test)
        k_te = h_te * vocab + w_te
        pos = np.minimum(np.searchsorted(uniq, k_te), len(uniq) - 1)
        c_hw = np.where(uniq[pos] == k_te, counts[pos], 0)
        hp = np.minimum(np.searchsorted(h_vals, h_te), len(h_vals) - 1)
        seen = h_vals[hp] == h_te
        tot = np.where(seen, totals[hp], 0).astype(np.float64)
        typ = np.where(seen, types[hp], 0).astype(np.float64)
        prev = p[ctx:]
        p[ctx:] = np.where(tot > 0, (c_hw + typ * prev) / np.maximum(tot + typ, 1), prev)
        out[order] = float(-np.log2(p).sum())
    return out


class Bytes:
    vocab_size = 256

    def encode(self, data):
        return list(data)


def word_stats(units_ids):
    words = [ids for unit, ids in units_ids if unit.strip(b" ").isalpha()]
    single = sum(1 for ids in words if len(ids) == 1)
    return {"words": len(words), "single_id_share": single / max(1, len(words)),
            "ids_per_word": sum(len(ids) for ids in words) / max(1, len(words))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("work")
    ap.add_argument("--native", default=None)
    a = ap.parse_args()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    train_paths = sorted(glob.glob(f"{a.data}/train/*.txt"))
    held_paths = sorted(glob.glob(f"{a.data}/heldout/*.txt"))
    ood = {Path(p).stem: open(p, "rb").read() for p in sorted(glob.glob(f"{a.data}/ood/*.txt"))}
    train, held = read(train_paths), read(held_paths)
    train_bytes, held_bytes = sum(map(len, train)), sum(map(len, held))
    corpus = work / "train_concat.txt"
    if not corpus.exists():
        corpus.write_bytes(b"".join(train))
    print(f"train {len(train)} books {train_bytes/1e6:.1f} MB; held-out {len(held)} books {held_bytes/1e6:.2f} MB", flush=True)

    counts = {flag: cv.count_units(train, flag) for flag in (False, True)}
    results = {"corpus": {"train_books": len(train), "train_bytes": train_bytes, "heldout_books": [Path(p).stem for p in held_paths],
                          "heldout_bytes": held_bytes, "ood_bytes": {k: len(v) for k, v in ood.items()}},
               "machine": {"python": platform.python_version(), "cpu": platform.processor() or platform.machine(),
                           "cores": os.cpu_count()},
               "configs": {}}
    for name, kind, vocab, phrases, units, flags in CONFIGS:
        t0 = time.time()
        dpath = None
        if kind == "bytes":
            enc, vsize = Bytes(), 256
        elif kind == "bpe":
            _, merges = cv.learn_pieces(counts[flags], vocab - cv.FIRST_ENTRY_ID, positional=False)
            enc = cv.BPEBaseline(merges, flags)
            vsize = cv.FIRST_ENTRY_ID + len(enc.ids)
        elif kind == "whole":
            top = [u for u, _ in sorted(counts[flags].items(), key=lambda t: (-t[1], t[0]))
                   if len(u) <= cv.MAX_ENTRY_LEN][:vocab - cv.FIRST_ENTRY_ID]
            d = cv.Dictionary([("S", u) for u in top], flags, 1)
            enc, vsize = cv.Encoder(d), d.vocab_size
        else:
            dpath = work / f"{name}.cv0d"
            timing = work / f"{name}.fit_seconds"
            if dpath.exists() and timing.exists():
                d = cv.Dictionary.load(str(dpath))
            else:
                d, _ = cv.fit(lambda: iter(train), vocab, phrases, units, flags)
                d.save(str(dpath))
                timing.write_text(f"{time.time() - t0:.1f}")
            enc, vsize = cv.Encoder(d), d.vocab_size
        fit_s = float(timing.read_text()) if dpath else time.time() - t0
        row = {"kind": kind, "vocab": vsize, "phrases": phrases, "case_flags": flags, "fit_seconds": round(fit_s, 1)}
        if dpath:
            row["dictionary_hash"] = f"{cv.Dictionary.load(str(dpath)).hash:016x}"

        held_ids = [enc.encode(x) for x in held]
        for x, ids in zip(held, held_ids):
            if kind in ("v0", "whole"):
                assert enc.decode(ids) == x, f"round trip failed for {name}"
        row["heldout_ids_per_kb"] = round(1000 * sum(map(len, held_ids)) / held_bytes, 1)
        row["ood_ids_per_kb"] = {k: round(1000 * len(enc.encode(v)) / len(v), 1) for k, v in ood.items()}
        if kind in ("v0", "whole"):
            pairs = []
            for x in held:
                for flag, unit in enc.normalised_units(x):
                    pairs.append((unit, enc.pieces(unit)))
            row.update(word_stats(pairs))
        elif kind == "bpe":
            pairs = []
            for x in held:
                for unit in cv.scan(x):
                    pairs.append((unit, enc._unit(unit)))
            row.update(word_stats(pairs))

        # Encode the training split (native when available) for the n-gram proxy and speed.
        if dpath and a.native:
            ids_file = work / f"{name}.ids"
            subprocess.run([a.native, "encode", str(dpath), str(corpus), str(ids_file)], check=True, capture_output=True)
            train_ids = np.array(cv.read_ids(str(ids_file)), dtype=np.int64)
            ref = cv.read_ids(str(ids_file))[:200000]
            # Parity spot check against the reference on the held-out books.
            for p, ids in zip(held_paths, held_ids):
                out = work / "parity.ids"
                subprocess.run([a.native, "encode", str(dpath), p, str(out)], check=True, capture_output=True)
                assert cv.read_ids(str(out)) == ids, f"native/reference mismatch for {name} on {p}"
            bench = json.loads(subprocess.run([a.native, "bench", str(dpath), str(corpus), "5"], check=True,
                                              capture_output=True, text=True).stdout)
            row["native_encode_mb_s"] = bench["encode_mb_s"]
            row["native_decode_mb_s"] = bench["decode_mb_s"]
            row["native_round_trip"] = bench["round_trip"]
            del ref
        else:
            train_ids = np.array([i for x in train for i in enc.encode(x)], dtype=np.int64)
        row["train_ids_per_kb"] = round(1000 * len(train_ids) / train_bytes, 1)
        test_ids = np.array([i for ids in held_ids for i in ids], dtype=np.int64)
        bits = wb_bits(train_ids, test_ids, vsize)
        row["ngram_bits_per_byte"] = {str(o): round(b / held_bytes, 4) for o, b in bits.items()}
        row["ngram_best_bits_per_byte"] = round(min(bits.values()) / held_bytes, 4)
        results["configs"][name] = row
        print(f"{name:18s} vocab {vsize:6d}  held-out {row['heldout_ids_per_kb']:6.1f}/KB  "
              f"md {row['ood_ids_per_kb'].get('repo_docs_markdown', 0):6.1f}  py {row['ood_ids_per_kb'].get('repo_python_code', 0):6.1f}  "
              f"1-ID words {100 * row.get('single_id_share', 0):5.1f}%  "
              f"n-gram bpb {row['ngram_bits_per_byte']}  "
              f"enc {row.get('native_encode_mb_s', '-')} MB/s  ({time.time() - t0:.0f}s)", flush=True)

    # Reference (Python) speed for one configuration, for scale.
    d = cv.Dictionary.load(str(work / "v0_32k.cv0d"))
    sample = b"".join(held)
    t = time.time()
    cv.Encoder(d).encode(sample)
    results["python_reference_encode_mb_s"] = round(len(sample) / (time.time() - t) / 1e6, 2)
    out = work / "compiler_v0_initial.json"
    out.write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
