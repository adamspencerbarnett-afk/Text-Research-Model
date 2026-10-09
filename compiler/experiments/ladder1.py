"""Ladder 1: the same small model trained on different encodings of the same text, each for
the same training-time budget on the same machine.

    python compiler/experiments/ladder1.py DATA WORK --native ./cv0 --budget 600 --out results/ladder1.json

Configurations (all with case flags off, width 128, 4 layers, context 256 tokens):
    bytes            raw bytes
    v0_8k_plain      Compiler v0 dictionary, 8,192 IDs, no phrases
    v0_8k_phr3       8,192 IDs of which 1,024 are phrases of up to 3 units
    v0_8k_phr6       8,192 IDs of which 2,048 are phrases of up to 6 units
    hash4096x4096    no dictionary: (group, member) hash codes per scanner unit, 16.8M code space
    v0_8k_table      v0_8k_plain plus a hashed bigram input table of 2^20 rows (134M parameters)

Dictionaries and the hash reverse map are fitted on the training split only. Every result
is in bits per original byte on the held-out books, so the encodings compare on one scale.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import compiler_v0 as cv  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import train_lm  # noqa: E402

DICTS = {  # name: (vocab, phrases, max_phrase_units)
    "v0_8k_plain": (8192, 0, 1),
    "v0_8k_phr3": (8192, 1024, 3),
    "v0_8k_phr6": (8192, 2048, 6),
}
CONFIGS = ["bytes", "v0_8k_plain", "v0_8k_phr3", "v0_8k_phr6", "hash4096x4096", "v0_8k_table"]


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def fit_dictionary(name: str, train_paths: list[str], work: Path) -> Path:
    path = work / f"{name}.cv0d"
    if not path.exists():
        vocab, phrases, units = DICTS[name]
        t0 = time.time()
        d, _ = cv.fit(lambda: (Path(p).read_bytes() for p in train_paths), vocab, phrases, units, case_flags=False)
        d.save(str(path))
        say(f"fitted {name}: {d.vocab_size:,} IDs, hash {d.hash:016x} ({time.time() - t0:.0f}s)")
    return path


def fit_hash(groups: int, members: int, train_paths: list[str], work: Path) -> Path:
    path = work / f"hash{groups}x{members}.json"
    if not path.exists():
        t0 = time.time()
        h = LE.HashCodes(groups, members)
        h.fit(Path(p).read_bytes() for p in train_paths)
        h.save(str(path))
        say(f"fitted {h.name}: {h.collisions} ({time.time() - t0:.0f}s)")
    return path


def make_encoding(name: str, train_paths: list[str], work: Path, native: str | None):
    """Return (encoding, table_rows, table_orders) for a configuration name; see parse_config."""
    base, table, orders, ngram, topk = parse_config_full(name)
    if base == "bytes":
        return LE.Bytes(), table, orders, ngram, topk
    if base.startswith("bpe_"):
        vocab = int(base[4:].rstrip("k")) * 1024
        path = work / f"{base}.merges.json"
        if not path.exists():
            t0 = time.time()
            LE.StandardBPE.fit((Path(p).read_bytes() for p in train_paths), vocab, base).save(str(path))
            say(f"fitted {base}: {vocab:,} IDs ({time.time() - t0:.0f}s)")
        return LE.StandardBPE.load(str(path)), table, orders, ngram, topk
    if base.startswith("hash"):
        g, m = (int(x) for x in base[4:].split("x"))
        return LE.HashCodes.load(str(fit_hash(g, m, train_paths, work))), table, orders, ngram, topk
    return LE.V0Dict(str(fit_dictionary(base, train_paths, work)), native), table, orders, ngram, topk


def parse_config(name: str) -> tuple[str, int, tuple[int, ...]]:
    """See parse_config_full; this keeps the three-value form used by the tests."""
    return parse_config_full(name)[:3]


def parse_config_full(name: str) -> tuple[str, int, tuple[int, ...], tuple[int, ...], int]:
    """Split a configuration name into (base encoding, table rows, table orders).

    The base is ``bytes``, ``hashGxM`` or a dictionary name from DICTS. An optional
    ``_table[N][_tri]`` suffix adds hashed input tables: ``v0_8k_table`` is a 2^20-row bigram
    table on v0_8k_plain, ``v0_8k_table22`` has 2^22 rows, ``hash4096x4096_table20_tri`` puts
    bigram and trigram tables of 2^20 rows each on the hash encoder; ``_q4`` adds a 4-gram table too.
    A final ``_ng`` adds exact bigram and trigram follower tables mixed into the output, with
    16 followers per context, or ``_ng64`` for 64.
    """
    ngram, topk = (), 16
    m = re.search(r"_ng(\d*)$", name)
    if m:
        ngram, topk = (2, 3), (int(m.group(1)) if m.group(1) else 16)
        name = name[:m.start()]
    base, table, orders = name, 0, (2,)
    if "_table" in name:
        base, spec = name.split("_table", 1)
        orders = (2,)
        for suffix, o in (("_tri", (2, 3)), ("_q4", (2, 3, 4))):
            if spec.endswith(suffix):
                orders, spec = o, spec.removesuffix(suffix)
        table = 1 << (int(spec) if spec else 20)
    if base == "bytes" or base.startswith("hash") or base.startswith("bpe_"):
        return base, table, orders, ngram, topk
    if base not in DICTS and f"{base}_plain" in DICTS:
        base = f"{base}_plain"
    if base not in DICTS:
        raise KeyError(f"no dictionary configuration for {name!r}")
    return base, table, orders, ngram, topk


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("work")
    ap.add_argument("--native", default=None); ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--configs", default=",".join(CONFIGS)); ap.add_argument("--out", default="results/ladder1.json")
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--eval-every", type=float, default=120)
    ap.add_argument("--patience", type=int, default=0); ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--valid-share", type=float, default=0.02)
    ap.add_argument("--width", type=int, default=128); ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--device", default="cpu", help="cpu, cuda or cuda:N")
    ap.add_argument("--max-steps", type=int, default=0)
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    train_paths = sorted(glob.glob(f"{a.data}/train/*.txt"))
    eval_sets = train_lm.eval_sets_from(a.data)
    results = {"ladder": "ladder1", "budget_s": a.budget, "seed": a.seed, "corpus": {
        "train_books": len(train_paths), "train_bytes": sum(Path(p).stat().st_size for p in train_paths),
        "heldout": [Path(p).stem for p in eval_sets["heldout"]], "ood": [Path(p).stem for p in eval_sets["ood"]]},
        "configs": {}}
    out = Path(a.out)
    if out.exists():
        results = json.loads(out.read_text())
    for name in a.configs.split(","):
        if name in results["configs"]:
            say(f"{name}: already done, skipping")
            continue
        enc, table, orders, ngram, topk = make_encoding(name, train_paths, work, a.native)
        t0 = time.time()
        tokens, lens = train_lm.load_training(enc, a.data, work)
        say(f"{name}: {len(tokens):,} training tokens ({time.time() - t0:.0f}s to encode)")
        r = train_lm.run(enc, tokens, lens, eval_sets, work, a.budget, table_rows=table, seed=a.seed,
                         eval_every_s=a.eval_every, log=say, save_path=str(work / f"ladder1_{name}.pt"),
                         table_orders=orders, ngram_orders=ngram, ngram_topk=topk, valid_share=a.valid_share,
                         patience=a.patience, min_delta=a.min_delta, width=a.width, ctx=a.ctx, layers=a.layers, device=a.device, max_steps=a.max_steps)
        r["config"] = name
        (work / f"ladder1_{name}.json").write_text(json.dumps(r, indent=1))
        results["configs"][name] = r
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=1))
    say("summary (held-out books):")
    say(f"{'config':16s} {'IDs/KB':>7s} {'bpb':>7s} {'MB seen':>8s} {'steps':>6s} {'tok/s':>7s} {'gen B/s':>8s}  params")
    for name, r in results["configs"].items():
        h = r["eval"]["heldout_overall"]; t = r["training"]; g = r["generation"]
        say(f"{name:16s} {h['ids_per_kb']:7.1f} {h['bits_per_byte']:7.4f} {t['bytes_seen']/1e6:8.2f} {t['steps']:6d} "
            f"{t['tokens_per_s']:7.0f} {g['bytes_per_s']:8.0f}  {r['model']['params']['total']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
