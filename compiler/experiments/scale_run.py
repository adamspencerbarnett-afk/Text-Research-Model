"""Phase 2.1, the scale run: does the table gain hold as the core grows?

    python compiler/experiments/scale_run.py data_v2 runs/scale --native data/cv0.exe --device cuda --amp \
        --sizes 5M,20M,50M --configs bpe_8k,v0_8k_ng64 --budgets 1800,2700,3600 --patience 4

For each core size and each configuration (standard BPE alone against compiler plus exact
tables), one run on corpus v2 books plus the Wikipedia split, with plateau stopping on the
validation split. The Wikipedia text goes first in the training stream and the books last, so
the validation split (the last 2%) stays on books, as in every earlier run; held-out books
and the out-of-domain files (including the 2 MiB Wikipedia head) are scored at the end. The
dictionary and the BPE merges are fitted on the whole training stream. Results go to one JSON
as each run finishes, so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ladder1  # noqa: E402
import train_lm  # noqa: E402

# width, layers, heads: about 5M, 20M and 50M parameters in the blocks
SIZES = {"5M": (256, 6, 8), "20M": (448, 8, 8), "50M": (640, 10, 10)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data"); ap.add_argument("work"); ap.add_argument("--native")
    ap.add_argument("--sizes", default="5M,20M,50M"); ap.add_argument("--configs", default="bpe_8k,v0_8k_ng64")
    ap.add_argument("--budgets", default="1800,2700,3600", help="training seconds per size, in --sizes order")
    ap.add_argument("--patience", type=int, default=4); ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--eval-every", type=float, default=120); ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--ctx", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1); ap.add_argument("--device", default="cuda"); ap.add_argument("--amp", action="store_true")
    ap.add_argument("--no-wiki", action="store_true", help="books only"); ap.add_argument("--out", default="results/scale_run.json")
    ap.add_argument("--max-steps", type=int, default=0, help="equal training across runs: stop at this many steps (budget is then a cap)")
    ap.add_argument("--align", action="store_true", help="training windows start at paragraph starts")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    books = sorted(glob.glob(f"{a.data}/train/*.txt"))
    wiki = [] if a.no_wiki else sorted(glob.glob(f"{a.data}/train_wiki/*.txt"))
    train_paths = wiki + books
    eval_sets = train_lm.eval_sets_from(a.data)
    out = Path(a.out)
    results = json.loads(out.read_text()) if out.exists() else {
        "run": "scale_run", "seed": a.seed, "amp": a.amp, "device": a.device,
        "corpus": {"train_books": len(books), "wiki_files": len(wiki),
                   "train_bytes": sum(Path(p).stat().st_size for p in train_paths),
                   "heldout": [Path(p).stem for p in eval_sets["heldout"]], "ood": [Path(p).stem for p in eval_sets["ood"]]},
        "sizes": {k: {"width": v[0], "layers": v[1], "heads": v[2]} for k, v in SIZES.items()}, "runs": {}}
    budgets = dict(zip(a.sizes.split(","), (float(b) for b in a.budgets.split(","))))
    concat = work / "train_concat.txt"
    if not concat.exists():
        concat.write_bytes(b"".join(Path(p).read_bytes() for p in train_paths))
    for size in a.sizes.split(","):
        width, layers, heads = SIZES[size]
        for name in a.configs.split(","):
            key = f"{size}/{name}"
            if key in results["runs"]:
                ladder1.say(f"{key}: already done, skipping"); continue
            enc, table, orders, ngram, topk = ladder1.make_encoding(name, train_paths, work, a.native)
            t0 = time.time()
            tokens, lens = enc.encode_file(str(concat), work)
            ladder1.say(f"{key}: {len(tokens):,} training codes ({time.time() - t0:.0f}s to encode), budget {budgets[size]:.0f}s")
            r = train_lm.run(enc, tokens, lens, eval_sets, work, budgets[size], width, layers, heads, a.ctx, a.batch, a.lr,
                             table, a.seed, a.eval_every, log=ladder1.say, save_path=str(work / f"scale_{size}_{name}.pt"),
                             table_orders=orders, ngram_orders=ngram, ngram_topk=topk, patience=a.patience,
                             min_delta=a.min_delta, device=a.device, amp=a.amp, max_steps=a.max_steps, align=a.align)
            r["config"], r["size"] = name, size
            results["runs"][key] = r
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(results, indent=1))
    ladder1.say("summary (held-out books / Wikipedia head, bits per byte; learned parameters; reply speed)")
    for key, r in results["runs"].items():
        ev = r["eval"]
        wiki_bpb = ev.get("ood", {}).get("enwik_head_2MiB", {}).get("bits_per_byte")
        ladder1.say(f"{key:18s} books {ev['heldout_overall']['bits_per_byte']:.4f}  wiki {wiki_bpb}  "
                    f"params {r['model']['params']['total']:,}  steps {r['training']['steps']}  "
                    f"seen {r['training']['bytes_seen'] / 1e6:.0f} MB  {r['generation']['bytes_per_s']:.0f} B/s  {r['training']['stop_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
