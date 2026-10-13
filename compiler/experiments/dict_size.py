"""Dictionary-size study, stage 1 (CPU only): how many codes does each dictionary size need per text?

    python -I compiler/experiments/dict_size.py data_core/mix_D3s_5M runs/dict_size --native data/cv0.exe \
        --sizes 8k,16k,32k,64k --out results/dict_size_codes.json

For each size, a plain v0 dictionary (no phrases, no case flags, as every model since 9 October) is fitted
on the mix's training text only, then every held-out file of the mix is encoded and counted. Reported per
size and per source: codes per KB (fewer codes = fewer generation steps per answer and more text per
window), the share of held-out bytes that fall back to single-byte codes, fit time and dictionary hash.
Stage 2 (GPU, queued separately) trains 5M cores and Q&A stages at each size at equal text and compares
bits per byte, answer scores, and seconds and joules per answer.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import ladder1  # noqa: E402
import ladder_encodings as LE  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mix"); ap.add_argument("work"); ap.add_argument("--native", default=None)
    ap.add_argument("--sizes", default="8k,16k,32k,64k"); ap.add_argument("--out", default="results/dict_size_codes.json")
    a = ap.parse_args()
    work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    train = sorted(glob.glob(f"{a.mix}/train/*.txt"))
    held = sorted(glob.glob(f"{a.mix}/heldout/*.txt"))
    out = {"mix": a.mix, "train_bytes": sum(Path(p).stat().st_size for p in train), "sizes": {}}
    if Path(a.out).exists():
        out = json.loads(Path(a.out).read_text())
    for size in a.sizes.split(","):
        name = f"v0_{size}_plain"
        if name in out["sizes"]:
            print(f"{name}: done, skipping"); continue
        t0 = time.time()
        path = ladder1.fit_dictionary(name, train, work)
        fit_s = time.time() - t0
        enc = LE.V0Dict(str(path), a.native)
        per, tot_codes, tot_bytes, tot_fallback = {}, 0, 0, 0
        for f in held:
            blob = Path(f).read_bytes()
            codes, lens = enc.encode(blob)
            codes = np.asarray(codes)
            fallback = int((codes < 256).sum())
            per[Path(f).stem] = {"bytes": len(blob), "codes": int(len(codes)), "codes_per_kb": round(1000 * len(codes) / max(1, len(blob)), 1),
                                 "byte_fallback_share": round(fallback / max(1, len(codes)), 4)}
            tot_codes += len(codes); tot_bytes += len(blob); tot_fallback += fallback
        out["sizes"][name] = {"vocab": enc.vocab, "dictionary_hash": f"{enc.d.hash:016x}", "fit_s": round(fit_s, 1),
                              "codes_per_kb": round(1000 * tot_codes / max(1, tot_bytes), 1), "bytes_per_code": round(tot_bytes / max(1, tot_codes), 3),
                              "byte_fallback_share": round(tot_fallback / max(1, tot_codes), 4), "per_source": per}
        base = out["sizes"].get("v0_8k_plain")
        rel = f", {out['sizes'][name]['codes_per_kb'] / base['codes_per_kb']:.3f}x the 8k codes" if base else ""
        print(f"{name}: {out['sizes'][name]['codes_per_kb']} codes per KB, {out['sizes'][name]['bytes_per_code']} bytes per code{rel} (fit {fit_s:.0f}s)", flush=True)
        Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
