"""data_real/shared/qa/dense_top4.npy: the four most similar training questions of each real Q&A exchange (MiniLM, as
train_chat.py built runs/chat5m_v2/dense_top4.npy for Alpaca); the Q&A and buried kinds use them as look-alike answers.
For SQuAD exchanges only the question is embedded, not the paragraph.

    python -m chat22.data.build_real_neighbours [--device cpu]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .. import encoder as E

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    import dense_keys
    t0 = time.time()
    qa_file = ROOT / "data_real" / "shared" / "qa" / "qa_train.txt"
    qa = E.parse_exchanges(qa_file.read_bytes())
    keys = [q.split(b"Question:", 1)[1].strip() if q.startswith(b"Passage:") and b"Question:" in q else q for q, _ in qa]
    dk = dense_keys.DenseKeys(a.device, weight=2.0)
    V = dk.embed(keys)
    nbrs = np.zeros((len(qa), 4), dtype=np.int64)
    for s in range(0, len(qa), 2048):
        sims = V[s:s + 2048] @ V.T
        sims[torch.arange(sims.shape[0]), torch.arange(s, s + sims.shape[0])] = -2
        nbrs[s:s + 2048] = torch.topk(sims, 4).indices.cpu().numpy()
    np.save(qa_file.with_name("dense_top4.npy"), nbrs)
    print(f"{len(qa):,} exchanges, neighbours saved, {time.time() - t0:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
