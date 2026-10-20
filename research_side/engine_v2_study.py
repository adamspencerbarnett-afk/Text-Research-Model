"""Engine v2 part 1 against today's engine, on the retrieval study's cases (same seeds, same held-out names):
does the right item reach the memory block, how often a look-alike or the outdated value comes with it, whether
"never told" questions get a clean block, how many items, how fast. No model answers; no training.
    python research_side/engine_v2_study.py [--n 40] [--pool 30000]"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler", ROOT):
    sys.path.insert(0, str(p))
import long_grid as LG  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
from chat22 import encoder as E  # noqa: E402
from chat22.memory import Memory  # noqa: E402
from chat22.search_v2 import SearchV2  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=40); ap.add_argument("--pool", type=int, default=30000)
    a = ap.parse_args()
    torch.set_num_threads(6)
    import dense_keys
    dk = dense_keys.DenseKeys("cpu", weight=2.0)
    core = torch.load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
    enc = E.load(ROOT / core["encoding"]["dictionary"])
    rng = np.random.default_rng(5)                               # the same sample as the retrieval study
    pool = [json.loads(l) for l in open(ROOT / "data_chat22/facts_pool.jsonl", encoding="utf-8")]
    V_all = np.load(ROOT / "data_chat22/facts_pool_minilm.npy")
    pick = np.sort(rng.choice(len(pool), a.pool, replace=False))
    G3.setup(test=True)
    test = json.loads((ROOT / "data_skills/test_v2.json").read_text())
    upl = [(pool[i]["q"].encode(), pool[i]["a"].encode(), pool[i]["key"].encode()) for i in pick] + \
          [(q.encode(), x.encode(), q.encode()) for q, x in test["facts"]]
    vecs = torch.from_numpy(np.concatenate([V_all[pick], dk.embed([q.encode() for q, _ in test["facts"]]).numpy()]))

    class _E: pass
    _E.enc = enc
    grid = LG.Grid(_E, np.random.default_rng(11))
    cases = []
    for task in ("one", "two", "distractor", "absent", "correction"):
        for L in (1024, 4096):
            for j in range(a.n):
                cases.append((task, L) + tuple(grid.case(task, L, (0.1, 0.5, 0.9)[j % 3])))
    engines = {
        "today (chat22 default)": lambda: Memory(dk, gate=(0.15, 0.55)),
        "today + step 1 stricter + study fixes": lambda: Memory(dk, gate=(0.3, 0.6), gate_facts=(0.5, 0.8), k_facts=2, k_session=4, names_fixed=True, recent_search=False),
        "engine v2 part 1": lambda: SearchV2(Memory(dk)),
    }
    res = {}
    for name, make in engines.items():
        eng = make(); eng.add_facts(upl, vecs=vecs)
        conv, times, paths = {}, [], {}
        for task, L, turns, q, want, wrong, absent, stating in cases:
            eng.forget_session()
            for n_, (u, r) in enumerate(turns):
                eng.add_turn(u, r, n_ + 1)
            t0 = time.perf_counter()
            out = eng.search(q, [t[0] for t in turns[-2:]], None)
            times.append(1000 * (time.perf_counter() - t0))
            items, info = out
            paths[info.get("path", "-")] = paths.get(info.get("path", "-"), 0) + 1
            txt = [(it.q + b" " + it.a).decode("utf-8", "replace").lower() for it in items]
            sess_turns = {it.turn for it in items if it.source == "turn"}
            gold = {s + 1 for s in stating}
            c = conv.setdefault(task, {"cases": 0, "hit": 0, "wrong_value_in_block": 0, "empty": 0, "items": 0})
            c["cases"] += 1; c["items"] += len(items); c["empty"] += not items
            c["hit"] += bool(gold) and gold <= sess_turns
            c["wrong_value_in_block"] += any(any(w.lower() in t for w in wrong if w) for t in txt)
        up = {"questions": 0, "hit": 0, "items": 0}
        for group, q, kw, extra in test["items"]:
            if group not in ("taught", "nearmiss", "update"):
                continue
            eng.forget_session()
            items, _ = eng.search(q.encode(), [], None)
            txt = [(it.q + b" " + it.a).decode("utf-8", "replace").lower() for it in items]
            up["questions"] += 1; up["items"] += len(items)
            up["hit"] += any(all(any(k in t for k in g) for g in kw) for t in txt)
        for c in conv.values():
            c["items"] = round(c["items"] / c["cases"], 1)
        up["items"] = round(up["items"] / max(1, up["questions"]), 1)
        res[name] = {"conversation": conv, "uploaded": up, "ms_per_search_p50": round(float(np.median(times)), 1),
                     "ms_per_search_p95": round(float(np.percentile(times, 95)), 1), "paths": paths}
        print(name, json.dumps(res[name]), flush=True)
    (ROOT / "research_side" / "engine_v2_study.json").write_text(json.dumps({"date": time.strftime("%Y-%m-%d %H:%M"), "n": a.n, "results": res}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
