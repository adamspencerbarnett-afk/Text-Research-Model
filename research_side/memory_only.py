"""Side research (Adam, 16 Oct): can the conversation window be removed, so the model answers from memory alone?
Same cases (long_grid tasks, held-out names), two modes per case:
  window+memory  normal: earlier turns in the window, memory for what scrolled out
  memory-only    no earlier turns in the window at all (history=[]); every turn is in session memory and must come
                 back through the search; the model sees only the memory slot and the message
Old engine (arm A, v2.1 seed 1), CPU, few threads. No training; nothing else changed.
    python research_side/memory_only.py runs/chat5m_v21/chat5m_v21_s1.pt [n]"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler", ROOT): sys.path.insert(0, str(p))
torch.set_num_threads(6)
import chat_engine as CE, long_grid as LG, gen_skills_v3 as G3
from chat22.tests import scoring as S
G3.setup(test=True)
model, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 12
eng = CE.ChatEngine(str(ROOT / model), native=str(ROOT / "data" / "cv0.exe"), device="cpu", log=lambda *x: None)
grid = LG.Grid(eng, np.random.default_rng(11))
res, t0 = {}, time.time()
for task in ("one", "two", "distractor", "absent", "correction"):
    for L in (1024, 4096):
        cnt = {"window+memory": [0, 0], "memory-only": [0, 0]}
        for j in range(n):
            turns, q, want, wrong, absent, stating = grid.case(task, L, (0.1, 0.5, 0.9)[j % 3])
            for mode in cnt:
                eng.forget()
                for u, r in turns: eng.remember(u, r)
                ans, info = eng.reply(q, 60, history=[] if mode == "memory-only" else None)
                a = ans.decode("utf-8", "replace")
                ok = (S.abstained(a) and not any(S.has(a, w) for w in wrong)) if absent else S.right(a, [[w] for w in want], wrong)
                cnt[mode][0] += ok; cnt[mode][1] += 1
        res[f"{task}|{L}"] = {m: f"{k}/{t}" for m, (k, t) in cnt.items()}
        print(task, L, res[f"{task}|{L}"], flush=True)
out = ROOT / "research_side" / f"memory_only_{Path(model).stem}.json"
out.write_text(json.dumps({"model": model, "n_per_cell": n, "rows": res, "seconds": round(time.time() - t0)}, indent=1))
print("saved", out)
