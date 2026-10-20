"""Long-context grid (15 October 2026, long-context review gap 5): answers by input length x evidence depth x task, with
the session memory off and on, through the chat engine. Our own RULER-style test: the bits-by-position curve and the one
code word said nothing about whether the model USES a fact buried in a long conversation or document.

Tasks (held-out names and phrasings only, G3.setup(test=True); filler = held-out conversations and held-out clean text):
  one         one fact stated in a turn at the given depth, asked at the end
  two         two-step: "My sister is called Kim." + "Kim lives in Oslo." -> "Where does my sister live?"
  distractor  the fact + the same field for someone else ("my brother lives in ...") at another depth
  absent      only the distractor: the right reply is "I don't have that" (not the other person's value)
  correction  the fact, then "Actually, ..." later: the new value, not the old
  doc         the fact sentence inside a long pasted document (one user turn) at the given depth

Each cell scores: answer right; wrong evidence used (distractor / old value); abstained; evidence available (the stating
turn is still in the window, or the value is in the evidence the memory returned). Retrieval = the engine's default
(word key + the model's dense key setting, recorded in the result).

    python compiler/experiments/long_grid.py runs/chat5m_v2/chat5m_v2_A.pt --device cuda [--n 8] [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import chat_engine as CE  # noqa: E402
import chat_tests as T  # noqa: E402
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402

TASKS = ["one", "two", "distractor", "absent", "correction", "doc"]
ACK = b"Got it. I'll remember that."


def lower(b) -> str:
    return (b.decode("utf-8", "replace") if isinstance(b, bytes) else b).lower()


class Grid:
    def __init__(self, eng, rng):
        self.eng, self.rng = eng, rng
        self.pairs = T.heldout_pairs()
        text = b"".join((ROOT / "data_core" / "eval_clean" / "heldout" / f).read_bytes()[:1_500_000] for f in ("books.txt", "fineweb.txt"))
        self.text = text.replace(b"\f", b"\n")
        self.bpc = len(self.text[:200_000]) / len(eng.enc.encode(self.text[:200_000])[0])     # bytes per code

    def n_codes(self, u, r) -> int:
        return len(self.eng.enc.encode(CE.render(u, r))[0])

    def other(self, f):
        for _ in range(50):
            d = G.Fact(self.rng, f.kind, f.field)
            if d.ent["id"] not in f.ent["id"] and f.ent["id"] not in d.ent["id"] and d.value != f.value:
                return d
        return None

    def conversation(self, L, placed):
        """Filler turns to about L codes with each (statement, depth) placed where its depth falls."""
        turns, sizes = [], []
        budget = L - sum(self.n_codes(st, ACK) for st, _ in placed)
        while sum(sizes) < budget - 32:
            for _ in range(30):                     # prefer a turn that does not overshoot the length
                u, r = self.pairs[int(self.rng.integers(len(self.pairs)))]
                c = self.n_codes(u, r)
                if sum(sizes) + c <= budget + 32:
                    break
            turns.append((u, r)); sizes.append(c)
        idx = {}
        for k, (st, dep) in enumerate(sorted(placed, key=lambda p: p[1])):
            cum = np.cumsum([0] + sizes)
            j = int(np.searchsorted(cum, dep * cum[-1]))
            j = min(j, len(turns))
            if idx and j <= max(idx.values()):
                j = max(idx.values()) + 1
            turns.insert(j, (st, ACK)); sizes.insert(j, self.n_codes(st, ACK)); idx[st] = j
        return turns, {st: turns.index((st, ACK)) for st, _ in placed}

    def case(self, task, L, depth):
        rng = self.rng
        f = G.Fact(rng)
        want, wrong, absent = [f.value], [], False
        if task == "two":
            rel, n = G.pick(rng, G.RELS), G.pick(rng, G.NAMES)
            f = G.Fact(rng, "p", None, {"e": n, "E": n, "ep": n + "'s", "Ep": n + "'s", "id": n})
            s1, s2 = f"My {rel} is called {n}.".encode(), f.stored()[1].encode()
            old = f.fmt; f.fmt = dict(old, e=f"my {rel}", E=f"My {rel}", ep=f"my {rel}'s", Ep=f"My {rel}'s")
            q = f._f(G.pick(rng, f.spec[2])); f.fmt = old
            turns, at = self.conversation(L, [(s1, depth), (s2, min(0.95, depth + 0.05))])
            return turns, q.encode(), [f.value], [], False, [at[s1], at[s2]]
        st = f.stored()[1].encode()
        if task == "doc":
            sent = b" " + st + b" "
            pre = [min((self.pairs[int(rng.integers(len(self.pairs)))] for _ in range(5)), key=lambda p: self.n_codes(*p))]
            n_bytes = int(max(64, L - self.n_codes(*pre[0]) - 30) * self.bpc)
            j = int(rng.integers(0, len(self.text) - n_bytes - 1))
            doc = self.text[j:j + n_bytes]
            cut = int(depth * len(doc)); cut = doc.find(b" ", cut) if doc.find(b" ", cut) > 0 else cut
            user = b"Here is a text:\n" + doc[:cut] + sent + doc[cut:]
            turns = pre + [(user, b"Okay.")]
            return turns, f.ask().encode(), want, [], False, [len(turns) - 1]
        placed = [] if task == "absent" else [(st, depth)]
        if task in ("distractor", "absent"):
            d = self.other(f)
            if d is None:
                return self.case(task, L, depth)
            dd = float(rng.choice([x for x in (0.1, 0.5, 0.9) if abs(x - depth) > 0.2])) if task == "distractor" else depth
            placed.append((d.stored()[1].encode(), dd)); wrong = [d.value]
            if task == "absent":
                want, absent = [], True
        corr = None
        if task == "correction":
            if not f.can_update():
                return self.case(task, L, depth)
            (_, new_s), _ = f.update()
            corr = ("Actually, " + new_s[0].lower() + new_s[1:]).encode()
            placed.append((corr, min(0.95, depth + 0.3)))
            want, wrong = [f.new_value], [f.value]
        turns, at = self.conversation(L, placed)
        stating = [] if absent else [at[corr if corr else st]]
        return turns, f.ask().encode(), want, wrong, absent, stating

    def run(self, task, L, depth, mem):
        turns, q, want, wrong, absent, stating = self.case(task, L, depth)
        eng = self.eng
        eng.forget()
        for u, r in turns:
            eng.remember(u, r)
        ans, info = eng.reply(q, 60, use_memory=mem)
        a = lower(ans)
        in_window = bool(stating) and min(stating) >= len(turns) - info["turns_in_window"]
        ev = " ".join(info["evidence"]).lower()
        avail = None if absent else (in_window or all(lower(w) in ev for w in want))
        abst = any(p.decode() in a for p in CE.ABSTAIN_B)
        right = (abst and not any(lower(w) in a for w in wrong)) if absent else \
                (all(lower(w) in a for w in want) and not any(lower(w) in a for w in wrong))
        return {"right": right, "wrong_evidence": any(lower(w) in a for w in wrong), "abstained": abst, "available": avail,
                "codes": sum(self.n_codes(u, r) for u, r in turns), "s": info["seconds"]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--device", default="cuda"); ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--lengths", default="512,1024,1900,4096"); ap.add_argument("--tag", default="")
    ap.add_argument("--quick", action="store_true", help="2 per cell, lengths 512 and 1024 (checks the script)")
    ap.add_argument("--tasks", default=",".join(TASKS)); ap.add_argument("--memory", default="both", choices=["both", "on", "off"])
    ap.add_argument("--passages", action="store_true", help="engine option: long turns and replies stored as pieces")
    ap.add_argument("--gate", default="", help="engine option 'word,meaning', e.g. 0.15,0.55: memory used only if it applies")
    a = ap.parse_args()
    G3.setup(test=True)
    rng = np.random.default_rng(11)
    gate = tuple(float(x) for x in a.gate.split(",")) if a.gate else None
    eng = CE.ChatEngine(str(ROOT / a.model), native=str(ROOT / "data" / "cv0.exe"), device=a.device, log=lambda *x: None,
                        passages=a.passages, gate=gate)
    grid = Grid(eng, rng)
    lengths = [512, 1024] if a.quick else [int(x) for x in a.lengths.split(",")]
    n = 2 if a.quick else a.n
    t0 = time.time(); rows = []
    mems = {"both": (False, True), "on": (True,), "off": (False,)}[a.memory]
    for task in a.tasks.split(","):
        for L in lengths:
            for depth in (0.1, 0.5, 0.9):
                for mem in mems:
                    rs = [grid.run(task, L, depth, mem) for _ in range(n)]
                    av = [r["available"] for r in rs if r["available"] is not None]
                    row = {"task": task, "length": L, "depth": depth, "memory": mem, "right": sum(r["right"] for r in rs), "n": n,
                           "wrong_evidence": sum(r["wrong_evidence"] for r in rs), "abstained": sum(r["abstained"] for r in rs),
                           "available": sum(av) if av else None, "codes_mean": int(np.mean([r["codes"] for r in rs])),
                           "s_per_answer": round(float(np.mean([r["s"] for r in rs])), 3)}
                    rows.append(row); print("grid", row, flush=True)
    summ = {}
    for key in ("task", "length", "depth", "memory"):
        summ[key] = {str(v): round(sum(r["right"] for r in rows if r[key] == v) / sum(r["n"] for r in rows if r[key] == v), 3)
                     for v in dict.fromkeys(r[key] for r in rows)}
    res = {"model": a.model, "window": eng.a.ctx, "passages": a.passages, "gate": gate, "retrieval": {"dense": getattr(eng.a, "dense", None), "word_key": True},
           "date": time.strftime("%Y-%m-%d %H:%M"), "n_per_cell": n, "summary_right_share": summ, "rows": rows,
           "seconds": round(time.time() - t0)}
    out = ROOT / "results" / f"long_grid_{Path(a.model).stem}{a.tag}.json"
    out.write_text(json.dumps(res, indent=1))
    print("summary", json.dumps(summ), "\nsaved", out, f"({res['seconds']} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
