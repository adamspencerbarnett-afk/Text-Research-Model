"""Score a Q&A model on the skills test (testset.py) and the 32-question chat probe, in several answering modes.

    .venv/Scripts/python.exe -B skills_lab/eval_skills.py MODEL.pt --name NAME [--modes orig,k1,k3close,k3all] [--threads 8]

Modes:  orig     the research engine as run_model.py uses it (one retrieved exchange, research exact match)
        k1       lab engine, one candidate, strict exact match
        k3close  lab engine, up to three candidates when their scores are close (within 0.25), oldest first
        k3all    lab engine, always the top three
Writes skills_lab/results/<name>.json (every answer) and prints a summary table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
ROOT = LAB.parent
sys.path.insert(0, str(LAB))
import torch  # noqa: E402

import engine as E  # noqa: E402
import testset as T  # noqa: E402

m2_qa = E.m2_qa
import chat_probe  # noqa: E402  (lab copy; m2_qa is already the lab module)

assert Path(chat_probe.__file__).resolve().parent == LAB / "lib" and E.m2_qa is sys.modules["m2_qa"]


def has(text: str, groups) -> bool:
    low = text.lower()
    return all(any(k in low for k in g) for g in groups)


def abstains(text: str) -> bool:
    low = text.lower().replace("’", "'")
    return any(p in low for p in T.ABSTAIN)


ITEMS, FACTS = T.items, T.facts_in_order       # test v1 by default; --testset swaps in another (test v2)


def run_mode(eng, mode: str, head=None):
    import decision as D

    def ask(q: str, history=None):
        if mode == "orig":
            out, info = eng.ask(q.encode(), 100, 3, fast=True, history=history)
            return out.decode("utf-8", "replace"), info.get("retrieved_question", info.get("memory"))
        k = 1 if mode == "k1" else 3
        out, info = eng.ask_k(q.encode(), k=k, policy="close" if mode == "k3close" else "all", history=history)
        if head is not None:     # the decision head replaces the reply when it judges the question unclear or unknown
            d = D.decide(head, D.features(eng, q.encode(), out, info))
            if d == "abstain":
                out = D.ABSTAIN_TEXT.encode()
            elif d == "clarify":
                out = D.clarify_text(eng, info).encode()
        return out.decode("utf-8", "replace"), info["candidates"]

    rows = []
    for group, q, kw, extra in ITEMS():
        ans, ev = ask(q)
        r = {"group": group, "question": q, "answer": ans, "evidence": ev}
        if group == "unknown":
            r["ok"] = abstains(ans)
        elif group == "clarify":
            r["ok"] = "?" in ans and has(ans, kw)
            fu, fu_ans = extra["followup"], None
            fu_ans, fu_ev = ask(fu, history=[(q.encode(), ans.encode())])
            rows.append(r)
            r = {"group": "followup", "question": f"{q} / {fu}", "answer": fu_ans, "evidence": fu_ev, "ok": has(fu_ans, extra["followup_kw"])}
        else:
            r["ok"] = has(ans, kw)
            if extra.get("wrong"):
                r["wrong_value"] = any(w in ans.lower() for w in extra["wrong"])
        r["abstained"] = abstains(ans)
        rows.append(r)
    probe = []
    for section, q, kw in chat_probe.PROBES:
        ans, ev = ask(q)
        probe.append({"section": section, "question": q, "answer": ans, "ok": has(ans, kw), "abstained": abstains(ans)})
    return rows, probe


def summary(rows, probe):
    s = {}
    for r in rows:
        g = s.setdefault(r["group"], [0, 0]); g[0] += r["ok"]; g[1] += 1
    out = {g: f"{a}/{n}" for g, (a, n) in s.items()}
    out["skills_total"] = f"{sum(r['ok'] for r in rows)}/{len(rows)}"
    out["update_old_value_given"] = sum(bool(r.get("wrong_value")) for r in rows)
    out["false_abstain_on_known"] = sum(r["abstained"] for r in rows if r["group"] in ("taught", "update", "nearmiss"))
    out["probe"] = f"{sum(p['ok'] for p in probe)}/{len(probe)}"
    out["probe_abstained"] = sum(p["abstained"] for p in probe)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--name", required=True)
    ap.add_argument("--modes", default="orig,k1,k3close,k3all"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--dense", type=float, default=2.0, help="MiniLM retrieval weight at answer time (2 = the research default; 0 = off)")
    ap.add_argument("--key-head", default="", help="our own learned key (K1/K2 key_head.pt) instead of MiniLM; implies --dense 0")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--testset", default="", help="a test file from build_skills_data.py (data_skills/test_v2.json); default: test v1")
    ap.add_argument("--decision", default="", help="decision head (decision.py) applied in its own mode; adds a mode '<mode>+head'")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    global ITEMS, FACTS
    if a.testset:
        tv = json.loads((ROOT / a.testset).read_text())
        ITEMS = lambda: [tuple(i) for i in tv["items"]]
        FACTS = lambda: [tuple(f) for f in tv["facts"]]
    t0 = time.time()
    eng = E.SkillEngine(str(ROOT / a.model) if not Path(a.model).is_absolute() else a.model, native=str(ROOT / "data" / "cv0.exe"),
                        device=a.device, dense=0.0 if a.key_head else a.dense, log=lambda *x: None)
    if a.key_head or a.dense == 0:
        eng.mem.learned = None        # the engine falls back to the model's own MiniLM setting when given 0: detach it
    if a.key_head:
        eng.mem.attach_learned(m2_qa.LearnedKeys(str(ROOT / a.key_head), str(ROOT / "data" / "cv0.exe"), a.device))
    for q, ans in chat_probe.TEACH + FACTS():
        eng.add(q.encode(), ans.encode())
    load_s = time.time() - t0
    res = {"model": a.model, "name": a.name, "testset": a.testset or "v1", "retrieval": a.key_head or (f"literal + MiniLM x{a.dense}" if a.dense else "literal only"), "date": time.strftime("%Y-%m-%d %H:%M"), "load_s": round(load_s), "modes": {}}
    head = None
    modes = a.modes.split(",")
    if a.decision:
        import decision as D
        head, hck = D.load_head(str(ROOT / a.decision))
        modes.append(hck["mode"] + "+head")
    for mode in modes:
        t1 = time.time()
        rows, probe = run_mode(eng, mode.split("+")[0], head if mode.endswith("+head") else None)
        res["modes"][mode] = {"summary": summary(rows, probe), "seconds": round(time.time() - t1), "rows": rows, "probe": probe}
        print(a.name, mode, json.dumps(res["modes"][mode]["summary"]), flush=True)
        (LAB / "results" / f"{a.name}.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
