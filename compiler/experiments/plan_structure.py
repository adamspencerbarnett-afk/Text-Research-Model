"""Pick the training structure for the 5M chat model v2 from sessions 31 and 33 (15 October 2026).

Per arm (S sequential, M mixed), averaged over the seeds present: probe v2 share (120 conversations) + history gain / 20
+ chat probe / 32; an arm whose stored-fact F1 falls more than 0.03 below the Q&A-only control's loses 0.2 (the
retention rule of docs/MULTITURN_PLAN.md). The higher score wins; with no session 33 results the mixed stage is used
(best history gain, keeps more Q&A detail, one stage). Writes results/chat5m_structure.json and prints the choice.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"


def load(p):
    try:
        return json.loads((R / p).read_text())
    except (OSError, ValueError):
        return None


def arm(names):
    vals = {"probe2": [], "history": [], "chat": [], "stored": []}
    for n in names:
        p2, mt, cp, qa = load(f"mtprobe2_qa_mt_{n}.json"), load(f"mtprobe_qa_mt_{n}.json"), load(f"probe_qa_mt_{n}.json"), load(f"mt_{n}.json")
        if p2:
            vals["probe2"].append(p2["total"]["ok"] / p2["total"]["n"])
        if mt and mt.get("history_gain"):
            vals["history"].append(mt["history_gain"]["history_gain_pct"])
        if cp:
            vals["chat"].append(sum(v["ok"] for v in cp["summary"].values()))
        if qa:
            vals["stored"].append(qa["groups"]["seen_network_with_own_exchange_in_context"]["f1"])
    return {k: (sum(v) / len(v) if v else None) for k, v in vals.items()} | {"seeds": len(names)}


def main():
    have2 = (R / "session33_done").exists()
    arms = {a: arm([a] + ([a + "2"] if have2 and (R / f"mt_{a}2.json").exists() else [])) for a in ("Q", "S", "M")}
    score = {}
    for a in ("S", "M"):
        v = arms[a]
        if v["probe2"] is None or v["history"] is None or v["chat"] is None:
            continue
        s = v["probe2"] + v["history"] / 20 + v["chat"] / 32
        if v["stored"] is not None and arms["Q"]["stored"] is not None and v["stored"] < arms["Q"]["stored"] - 0.03:
            s -= 0.2
        score[a] = round(s, 4)
    choice = ("sequential" if score.get("S", -9) > score.get("M", -9) else "mixed") if score else "mixed"
    out = {"structure": choice, "scores": score, "arms": arms, "session33_complete": have2,
           "rule": "probe v2 share + history gain/20 + chat probe/32, -0.2 if stored F1 < control - 0.03"}
    (R / "chat5m_structure.json").write_text(json.dumps(out, indent=1))
    print(choice)


if __name__ == "__main__":
    main()
