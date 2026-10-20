"""data_real/<size>/plan.json: the training plan of each model size on the real datasets: step budget, shares, files, and
how many times each dataset is seen (passes) at the step budget, from the files as built. Rules from our results:
real data at most ~2 passes (Q&A, plain text, long documents best near 1), generated kinds need variety more than volume.

    python -m chat22.data.plan_real
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
D = ROOT / "data_real"
BYTES_PER_CODE = 3.6
WINDOW = 2049
# draft merged shares (CHAT5M_V2_DESIGN.html#v24), to be set from runs D, E, F
SHARES = {"qa": 0.13, "skills": 0.13, "convo": 0.10, "recall": 0.14, "buried": 0.14, "docqa": 0.06, "story": 0.08,
          "rehearse": 0.11, "longconv": 0.11}
SIZES = {
    "5m": {"core": "runs/core_mt5m/scale_custom_v0_8k.pt", "batch": 16, "min_steps": 30000, "max_steps": 48000, "lr": 1e-4,
           "speed": "measured: 290 steps/min alone (variant A) -> 48k steps ~2.75 h",
           "why": "every kind was still improving at 26-33k steps (A, B); with 95,886 real Q&A the 2-pass limit allows ~48k steps"},
    "100m": {"core": "runs/core_P100/scale_custom_v0_8k.pt", "batch": 16, "min_steps": 25000, "max_steps": 40000, "lr": 1e-4,
             "speed": "estimate: 2.5-3 s/step at 2,048 codes (the 100M Q&A stage ran 1.45 s/step at 1,024) -> 40k steps ~28-33 h; "
                      "measure on the first run",
             "why": "compute-bound on one RTX 3090 (about two nights); a larger model memorizes faster, so real data near 1 pass"},
}


def count_lines(p: Path) -> int:
    with open(p, encoding="utf-8") as f:
        return sum(1 for _ in f)


def main() -> int:
    sh = D / "shared"
    qa = sum(1 for x in (sh / "qa" / "qa_train.txt").read_text(encoding="utf-8").split("\n\n") if x.strip())
    skills = count_lines(sh / "skills" / "train_v4c.jsonl") - 2000
    convo = count_lines(sh / "convo" / "dialogues_train.jsonl")
    lc_files = sorted((sh / "longconv").glob("longconv_*.jsonl")) if (sh / "longconv").exists() else []
    longconv = int(sum(count_lines(f) for f in lc_files) * 0.98)
    for size, cfg in SIZES.items():
        sd = D / size
        reh = sorted((sd / "text").glob("rehearse_REAL_whole_*MB_seed1.txt"))
        long_f = sd / "long" / "long_over8k_real.txt"
        if not reh or not long_f.exists():
            print(size, "not built yet"); continue
        reh_codes = reh[-1].stat().st_size / BYTES_PER_CODE
        long_docs = long_f.read_bytes().count(b"\f") + 1
        long_codes = long_f.stat().st_size / BYTES_PER_CODE
        W = cfg["max_steps"] * cfg["batch"]
        uses = {"Q&A (qa + 60% of buried)": (W * (SHARES["qa"] + 0.6 * SHARES["buried"]), qa),
                "skills (skills + 40% of buried)": (W * (SHARES["skills"] + 0.4 * SHARES["buried"]), skills),
                "conversations (convo)": (W * SHARES["convo"], convo),
                "any-length conversations": (W * SHARES["longconv"], longconv or float("nan")),
                "plain text (windows of 2,049 codes)": (W * SHARES["rehearse"] * WINDOW, reh_codes),
                "long documents (story continuation half)": (W * SHARES["story"] / 2 * WINDOW, long_codes)}
        passes = {k: {"uses": round(u), "have": round(h), "passes": round(u / h, 2) if h == h and h else None} for k, (u, h) in uses.items()}
        plan = {"size": size, **cfg, "shares": SHARES, "windows_at_max_steps": W,
                "files": {"qa": "shared/qa/qa_train.txt (+ dense_top4.npy)", "qa_heldout": "shared/eval/qa_heldout.txt",
                          "skills": "shared/skills/train_v4c.jsonl", "dialogues": "shared/convo/dialogues_train.jsonl",
                          "rehearsal": f"{size}/text/{reh[-1].name}", "long_and_story": f"{size}/long/long_over8k_real.txt",
                          "look_alikes": "shared/look/facts_pool_real.jsonl", "any_length": "shared/longconv/longconv_*.jsonl",
                          "fact_store_at_answer_time": "shared/knowledge/knowledge_real.jsonl"},
                "data": {"qa": qa, "skills": skills, "dialogues": convo, "longconv_rows": longconv,
                         "rehearsal_codes": round(reh_codes), "long_docs": long_docs, "long_codes": round(long_codes)},
                "passes_at_max_steps": passes,
                "train_command": f"python -m chat22.train --name real_{size}_s1 --data-dir data_real/{size} --core {cfg['core']} "
                                 f"--mix v22 --parts longconv,longdocs --shares '{json.dumps(SHARES)}' --min-steps {cfg['min_steps']} "
                                 f"--max-steps {cfg['max_steps']} --patience 10 --gpu-frac 0.6"}
        if size == "100m":
            plan["before_training"] = ["gradient accumulation in chat22/train.py (batch 16 x 2,049 codes will not fit 24 GB at 100M)",
                                       "rope scale for 2,048 codes if the 100M core was trained at 1,024",
                                       "Q&A near 1 pass needs ~140k real exchanges: Natural Questions / TriviaQA / HotpotQA / SciQ (download approval)",
                                       "skills v5 for variety; more any-length sessions (8 shards) for variety"]
        (sd / "plan.json").write_text(json.dumps(plan, indent=1))
        print(f"== {size}: {cfg['max_steps']:,} steps x {cfg['batch']} = {W:,} windows")
        for k, v in passes.items():
            print(f"   {k:44} uses {v['uses']:>12,}  have {v['have']:>12,}  passes {v['passes']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
