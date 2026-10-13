# MASTER: Text Research Model

*Owner and research direction: Adam Barnett. Working branch `local-gpu-phase2`, pushed to `main` at each publish.*

## Goal

The most intelligent chat model that fits a small size and one consumer GPU (RTX 3090): knowledge
kept in memory outside the weights, a small network trained to read and copy from it, exact tools
where exactness is possible, measured honestly (answer accuracy on held-out questions, equal steps
and text, two seeds before building on a win, cost per run).

## Standing rules

- Precision: train in bf16; answer with bf16 weights and check int8. No fp32 or int4 runs (12 Oct).
- Math paused (12 Oct): no GSM8K stages and no worked maths in new mixes; its own line later (archive/math_line).
- Focus: text, Q&A and conversation, using the best current methods.
- Git holds scripts, logs, results and docs only: no model checkpoints (`*.pt` ignored, 13 Oct).
- Every run writes `results/*.json` with cost; findings go into `compiler/experiments/run_ledger.py`.
- Queue runs automatically as chain scripts in `compiler/experiments/chains/`, with `watch.sh` wake-ups.

## Best results now

- Best chat model: 5M, dense retrieval in training + right answers under reworded questions; probe 24 and 23/32 (two seeds). Local file `models/qa_best_5M_dense.pt`; recipe in `docs/SETUP.md` section 6.
- Core recipe from 100M: modern block, Muon + warmup-stable-decay, bf16, resumable.
- Answer path: CUDA graph, 0.08 s and 10 J per answer at 5M.

## Scope now

Session 27: the 100M core and its Q&A stage. Then build toward `HYBRID_DESIGN.html`: retrieval
confidence into the trust head, a reranker trained on outcomes, the passage tier, dialogue data,
the program tier, reasoning with thinking codes, the support check, continuous learning.

## File map

| path | what |
| --- | --- |
| `run_model.py` | user entry: train on a folder, chat with a model (`/add`, `/load`), option 3 = best model |
| `compiler/` | compiler v0, native encoder, tests |
| `compiler/experiments/train_lm.py`, `scale_run.py` | core training (blocks, optimisers, schedules, bf16, resume) |
| `compiler/experiments/m2_qa.py` | Q&A stage, memory, retrieval, training-window options, engine (`QAEngine`) |
| `compiler/experiments/fast_decode.py` | CUDA-graph answer step |
| `compiler/experiments/dense_keys.py`, `precision.py`, `cost.py` | dense retrieval key; weight rounding and kernels; cost meter |
| `compiler/experiments/core_eval.py`, `chat_probe.py`, `run_ledger.py` | report card, chat probe, ledger generator |
| `compiler/experiments/prepare_core_mix.py`, `evidence_parallel.py`, `prepare_qa.py` | data: mixes (D3, P50, P100), evidence pairs, Q&A data |
| `compiler/experiments/chains/` | sessions 19–27 and `watch.sh` (older sessions in `archive/experiments_oct2026/chains/`) |
| `docs/SETUP.md` | environment and full pipeline as commands |
| `docs/RUN_LEDGER.md` | every run, the queue, findings by session |
| `docs/ENCODER_RESEARCH_PLAN.md` | dated research log (section 8) |
| `HYBRID_DESIGN.html` | the next system's design, research plan, pseudocode, revision log |
| `README.md` | public summary |
| `archive/` | September research, Version 1, finished October experiments, the maths line |
