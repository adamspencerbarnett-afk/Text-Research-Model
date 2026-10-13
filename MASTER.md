# MASTER: Text Research Model

*Owner and research direction: Adam Barnett. Working branch `local-gpu-phase2`, merged to `main` at each publish.*

## Goal

The most intelligent chat model that fits a small size and one consumer GPU (RTX 3090): knowledge
kept in memory outside the weights, a small network trained to read and copy from it, exact tools
where exactness is possible, measured honestly (answer accuracy on held-out questions, equal steps
and text, two seeds before building on a win, cost per run).

## Standing rules

- Precision: train in bf16; answer with bf16 weights and check int8. No fp32 or int4 runs (12 Oct).
- Math paused (12 Oct): no GSM8K stages; math gets its own research line later (HYBRID_DESIGN 3.11).
- Focus: text, Q&A and conversation, using the best current methods.
- Large checkpoints (50M and up) stay local in `runs/`; results JSON and docs go to git.
- Every run writes `results/*.json` with cost; findings go into `compiler/experiments/run_ledger.py`.
- Queue runs automatically as chain scripts in `compiler/experiments/chains/`, with `watch.sh` wake-ups.

## Scope now

Build toward the hybrid memory model in `HYBRID_DESIGN.html`: memory tiers (exact, passage,
session, program, learned), a retrieval cascade, a retrieval-trained generator of 100–300M,
the CUDA-graph decoder with copy-as-draft, reasoning with thinking codes and self-requested
search, a support check, and continuous learning.

## File map

| path | what |
| --- | --- |
| `run_model.py` | user entry: train on a folder, chat with a model (`/add`, `/load`) |
| `compiler/` | compiler v0, native encoder |
| `compiler/experiments/train_lm.py`, `scale_run.py` | core training (blocks, optimisers, schedules, bf16) |
| `compiler/experiments/m2_qa.py` | Q&A stage, memory, retrieval, engine (`QAEngine`) |
| `compiler/experiments/fast_decode.py` | CUDA-graph answer step |
| `compiler/experiments/dense_keys.py`, `precision.py`, `cost.py` | dense retrieval key; weight rounding and kernels; cost meter |
| `compiler/experiments/core_eval.py`, `chat_probe.py`, `run_ledger.py` | report card, chat probe, ledger generator |
| `compiler/experiments/chains/` | the queued sessions and `watch.sh` |
| `docs/RUN_LEDGER.md` | every run, the queue, findings by session |
| `docs/ENCODER_RESEARCH_PLAN.md` | dated research log (section 8) |
| `HYBRID_DESIGN.html` | the next system's design, research plan, pseudocode |
| `README.md` | public summary |
| `archive/` | September semantic-compiler research and frozen Version 1 |
