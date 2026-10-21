# MASTER: Text Research Model

*Owner and research direction: Adam Barnett. Working branch `local-gpu-phase2`, pushed to `main` at each publish. Updated 20 Oct 2026 (handoff).*

## Goal

The most capable chat model that fits a small size and one consumer GPU (RTX 3090): knowledge kept in memory outside
the weights (a conversation memory of every turn and a permanent store of real facts), a small network trained to read
and copy from it, exact tools where exactness is possible, measured honestly (answer accuracy on held-out questions,
equal steps and data, two seeds when a choice is close, cost per run). Intended use at this size (Adam, 20 Oct): simple
chat such as a game character, and finding facts quickly for someone.

## Standing rules

- Strongest-model rule (Adam, 20 Oct): BEST is the strongest model we have; a model that is stronger overall replaces it.
  A candidate with no front lost and at least one won is adopted automatically by the loop; a mixed result is Adam's call.
- Precision: train in bf16; answer with bf16 weights and check int8. No fp32 or int4 runs (12 Oct).
- Math paused (12 Oct); focus on text, Q&A and conversation with the best current methods.
- Permanent store = real facts only; invented names and stories only in conversation-memory training kinds (16-17 Oct).
- Training data that works is not changed without a stated reason; one training at a time with `--gpu-frac`, at most 3/4 of the GPU and RAM.
- No model above 100M parameters without Adam; the 100M training starts on Adam's (or the collaborator's) go.
- Harmful-request handling is not a research target (a deterministic filter later, 20 Oct).
- Every run writes `results/*.json` with cost; findings go into LOG.md and `compiler/experiments/run_ledger.py`.
- Git (for collaborators): code, research record, the best model (LFS) and the training data bundles (`datasets/`, LFS,
  fetched on request); everything else in `runs/`, `data_*/`, `models/`, `archive/` stays local (`.git/info/exclude`
  lists legacy kept local). Push only when Adam asks.

## The best model and recipe

**BEST = BASE = C2b** (`runs/chat22/combo2b_5m_s1.pt`, best v3, 20 Oct; `run_model.py` option 3): round 5's recipe +
multi-part messages. `recipes/BEST.json` holds the recipe, the answer-time settings, the tests and baselines, every adopted
lesson and rejected change with evidence, the open fronts and the history. A model improves by improving this recipe:
every candidate retrains the one chat stage from the same 5M core with BASE plus its changes (about 2 hours), so adopted
pieces are in every later run.

- Train: core `runs/core_mt5m/scale_custom_v0_8k.pt`; `chat22.train --data-dir data_real/5m_mixed --mix v22 --parts
  longconv,longdocs,alpacaqa,multi2 --shares <BEST.json> --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts
  <stricter> --gpu-frac 0.6` (`bash compiler/experiments/chains/train_best.sh SEED NAME`).
- Answer: engine search v2 + `hop2` + `split`; store `data_real/shared/knowledge/knowledge_real.jsonl` (311,010 real facts).
- Data: `datasets/c2b_5m/c2b_5m_training_data.tar.xz` (590 MB, 2.75 GB unpacked; `datasets/README.md`).

## Results (C2b, one seed, vs round 5)

- Wins: multi-part messages 77% (27%); casual wording 90% (80%); conversation tasks 98% (92%); near-miss facts 70% (48%);
  story questions 87% (76%). Losses: story contradictions 0.08 per scene (0.00); told fact + stored fact 10% (32%).
- Held: right with memory 89%, invented 6.1%; recall 87/90/90% at 100/1k/10k turns; never told 90-100%; documents 57/39%;
  direct store questions 100%; told skills 47%, stored 69%; first word 20 ms.
- Not in C2b yet: counting / first thing told / summary (0%; learned to 90-100% in c1c, with costs).

## Where it stands (20 Oct, evening)

- The 5M loop (`recipes/LOOP.json`, `research_side/loop_5m.py`) is **paused** at C2b for Adam's review and the handoff.
  Next candidate: c4 = C2b + `reasoncr` (counting and recaps, no "first thing told") at 15% of recall.
- The 100M is ready to train: `docs/TRAIN_100M.md` (C2b recipe on the 109M core, `--rope-scale 2 --accum 2`; measured
  0.83 s/step, 11.5 GB peak; about 10-11 hours per seed; core in `datasets/100m/`).
- Lessons of 13 runs (design page #c2b): retrieval first (hop2, split), dose and form decide whether a skill helps, combine
  only clean pieces, read the answers behind a lost front.
- Later: a 16k window as a short extension stage; store-to-store chains; comparisons and time order over told facts.

## File map

| path | what |
| --- | --- |
| `README.md` | handoff entry: what C2b does, quickstart, the two memories and how they are trained, reproduce, continue, 100M |
| `run_model.py` | chat with C2b (option 3; hop2 and split on), train on a folder, device; `/add`, `/load`, `/forget`, `/memory`, `/info`, `/codes` |
| `recipes/BEST.json`, `recipes/LOOP.json` | the best recipe and all evidence; the loop's candidates and statuses |
| `research_side/loop_5m.py`, `compare_two_seeds.py` | the loop driver (pause options); the verdict against BEST (merges `_ev2new`) |
| `compiler/experiments/chains/train_best.sh`, `watch.sh` | train + test BEST's recipe (plus one change) on one seed; wake-up watcher |
| `chat22/` | the chat package: `engine.py` (memories, search v2, `hop2`, `split`, window, streaming), `memory.py`, `search_v2.py`, `window.py`, `model.py`, `tables.py`, `train.py` (`--parts`, `--accum`, `--rope-scale`, `--gpu-frac`), `windows_train.py` (every kind; switches incl. multi2, reasoncr, twomem2), `data/` builders, `tests/run_suite.py` + `tests/suite_v2.py` (the 20 Oct tests) + `tests/gate22.py` |
| `compiler/` | the lossless text compiler, core training (`experiments/train_lm.py`, `scale_run.py`), data preparation, probes |
| `datasets/` | `c2b_5m/` training data and `100m/` the 109M core (LFS, excluded from a normal clone; `datasets/README.md`) |
| `docs/SETUP.md`, `docs/TRAIN_100M.md` | 5M commands (environment, data, train, tests, loop); the 100M guide |
| `docs/CHAT5M_V2_DESIGN.html` | design and revision log; #c2b = C2b's recipe, methods, how to add reasoning skills |
| `docs/RUN_LEDGER.md`, `docs/RESEARCH_STORY.html`, `docs/TRAINING_MAP.html` | every run; the research story; the training map |
| `BRAIN_DESIGN.html`, `HYBRID_DESIGN.html`, `Traditional_Hybrid.html`, `COMPILER_AND_MODERN_METHODS.html`, `MODEL_EXPLAINED.html` | Adam's designs and forward ideas |
| `data_real/` | the permanent store (LFS), data notes (README), the 100M plan (`100m/plan.json`) |
| `data_chat22/`, `data_core/`, `data_mt/`, `data_skills/` | the held-out test files the suite reads (training data comes from the bundle) |
| `archive/`, `inference_lab/` | earlier research, local only |
