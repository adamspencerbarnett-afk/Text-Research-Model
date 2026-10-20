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
- Git (19 Oct, for collaborators): code, research record (LOG, results, docs, recipes) and the current best model only. The best chat model, its core and the permanent store go through Git LFS (`git add -f`, see `.gitignore`); everything else in `runs/`, `data_*/`, `models/` and `archive/` stays local. Legacy kept local only is listed in `.git/info/exclude`. A new best replaces the shipped model.
- Every run writes `results/*.json` with cost; findings go into `compiler/experiments/run_ledger.py`.
- Queue runs automatically as chain scripts in `compiler/experiments/chains/`, with `watch.sh` wake-ups.

## The best recipe, cumulative (19 Oct)

`recipes/BEST.json` is the single record of everything that works: BEST (the model to beat: variant A, two seeds), BASE (the recipe new rounds build on: round 4, A + real memory), the data every kind reads, the answer-time settings, every adopted lesson with its evidence, every rejected change with its evidence, and the open fronts with their plan. `bash compiler/experiments/chains/train_best.sh SEED NAME [one change]` trains the base recipe (plus one change) and tests it; `python research_side/compare_two_seeds.py NAME_s1 NAME_s2` gives the verdict against BEST (no front lost beyond noise, more wins than losses, two seeds). A winner becomes the new BEST and BASE in the file; the history stays.

## Best results now (19 Oct)

- Best chat model: **round 5, BEST v2** (`runs/chat22/mix_5m_s1.pt` / `mix_5m_s2.pt`, two seeds; seed 2 is the one used and shipped in git; `run_model.py` option 3; recipe = recipes/BEST.json): A's data + real sessions on the mixed pool. vs A's two seeds: 6 significant wins (right with memory 88% vs 84%, corrections and distractors from memory, documents 55%/41% vs 45%/30%), 0 losses; story contradictions 0.00 under the fixed metric. Earlier line: Two seeds vs A's two seeds (engine v2, fixed suite): better on distractors from memory 368 vs 342/408, near-miss skills 59 vs 46/80, story questions 117 vs 100/150, told skills 169 vs 151; level on never told, two-step, recall (flat 100 -> 10,000 turns), fluency; worse on skills "unknown" 46 vs 65/80 and story contradictions 1.62 vs 0.38 (round 5 targets these).
- **To reproduce round 4:** `bash compiler/experiments/chains/session52_hybrid.sh` (seed 1) / `session53_hybrid_s2.sh` (seed 2): `chat22.train --data-dir data_real/5m_hybrid --mix v22 --parts longconv,longdocs,alpacaqa --shares <data_real/5m/recipe.json> --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts <stricter> --gpu-frac 0.6`. Data: Alpaca Q&A (data_qa), P100 rehearsal + novels (data_chat22, data_core), skills v4c, dialogues; real parts from data_real/shared: longconv (sessions on the real pool), look (facts_pool_real), knowledge (knowledge_real.jsonl, 311k, the deployed store). The training log prints the resolved "data sources" and "effective shares". Tests: `run_suite --tier full --engine-opts '{"search": "v2"}' --tag _ev2full` (the suite rebuilds the engine after stored skills since 18 Oct).
- The improvement loop (18-19 Oct): one change per round as a `--parts` switch (old recipes verified byte-identical), one seed, full suite vs A with a 2-SE check, second seed only for a win; LOG.md holds every round's numbers and verdict.

## Earlier best (17 Oct)

- Best chat model: **variant A** (`runs/chat22/v24a_s1.pt`, one seed): v2.2's recipe + any-length conversations 12% + full-length documents. Best or tied-best on 26 of 39 suite fronts. Memory on: one fact 87%, never told 98%, two-step 90%; recall of 30 facts 25/24/25 at 100/1k/10k turns; told skills 79/180; grounding 79% right, 10% invented. Open: documents ~30% (v2.1: 80%), skill updates 8/40, story consistency.
- Answer path: chat22 engine; search engine v2 rules (`search="v2"`) beat today's search on every memory line at ~0.1 ms; permanent fact store = real facts only (`data_chat22/knowledge_wiki.jsonl`, Wikipedia, `facts_path`); memory setting: step 1's stricter.
- Core: 5M modern block, Muon + WSD, P100 mix (`runs/core_mt5m`); 50M and 100M cores kept (`runs/core_P50`, `runs/core_P100`) for after the 5M study.

## Scope now (17 Oct)

The merged model (design: `docs/CHAT5M_V2_DESIGN.html#v24`): three training stages instead of the labs' five: (1) core, have it; (2) one merged chat stage with every specialised kind at its evidence-based share; (3) a grounding stage of auto-graded answer pairs on top. At answer time: engine v2 + the Wikipedia fact store. Later: 16k window stage on the winner; reinforcement learning only at the 100M.
Running: session 43 (variants B, C) and session 44 (D, E documents; F long windows); then weight-merge test, learning-rate decision, merged recipe, stage 2 overnight (2 seeds).
Rules: permanent knowledge is real facts only (invented material only in conversation memory kinds); training data that works is not changed without a stated reason and Adam's approval; no 100M training until the 5M study is complete; no new downloads until the 100M (real datasets already on disk in `data_uploads/`); one training at a time with `--gpu-frac`; long context capped at 65,536 codes.
Pending decisions (Adam): rebalance any-length never-told endings 30% -> 15%; grounding stage; SQuAD (on disk) for evidence reading.

## File map

| path | what |
| --- | --- |
| `run_model.py` | user entry: train on a folder, chat with a model (`/add`, `/load`), option 3 = best model, option 4 = device and threads; core chat KV-cached and streamed |
| `compiler/experiments/k2_question_key.py` | K2, our own learned retrieval key (session 32); uses `gen_skills.py` and `testset.py` (copies of the lab's) |
| `compiler/experiments/chat_engine.py`, `train_chat.py`, `chat_tests.py`, `plan_structure.py` | 5M chat model v2 (session 35): shared window layout + conversation engine (session memory, streaming), trainer (`--stage v21`: recall v2, buried evidence, pointer target), tests, structure picker; design in `docs/CHAT5M_V2_DESIGN.html` |
| `chat22/` | the chat package (16-17 Oct; `train.py --parts` switches longconv,crowd,stored,budget,longdocs,mem1024 and `--gpu-frac`; `search_v2.py` engine v2 + JudgeScorer; `judge.py`; `data/build_v23.py`, `data/build_search_data.py`, `data/build_knowledge.py` -> the Wikipedia fact store): `encoder.py`, `blocks.py`, `tables.py`, `model.py` (ChatLM), `window.py` (the one layout), `memory.py` (two pools), `engine.py`, `tools.py`, `windows_train.py` (all kinds incl. story, docqa), `train.py` (held-out plateau), `data/build_data.py` -> `data_chat22/`, `data/story_world.py`, `tests/` (`test_equivalence`, `test_engine`, `audit_windows`, `run_suite`, `scoring`, `gate22`) |
| `compiler/experiments/long_grid.py` | long-context grid: 6 tasks x length x evidence depth x session memory, through the chat engine (session 37) |
| `docs/RESEARCH_STORY.html` | the research story (local page): the journey from September to 17 October, 158 tests with settings, data, results and lessons; built by `docs/story/build.py` from `docs/story/tests.json` + `story.json` (`story_src.py`; 17 Oct additions in `story_update_17oct.py`) |
| `docs/TRAINING_MAP.html` | training map (local page): core text % -> cores -> fine-tuning data % -> recipes -> models (seeds) -> results; hover/click highlights everything connected; "I want a model that..." picks the best model and its recipe |
| `docs/REASONING_MAP.html` | interactive map of the reasoning research (local page) |
| `inference_lab/` | inference speed lab (own copies of the scripts in `lib/`): `bench.py`, `bench_gpu.py`, `chain_gpu.sh`, `RESULTS.md`; local only since 19 Oct (finished; the result lives in `fast_decode.py`) |
| `compiler/` | compiler v0, native encoder, tests |
| `compiler/experiments/train_lm.py`, `scale_run.py` | core training (blocks, optimisers, schedules, bf16, resume) |
| `compiler/experiments/m2_qa.py` | Q&A stage, memory, retrieval, training-window options, engine (`QAEngine`) |
| `compiler/experiments/fast_decode.py` | CUDA-graph answer step |
| `compiler/experiments/dense_keys.py`, `precision.py`, `cost.py` | dense retrieval key; weight rounding and kernels; cost meter |
| `compiler/experiments/core_eval.py`, `chat_probe.py`, `run_ledger.py` | report card, chat probe, ledger generator |
| `compiler/experiments/prepare_core_mix.py`, `evidence_parallel.py`, `prepare_qa.py` | data: mixes (D3, P50, P100), evidence pairs, Q&A data |
| `compiler/experiments/chains/` | running and recent sessions (42-44, step 1, longconv generation) and `watch.sh`; finished sessions 19-41 in `archive/experiments_oct2026/chains/` |
| `docs/SETUP.md` | environment, what git carries, the training data of the best recipe (paths, sizes, builders), train and test commands (19 Oct) |
| `docs/RUN_LEDGER.md` | every run, the queue, findings by session |
| `docs/ENCODER_RESEARCH_PLAN.md` | dated research log (section 8) |
| `HYBRID_DESIGN.html` | the next system's design, research plan, pseudocode, revision log |
| `README.md` | collaborator entry: quickstart (clone with LFS, chat with the best model, check the suite), best results, the improvement loop, reading order |
| `recipes/BEST.json` | the cumulative best recipe; `compiler/experiments/chains/train_best.sh` and `research_side/compare_two_seeds.py` use it |
| `research_side/` | side studies: memory-only answering, retrieval study, engine v2 speed and study |
| `data_chat22/` | chat22 data: rehearsal sample, long pools, dialogues, facts_pool (training look-alikes only), knowledge_wiki (permanent store), longconv sessions, search_v2 sets |
| `data_real/` | test folder of real, non-fiction datasets (17 Oct; `chat22/data/build_real.py`; audit and contents in its README): real Q&A 95,886 (replaces Alpaca for tests), fact store 215,602, non-fiction text 555 MB and long documents 9,090 (no novels), real look-alikes, human OASST threads, held-out sets |
| `data_uploads/`, `data_qa_squad/` | downloaded real datasets (SQuAD, ARC, OpenBookQA, CommonsenseQA, UltraChat, OASST, Dolly, FineWeb-Edu) for the merged model and the 100M |
| `archive/` | September research, Version 1, finished October experiments (cleanup of 17 Oct: 96.7 GB, list in `archive/experiments_oct2026/MOVED_17OCT2026.md`), the maths line |
