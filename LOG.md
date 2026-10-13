# LOG

Short session entries: date, task, files touched, decisions, blockers, next action. Full results are in `docs/RUN_LEDGER.md`.

## 2026-10-12

- **Task:** sessions 16–19 (modern block, depth sweep, modern-core Q&A, precision); speed and retrieval engineering; hybrid design; repository published to `main`.
- **Results:** depth 3 = 6 > 11 > 16 layers; modern block −1 to −7% text error, answers level at 5M; bf16 training = fp32 (0.1%) at 2.2x speed; int8 weights free at 5M and 50M; int4 −13% stored F1 at 5M; CUDA-graph answer step 3.6–9.3x faster with identical replies; low-bit kernels slower at these sizes; dense key reworded recall 75.1% → 80.9%.
- **Files:** `fast_decode.py`, `precision.py`, `dense_keys.py`, `cost.py` (new); `m2_qa.py` (graph path, kernels, dense keys, `--amp`), `train_lm.py` / `scale_run.py` (Muon, WSD, fused), `chat_probe.py`, `core_eval.py`, `run_model.py` (`/load`, graph chat), chains 18–21, `HYBRID_DESIGN.html`, `README.md`, `MASTER.md`, `LOG.md`.
- **Decisions (Adam):** math paused; bf16 + int8 only; focus on text, Q&A, conversation; build the hybrid design.
- **Blockers:** none.
- **Next:** session 20 (Muon / WSD recipe, running), session 21 (dense retrieval in training), then session 22 (core reranking, bad-evidence training, copy-as-draft decoding).

## 2026-10-13 (overnight)

- **Task:** sessions 20 (training recipe) and 21 (dense retrieval) completed; session 22 written and started.
- **Results:** Muon + warmup-stable-decay best text core (-0.9% bits/byte), WSD alone -0.8% at no cost, answers unchanged at 5M; dense retrieval: retrieved-answer F1 +11-13%, reworded 0.259 -> 0.276, new 0.290 -> 0.299, probe 22/32.
- **Files:** `m2_qa.py` (`--distractor-p`), `chains/session22_evidence.sh`, ledger, README.
- **Decisions:** WSD adopted for larger cores; Muon + WSD pending seed 2; dense keys adopted.
- **Next:** session 22 (bf16 Q&A control, bad-evidence training, Muon + WSD seed 2), then session 23 (passage tier, evidence cache, in-domain retrieval key).
- **Session 22 (done 05:56):** bf16 Q&A stage is level with fp32 (stored 0.891); bad-evidence training lifts network-alone F1 0.229 -> 0.274 and stored to 0.901 / 62% exact, reworded falls; Muon + WSD text gain confirmed on seed 2 (1.1826 vs 1.1810). Session 23 (core reranking, copy under rewording) running.
- **Session 23 (done 06:46):** copy under rewording (dense + --perturb-self 0.3) gives probe 24/32 (best 5M, equals best 50M), reworded section 5/5, new 0.311, stored 0.901 / 62%; adopted. Core reranking by query likelihood hurts (probe 20/32); rejected. Session 24 (all lessons combined) running.
- **Session 24 (done 07:15):** all lessons combined: probe 22/32, exact 67%; lessons partly cancel at 5M. Best chat model (23 B) copied to `models/qa_best_5M_dense.pt` and made the `run_model.py` default. Session 25 (seed 2 of 23 B) running.
- **Session 25 (done 07:47):** seed 2 of the best chat model: probe 23/32, reworded 4/5 (seed 1: 24, 5/5); adopted. Model upload to GitHub timing out (HTTP 408) since 07:15; file is local in `models/` and retried at each wake-up.

## 2026-10-13 (morning)

- **Task:** session 26 logged; repository cleanup; README and HYBRID_DESIGN updated from sessions 20–26; 100M core prepared.
- **Results:** synonym rewording ties random edits (option kept); finding: the trust head never saw retrieval confidence in the stage-2 runs (likely why combined lessons cancelled).
- **Files:** archived 11 old scripts and 18 old chains (`archive/experiments_oct2026/`, `archive/math_line/`); `docs/HANDOFF_LOCAL.md` archived, `docs/SETUP.md` new; model checkpoints untracked (`*.pt` in `.gitignore`); `requirements.txt` to the verified environment; `train_lm.py` / `scale_run.py` resume and rotary-table cache; `prepare_core_mix.py` recipe P100; `chains/session27_100m.sh`.
- **Decisions (Adam):** no models in git; 100M next.
- **Next:** session 27 (100M core ~11 h, then Q&A stage and probes); then the 5M retrieval-confidence test.
- **09:35:** BUG: native encoder silently stopped reading at 2 GiB (C++ stringstream on Windows). 50M core lost its last 184 MB (30% of its Wikipedia); first 100M start would have lost 1.06 GB. Fixed (chunked 64-bit reads), round trip of 3.2 GB byte-identical, tests pass; 100M restarted on 885.9M codes. Dictionary counts: 16k/32k/64k = 0.92x/0.87x/0.84x the 8k codes.
- **10:15:** EVALUATION LEAK: evidence pairs carry held-out paragraphs into training (P100/P50 mixes: 20-23% of held-out Wikipedia, 10% of web, 83-95% of held-out evidence; 5M mixes 0.4%); evidence-gain pairs come from trained documents in every mix. 5M-to-5M comparisons and all Q&A/probe/speed results stand; 50M fluency and all evidence-gain figures optimistic. Built clean_eval.py (data_core/eval_clean), core_eval --evidence-file, clean rescoring chain, and a leak filter in prepare_core_mix.py. 100M run continues.
- **11:05:** clean evaluation set verified (0 leaked paragraphs against P50 and P100); clean evidence gain is 8.7-9.2% at 5M, not 33.5%. Stale-cache bug in encode_file fixed (rebuild when the text file is newer); 5M baseline clean card rerunning.
