# LOG

Short session entries: date, task, files touched, decisions, blockers, next action. Full results are in `docs/RUN_LEDGER.md`.

## 2026-10-12

- **Task:** sessions 16–19 (modern block, depth sweep, modern-core Q&A, precision); speed and retrieval engineering; hybrid design; repository published to `main`.
- **Results:** depth 3 = 6 > 11 > 16 layers; modern block −1 to −7% text error, answers level at 5M; bf16 training = fp32 (0.1%) at 2.2x speed; int8 weights free at 5M and 50M; int4 −13% stored F1 at 5M; CUDA-graph answer step 3.6–9.3x faster with identical replies; low-bit kernels slower at these sizes; dense key reworded recall 75.1% → 80.9%.
- **Files:** `fast_decode.py`, `precision.py`, `dense_keys.py`, `cost.py` (new); `m2_qa.py` (graph path, kernels, dense keys, `--amp`), `train_lm.py` / `scale_run.py` (Muon, WSD, fused), `chat_probe.py`, `core_eval.py`, `run_model.py` (`/load`, graph chat), chains 18–21, `HYBRID_DESIGN.html`, `README.md`, `MASTER.md`, `LOG.md`.
- **Decisions (Adam):** math paused; bf16 + int8 only; focus on text, Q&A, conversation; build the hybrid design.
- **Blockers:** none.
- **Next:** session 20 (Muon / WSD recipe, running), session 21 (dense retrieval in training), then session 22 (core reranking, bad-evidence training, copy-as-draft decoding).
