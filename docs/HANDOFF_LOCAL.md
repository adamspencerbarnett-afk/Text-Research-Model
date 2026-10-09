# Handoff: continue the encoder research on a local machine with a GPU

Paste everything below into the local AI assistant.

---

You are continuing a research project. Read before acting, then work in small, measured steps.

## 1. Get the code

```bash
git clone https://github.com/adamspencerbarnett-afk/Text-Research-Model.git
cd Text-Research-Model
git checkout claude/gracious-hypatia-hk74hn
```

All current work is on that branch; `main` is out of date. Do not push to `main`. Create a new branch for your work (for example `local-gpu-phase2`) and commit there.

## 2. Read these first, in this order

1. `docs/VERSION_1.md` — what was built, why, and what was learned (short).
2. `docs/DESIGN_V2.md` and `docs/figures/design_v2_flow.svg` — the current architecture and why.
3. `docs/ENCODER_RESEARCH_PLAN.md` — the full research record. Section 3a and the dated log (section 8) hold every result with its cause. The last entries are the newest.
4. `compiler/README.md` — how the compiler and experiment scripts are used.

## 3. The project in one paragraph

The goal (owner: Adam Barnett) is to test whether a model trained on text compiled by a small deterministic encoder can be lighter, faster and smarter than a conventional model on the same text, with far more capacity held in cheap lookup structures, and with the same compiler translating in both directions at conversation time. Findings so far: the compiler alone matches a standard BPE tokenizer (a lossless encoder cannot add information by itself); exact n-gram count tables mixed into the output by a learned trust head, with leave-one-out counting in training, make the model 9 to 11% better than standard BPE at equal compute, and the gain survived 4.4x more data; the tables learn by reading (adding a book to the counts helps without retraining); a question-keyed exchange memory answers 99% of stored questions exactly, including ones added after training; but the 0.79M-parameter network cannot yet carry retrieved memory into an answer by itself, and the local tables overpower the question. Everything so far ran on a 4-core CPU for minutes per run.

## 4. Set up the environment

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install numpy torch matplotlib                      # install the CUDA build of torch for your GPU
g++ -O2 -std=c++17 -o data/cv0 compiler/native/cv0.cpp  # Windows: cl /O2 /std:c++17 /EHsc compiler\native\cv0.cpp
python -B -m unittest compiler/test_compiler_v0.py compiler/experiments/test_ladder.py
```

All tests must pass before any experiment.

## 5. Rebuild the data (corpora are not in git; manifests with SHA-256 are in `results/`)

```bash
python compiler/experiments/prepare_books.py data                       # 45 MB corpus
python compiler/experiments/prepare_corpus_v1.py data data_v1 --target-mb 200 \
    --repo-list <GITenberg_repo_list.tsv> --curated <optional list>     # the list ships in the gitberg repo: gitenberg-dev/gitberg, gitenberg/data/
# Q&A data (Alpaca, non-commercial research use):
curl -o alpaca_data.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
python -I compiler/experiments/prepare_qa.py alpaca_data.json data_qa
```

Corpus v2 also used two files Adam supplied (a 100-book combined file and an enwik 50 MiB prefix); ask him for them and run `compiler/experiments/prepare_corpus_v2.py`.

## 6. First code change: run on the GPU

The harness (`compiler/experiments/train_lm.py`, `m2_qa.py`) is CPU-only. Add a `--device` option, move the model and every batch tensor to it, and keep the n-gram table lookups (numpy, on CPU) producing tensors that are then moved to the device. Verify by re-running one known screen and matching its CPU result within noise (for example `ladder1.py data data/work --native data/cv0 --budget 240 --configs v0_8k_ng64`), before trusting any new number.

## 7. What to do next, in order

1. **The scale run (Phase 2.1).** Answer the owner's question in his terms: smallest and fastest model reaching a quality target, traditional against ours. Cores of about 5M, 20M and 50M parameters (widths 256 to 512, 6 to 8 layers), each trained as (a) standard BPE 8k alone (`bpe_8k`) and (b) compiler + exact tables with 64 followers (`v0_8k_ng64`), on corpus v2 books plus the Wikipedia split, with `--patience` plateau stopping and the validation split. Report held-out bits per byte (books and Wikipedia), reply bytes per second, the by-class split, learned parameters vs table size. Key question: does the table gain narrow or hold as the core grows?
2. **Design v2 adjustments on the Q&A data** (`compiler/experiments/m2_qa.py`; last core `models/m2_qa_run3.pt`): a learned copy head over the retrieved exchange instead of the constant `--copy-bias`; local tables conditioned on the retrieved exchange; better question keys (inflection folding); then the deterministic check loop (verify, refine, restart, or say unknown) and the number channel. Measure answer F1 on seen, paraphrase, unseen and added-after-training questions, as the script already does.
3. **Learning by reading at scale** (`compiler/experiments/learn_by_reading.py`): add several of an author's books, and re-tune only the trust head after an addition.

## 8. Rules that kept the results honest (keep them)

- Fit dictionaries, BPE merges and count tables on the training split only. Held-out books are scored once per run.
- Count tables built on the training text must use leave-one-out during training (already implemented); without it the model stalls and fails off-domain.
- Compare at equal compute and at equal text; report both. Use the same loss code for every model.
- Screen ideas cheaply first, confirm the best with a second seed, and only then scale. Seed noise was 0.2 to 0.6%.
- Every run writes a JSON result in `results/`; every finding goes into `docs/ENCODER_RESEARCH_PLAN.md` (results table, cause, next step) and a dated line in its log.
- Report failures plainly with the cause. Do not claim intelligence from bits per byte; use answer accuracy for that.
- Commit and push after each completed step.
