# Text Research Model

*Research concept and direction by Adam Barnett. Updated 12 October 2026.*

A small language model that keeps its knowledge outside its weights. Instead of one large
network that must memorise everything, this project pairs a small network (5 to 50 million
parameters, trained on one consumer GPU) with a memory of past conversations it can read from
and copy out of, a calculator it can call, and a deterministic text compiler that turns text
into codes and back without loss.

## What the research is trying to accomplish

Modern language models get better mainly by getting bigger, which makes them expensive to
train, slow to run and impossible to update without retraining. This project asks whether a
different split does better at small scale:

- **Keep facts in a memory the model can look up**, so new facts are learned instantly, with
  no training, by adding one question-and-answer pair.
- **Let the model copy rather than generate** when the right words are already in front of it.
- **Give it exact tools** (a calculator, a lossless compiler) for the parts a network does badly.
- **Measure honestly.** Every claim is an answer-accuracy number on held-out questions, at equal
  training steps and equal text, repeated with a second seed before anything is built on it.

The goal is a model that is lighter, faster and more trustworthy than a conventional model of
the same size, and that can keep learning after training.

## How it works, in one picture

```text
  your question
       |
       v
  compiler: text -> codes (lossless, 8,192-word dictionary)
       |
       v
  memory: find the most similar stored question and its answer  --+
       |                                                           |
       v                                                           |
  small network reads [stored exchange + your question]           |
       |                                                           |
       +--> at each word: generate a new word, OR copy one  <------+
       |    from the stored answer (learned "copy head")
       |
       +--> arithmetic steps are checked by a calculator
       |
       v
  compiler: codes -> text
```

[`MODEL_EXPLAINED.html`](MODEL_EXPLAINED.html) walks through this step by step with real
examples. Open it in a browser.

## Key results

All numbers come from held-out test questions the model never trained on, unless marked
"stored". Full settings and every run are in [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md).

### 1. Memory plus copying: answer accuracy 13% → 89%

| model (5M parameters, Alpaca instructions) | stored questions, F1 | exact answers |
| --- | ---: | ---: |
| network alone | 13% | 0% |
| + memory lookup + copy head | 89% | 52% |

F1 is the overlap between the model's words and the reference answer (100% is word-perfect).
A network far too small to memorise facts answers accurately once answering is split into
"find the right memory" and "copy from it". Facts added after training with no retraining
score 77% F1, and the model stops cleanly at the end of an answer.

### 2. Bigger helps reading, not arithmetic

| task | 5M model | 50M model |
| --- | ---: | ---: |
| SQuAD reading comprehension (exact match) | 1% | 28% |
| GSM8K grade-school maths (correct final number) | 2.5% | 1.5% |

Finding an answer inside a passage you are shown improves ten times over with size.
Choosing the steps of a word problem does not. The calculator confirms the arithmetic is
right; the model picks the wrong operations. Verified self-training (keep only the model's own
checked solutions, retrain) raised the 5M score to 4.0%. The next lever is retrieving a
solution template from memory, which is what the architecture is built for.

### 3. Better data beat more data

At the same size and training steps, cleaning the training mix (unwrapped books, cleaned
Wikipedia, real conversations, worked maths) cut prediction error on conversation text by 36%
(2.02 → 1.30 bits per byte) and on maths by 59% (2.23 → 0.91). Bits per byte is how surprised
the model is by unseen text; lower is better.

### 4. A modern Transformer block is 1% to 7% better at equal cost

Swapping the network's block for the design used by current models (rotary positions, RMS
normalisation, gated feed-forward) lowered error on every held-out source, with 2% fewer
parameters: Wikipedia -1.3%, web -1.3%, maths -2.9%, chat -1.5%, question pairs -7.2%.

### 5. What did not work, and why that matters

- **A cleverer text encoding is not intelligence.** The lossless compiler matched a standard
  tokenizer to within 1%. Kept because it is exact and makes copying simple, not because it
  makes the model smarter.
- **Hand-built word statistics reverse with size.** Exact n-gram count tables made a 5M model
  9 to 11% better and a 20M model 1% worse. Dropped for larger cores.
- **Depth alone does not help yet.** A 3-layer wide core matches a 6-layer core on language at
  equal steps (within 0.3%) and generates 30% faster. The remaining depth tests are running.
- **Speed is not the network.** A 50M model and a 5M model answer 32 questions in 18.3 s and
  16.7 s. The per-step Python loop, not the Transformer, sets the time, so that loop is the
  engineering target before any speed claim.

### 6. Chat quality, graded by hand

A 32-question battery (stored facts, rewordings, general knowledge, facts taught during the
chat, arithmetic, small talk) is run on every model. Best so far: 24 of 32 (75%) at 50M with a
memory gate that lets the model decline a poor match, against 11 of 32 (34%) for the network
alone. Transcripts and grades are in [`docs/CHAT_NOTES.md`](docs/CHAT_NOTES.md).

## Try it

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_model.py
```

`run_model.py` shows a menu: train on a folder of text files, or chat with a saved model
(`models/qa_best_5M_512.pt` is the best small model in git). Settings are at the top of the
script. Full setup, including corpus rebuilds and GPU use, is in
[`docs/HANDOFF_LOCAL.md`](docs/HANDOFF_LOCAL.md).

## Read more

| document | what it is for |
| --- | --- |
| [`MODEL_EXPLAINED.html`](MODEL_EXPLAINED.html) | how the model works, with real codes, lookups and answers |
| [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md) | every run's settings and results, the queue, and the findings per session |
| [`docs/ENCODER_RESEARCH_PLAN.md`](docs/ENCODER_RESEARCH_PLAN.md) | the dated research log with the cause behind each result |
| [`docs/REASONING_PLAN.md`](docs/REASONING_PLAN.md), [`docs/PLAN_50M.md`](docs/PLAN_50M.md) | the maths and scale plans |
| [`docs/DATASETS.md`](docs/DATASETS.md), [`docs/CORE_TRAINING_PLAN.md`](docs/CORE_TRAINING_PLAN.md) | what the model reads and how the mix was built |
| [`BRAIN_DESIGN.html`](BRAIN_DESIGN.html) | the follow-on design: a brain-inspired model with full reconstruction spec and history |
| [`Traditional_Hybrid.html`](Traditional_Hybrid.html) | the best ideas here applied to a standard Transformer |
| [`COMPILER_AND_MODERN_METHODS.html`](COMPILER_AND_MODERN_METHODS.html) | why a compressed encoding cannot add intelligence; review of modern methods; ranked tests |
| [`docs/VERSION_1.md`](docs/VERSION_1.md), [`docs/DESIGN_V2.md`](docs/DESIGN_V2.md) | the frozen first version and the design that replaced it |
| [`archive/`](archive/README.md) | the original September research (semantic parser and executor) and the frozen Version 1 |

## Repository layout

```text
.
├── run_model.py        # menu: train on a folder or chat with a model
├── compiler/           # the text compiler, native encoder, tests, experiments/ (all training scripts)
├── docs/               # plans, ledger, research log, chat transcripts
├── results/            # one JSON per run; corpus manifests with SHA-256
├── models/             # small saved models (large 50M checkpoints stay local in runs/)
├── archive/            # earlier research, kept whole
├── *.html              # design documents, open in a browser
└── requirements.txt
```

Training corpora are not in git. `results/corpus_v*_manifest.json` records every file's size
and hash, and `docs/HANDOFF_LOCAL.md` has the rebuild commands.

## Research standard

Results are labelled by what was measured. Dictionaries and tables are fitted on the training
split only. Comparisons are made at equal steps and equal text. Ideas are screened cheaply,
confirmed with a second seed, then scaled. Every run writes a JSON result and every finding
goes into the log with its cause. Failures are reported plainly. Bits per byte is never
claimed as intelligence; answer accuracy is.
