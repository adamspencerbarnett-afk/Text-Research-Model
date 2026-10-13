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
checked solutions, retrain) raised the 5M score to 4.0%. Neither depth (3 to 16 layers) nor a
modern block moved it. Maths is now paused and will get its own research line, starting from
the September typed-program work: retrieve a program, fill its slots, execute it exactly.

### 3. Better data beat more data

At the same size and training steps, cleaning the training mix (unwrapped books, cleaned
Wikipedia, real conversations, worked maths) cut prediction error on conversation text by 36%
(2.02 → 1.30 bits per byte) and on maths by 59% (2.23 → 0.91). Bits per byte is how surprised
the model is by unseen text; lower is better.

### 4. Answers are now 4 to 7 times faster, with identical replies

| model | before, ms per word piece | after | speedup |
| --- | ---: | ---: | ---: |
| 5M | 11.7 | 1.75 | 6.7x |
| 50M | 14.6 | 3.6 | 4.1x |

On an idle GPU a full answer now takes 0.08 s and 10 joules at 5M (from 0.45 s and 29 J), and
0.15 s and 24 joules at 50M, with the same chat-probe score.

At these sizes the GPU was spending its time launching about 300 tiny jobs per generated word
piece, not computing. The answer step is now recorded once as a CUDA graph and replayed with
a single launch. Replies are byte-identical to the old loop on every test question. Low-bit
kernels (int8, 4-bit) were tried and were slower, for the same reason.

### 5. Precision: bf16 training is free, and so are 8-bit weights

| test | result |
| --- | --- |
| training in bf16 vs fp32, same steps | error 1.1915 vs 1.1901 (0.1% apart); bf16 2.2x faster |
| answering with 8-bit weights | within noise of full precision at 5M and 50M |
| answering with 4-bit weights | 13% worse on stored answers at 5M; 8% worse on new questions at 50M |

From here on every model trains in bf16 and is checked at 8 bits. 4-bit is not used.

### 6. Shape and block: shallow and wide is enough

At equal parameters and steps, 3 layers matched 6 on language and generated 30% faster; 11
and 16 layers were worse and slower. A modern Transformer block (rotary positions, RMS
normalisation, gated feed-forward) lowered error on every text source by 1% to 7%, but at 5M
it did not improve answers, so it is kept for the larger cores, where the network does more of
the work.

### 7. Retrieval: finding reworded questions

A small pretrained sentence encoder used beside the word-based key raised the rate at which a
reworded question finds its stored twin from 75.1% to 80.9%. It now finds "What GPU does the
research computer use?" when asked about the graphics card. Retrieved answers are 11–13% closer
to the right answer on reworded and new questions. Trained with this retrieval in the loop, the
5M model reached its best score yet on new questions (0.299 F1) and 22 of 32 on the chat probe,
with no loss on stored answers.

### 8. What did not work, and why that matters

- **A cleverer text encoding is not intelligence.** The lossless compiler matched a standard
  tokenizer to within 1%. Kept because it is exact and makes copying simple, not because it
  makes the model smarter.
- **Hand-built word statistics reverse with size.** Exact n-gram count tables made a 5M model
  9 to 11% better and a 20M model 1% worse. Dropped for larger cores.
- **Depth does not buy intelligence at this size.** Past six layers it cost language quality
  and speed, and it never moved maths.
- **The bottleneck was never the network.** A 50M and a 5M model answered at nearly the same
  speed, which is what pointed to launch overhead and the graph fix above.

### 9. Chat quality, graded by hand

A 32-question battery (stored facts, rewordings, general knowledge, facts taught during the
chat, arithmetic, small talk) is run on every model. Best so far: 24 of 32 (75%) at 50M with a
memory gate that lets the model decline a poor match, against 11 of 32 (34%) for the network
alone. Transcripts and grades are in [`docs/CHAT_NOTES.md`](docs/CHAT_NOTES.md).

## Where it is going

[`HYBRID_DESIGN.html`](HYBRID_DESIGN.html) designs the next system: a chatbot, not a lookup
tool. It keeps what the research proved (exact memory, retrieval-trained copying, the graph
decoder, bf16 with 8-bit weights) and adds what modern retrieval systems do well: passages as
well as Q&A pairs, combined word and meaning search with reranking, training on misleading
evidence so the model learns when not to copy, short thinking steps, searches the model can ask
for mid-answer, exact typed programs for counting, dates and conversation state, and a check
that says "I don't know" when nothing supports an answer. The generator grows to 100–300M
parameters, the size where reading is likely to become good, while knowledge stays in memory.

Latest training-recipe result: a warmup-stable-decay schedule lowers text error 0.8% at no cost,
and with the Muon optimiser 0.9%; at 5M neither changed answer scores. Running now (see the
queue in [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md)): training on misleading evidence so the
model learns when not to copy, and a second seed of the best training recipe.

## Try it

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_model.py
```

`run_model.py` shows a menu: train on a folder of text files, or chat with a saved model
(`models/qa_best_5M_512.pt` is the best small model in git). Chat commands include `/add
question || answer` to teach a fact and `/load file.txt` to teach many at once. Settings are at
the top of the script. Full setup, including corpus rebuilds and GPU use, is in
[`docs/HANDOFF_LOCAL.md`](docs/HANDOFF_LOCAL.md).

## Read more

| document | what it is for |
| --- | --- |
| [`MODEL_EXPLAINED.html`](MODEL_EXPLAINED.html) | how the model works, with real codes, lookups and answers |
| [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md) | every run's settings and results, the queue, and the findings per session |
| [`docs/ENCODER_RESEARCH_PLAN.md`](docs/ENCODER_RESEARCH_PLAN.md) | the dated research log with the cause behind each result |
| [`docs/REASONING_PLAN.md`](docs/REASONING_PLAN.md), [`docs/PLAN_50M.md`](docs/PLAN_50M.md) | the maths and scale plans |
| [`docs/DATASETS.md`](docs/DATASETS.md), [`docs/CORE_TRAINING_PLAN.md`](docs/CORE_TRAINING_PLAN.md) | what the model reads and how the mix was built |
| [`HYBRID_DESIGN.html`](HYBRID_DESIGN.html) | the next system: a hybrid memory model that keeps what the research proved, adds the modern retrieval stack, reasons in steps, and a research plan with UML and hardware diagrams |
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
