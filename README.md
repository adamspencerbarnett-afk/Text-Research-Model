# Text Research Model

*Research concept and direction by Adam Barnett. Updated 9 October 2026.*

The question this project tests: can a model that keeps most of its knowledge in cheap,
deterministic lookup structures around a small learned core be lighter, faster and more
accurate than a conventional model trained on the same text, and learn new facts without
retraining? The same deterministic compiler turns text into codes for training and turns the
user's text into codes, and the model's codes back into text, at conversation time.

The work has two tracks. Track 2 is where the research is now.

## Track 2 (current): the deterministic compiler, lookup tables and memory

Read in this order:

1. [`docs/VERSION_1.md`](docs/VERSION_1.md): what was built and learned, frozen 9 October.
2. [`docs/DESIGN_V2.md`](docs/DESIGN_V2.md): why encoding alone could not add knowledge, and
   the architecture that follows (compiler as key generator, count tables, a question-keyed
   exchange memory, a small core, a trust head, a deterministic check loop).
3. [`docs/ENCODER_RESEARCH_PLAN.md`](docs/ENCODER_RESEARCH_PLAN.md): every result with its
   cause, in section 3a and the dated log in section 8.
4. [`docs/HANDOFF_LOCAL.md`](docs/HANDOFF_LOCAL.md): how to set up a machine, rebuild the
   corpora and continue.
5. [`compiler/README.md`](compiler/README.md): the compiler and the experiment scripts.

What is established so far (all on public-domain books, Wikipedia text and the Alpaca Q&A set;
numbers are learning-efficiency results, not intelligence results):

- The compiler alone matches a standard BPE tokenizer: a lossless encoder cannot add knowledge.
- Exact n-gram count tables, mixed into the output by a learned trust head with leave-one-out
  counting, make the model 9 to 11% better than standard BPE at equal compute. The gain held
  with 4.4x more data. Adding a book to the tables helps without retraining.
- A memory keyed on the question answers 99% of stored questions exactly, including questions
  added after training. A learned copy head is under test as the way the network carries the
  retrieved answer into its own reply (first screens, 9 October).
- Open: whether the table gain holds as the core grows (the scale run, 5M to 50M parameters on
  a GPU); the check loop and the number channel, which reconnect this track to Track 1's
  verifiable reasoning.

Run it yourself: `python run_model.py` (menu; trains on a folder of text files or chats with a
saved model; settings at the top of the script). Setup is in `docs/HANDOFF_LOCAL.md`.

## Track 1 (frozen, 15 September): Compact Semantic Reasoning

**V1 Experimental** translates controlled natural-language problems into typed events, then
answers them with either a small learned reasoning core or an exact executor. It asked whether
explicit semantic structure improves accuracy and reasoning reliability when model size and
compute are limited. Within its narrow domain (synthetic accounting and ordering tasks) it is
very accurate, with 108 automated tests; it is not a general language model or a chatbot. Its
lasting lessons, which Track 2 inherits: structure makes a small specialist precise; the
language boundary is the main risk; answer accuracy can hide structural failure; fair controls
start from the same raw text.

- [`MASTER_PLAN.md`](MASTER_PLAN.md) and [`MASTER_LOG.md`](MASTER_LOG.md): the V1 plan, status
  and acceptance standard.
- [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md): the chronology from V6 to V1,
  including the original question and the negative results.
- [`docs/CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md`](docs/CONTINUOUS_IMPROVEMENT_V1_RESEARCH.md):
  the proposed continuous-improvement design (memory first, weights later), which Track 2's
  exchange memory now implements in its first form.
- [`README.html`](README.html): the interactive V1 overview (historical).

V1 quick start (CPU is enough):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts\semantic_model.py --text "Alice starts with 8 coins. Bob starts with 3 coins. Alice gives Bob 3 coins. How many coins does Alice have now?" --mode executor
.\.venv\Scripts\python.exe -B -m unittest discover -s scripts -p "test_semantic_v1.py" -v
```

## Repository layout

```text
.
├── run_model.py                 # train on a folder / chat with a model (menu, popups, settings at the top)
├── compiler/                    # Track 2: compiler v0, native encoder, tests, experiments/
├── docs/                        # VERSION_1, DESIGN_V2, ENCODER_RESEARCH_PLAN, HANDOFF_LOCAL, RESEARCH_HISTORY, ...
├── results/                     # one JSON per run; corpus manifests with SHA-256
├── models/                      # saved cores (Track 2) and the V1 checkpoints
├── releases/version1.zip        # Track 2 Version 1, frozen
├── scripts/                     # Track 1: semantic_model.py, semantic_tasks.py, semantic_experiment.py, tests
├── MASTER_PLAN.md, MASTER_LOG.md
└── requirements.txt
```

Corpora are not in git; `results/corpus_v*_manifest.json` records every file's size and hash,
and `docs/HANDOFF_LOCAL.md` has the rebuild commands.

## Research standard

Results are labelled by what was measured. Dictionaries, merges and tables are fitted on the
training split only; comparisons are made at equal compute and at equal text with the same
loss code; ideas are screened cheaply, confirmed with a second seed, then scaled; every run
writes a JSON result and every finding goes into the plan with its cause; failures are reported
plainly; bits per byte is never claimed as intelligence, answer accuracy is.
