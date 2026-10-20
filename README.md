# Text Research Model

*Research concept and direction by Adam Barnett. Updated 19 October 2026.*

A small chat model that keeps its knowledge outside its weights. A 5M-parameter network, trained on one consumer GPU,
works with two memories: a **conversation memory** that keeps every turn of a chat (so it can recall something said
10,000 turns ago), and a **permanent store** of 311,010 real facts it can look up. A search engine picks what goes into
the model's 2,048-code window, and the network learns to read and copy from it instead of memorising facts. A lossless
text compiler turns text into codes and back.

The question behind the research: at small sizes, does splitting "know the facts" (memory) from "use the facts" (a
small network) give a model that is lighter, faster, easier to update and more honest than a conventional model of the
same size?

## Start here (about 15 minutes)

You need Python 3.12, Git with **Git LFS** (`git lfs version`; it ships with Git for Windows), and ideally an NVIDIA
GPU. Chatting also works on the CPU, only slower.

```powershell
git lfs install
git clone https://github.com/adamspencerbarnett-afk/Text-Research-Model.git
cd Text-Research-Model
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_model.py          # menu: choose 3, "Chat with the best chat model"
```

On Linux or macOS use `.venv/bin/python`. For CPU only, install torch with `pip install torch`.

The clone brings the best model through Git LFS (about 235 MB): `runs/chat22/mix_5m_s2.pt` and its n-gram tables,
the core it was trained from, and the permanent fact store. If those files are only a few hundred bytes, LFS did not
download them: run `git lfs pull`.

The first chat downloads the MiniLM sentence encoder (about 90 MB, from Hugging Face) and builds the fact store's
meaning vectors once (`data_real/shared/knowledge/knowledge_real_minilm.npy`, 0.5 GB, a few minutes on a GPU). After
that the model loads in under a minute (311,010 facts). To skip the menu: `python run_model.py --chat runs/chat22/mix_5m_s2.pt`.

### Things to try in the chat

```text
you> My dog's name is Rex.
you> What is my dog's name?                       -> Your dog Rex.
you> My favorite color is green.
you> Actually, my favorite color is blue now.
you> What is my favorite color?                   -> Your favorite color is blue now.
you> What is the capital of Australia?            -> The capital of Australia is Canberra   (permanent store)
you> What is photosynthesis?                      -> Photosynthesis is the biological process by which plants ...
you> What is 12 plus 30?                          -> 12 plus 30 is 42.   (calculator tool)
you> What is my brother's name?                   -> I don't have that in my memory. Could you tell me?
you> /forget
you> /add When is the team meeting? || The team meeting is on Friday at 3 pm.
you> When is the team meeting?                    -> The team meeting is on Friday at 3 pm.
```

These replies come from the shipped model, typed in this order as one conversation (checked from a fresh clone on
19 October). You will also see its weak spots: some ways of telling it a fact are refused ("My wifi password is
maple42", "My name is Sam"), some general questions miss ("Who wrote Pride and Prejudice?"), stories wander after a
sentence or two, and a refusal tends to make the next answer refuse too (asked right after "What is my brother's
name?", even "What is 12 plus 30?" is refused; `/forget` starts fresh). These are the targets of the next rounds.

Under each reply a short line shows where the answer came from (`memory: stored fact ...`, `memory: turn N ...`, a
tool, or "nothing found, the model answered on its own"), how full the window is, and the time. Commands: `/add`,
`/load file.txt`, `/forget`, `/memory off|on`, `/info off|short|full`, `/codes N`, `/quit`. Settings are at the top of
`run_model.py`.

### Check your setup against our numbers

```powershell
.\.venv\Scripts\python.exe -B -m chat22.tests.run_suite runs/chat22/mix_5m_s2.pt --tier full --engine-opts '{\"search\": \"v2\"}' --tag _check
```

This runs the full test suite (about 35 minutes on an RTX 3090) and writes `results/suite_mix_5m_s2_check.json`.
Compare it with `results/suite_mix_5m_s2_ev2full.json`, our run of the same model. The held-out test files the suite
reads are in the repository. (In bash, write the option as `--engine-opts '{"search": "v2"}'`.)

## The best model so far

**Round 5, "BEST v2"** (19 October 2026), recipe in [`recipes/BEST.json`](recipes/BEST.json). Seed 2 is the one
shipped: same recipe as seed 1, but fewer wrong refusals of facts the user told it.

Against the previous best (variant A), over two seeds each: 7 significant wins, 0 losses. Seed 2 on the full suite:

| what is tested | result |
| --- | --- |
| questions answered from memory (1,344): right / invented / copied the wrong fact | 88% / 6% / 0.7% |
| 30 facts told during a conversation, asked after 100 / 1,000 / 10,000 turns | 87% / 93% / 87% |
| questions about things never told: says it doesn't know | 90-100% |
| conversation tasks: recall, correction, choosing between facts, returning to a topic | 96% (115/120) |
| a fact inside 4,096 codes of text (twice the window): window alone / with memory | 31% / 94% |
| questions about a story it was told; contradictions per scene | 83%; 0.00 |
| first word of a reply; whole reply with 10,000 turns in memory (RTX 3090) | 20 ms; about 0.27 s |

Where it is still weak, and what the next rounds target: questions phrased the way people really talk (22% on real
phrasings, against 96% on generated ones), personal facts told in conversation (39%; 67% when stored with `/add`),
updating a fact (20%), asking a clarifying question (0%), and answering with a list. `recipes/BEST.json` lists every
open front with its plan.

## How it works

```text
 you type a message
       |
 compiler: text -> codes (lossless, 8,192-entry dictionary)
       |
 search engine v2: fact table + fast lane + decide
       |   looks in the conversation memory (every earlier turn)
       |   and the permanent store (311,010 real facts)
       v
 window (2,048 codes) = memory block (the chosen facts and turns) + recent turns + your message
       |
 5M network (6 layers, width 256), at each step:
       |   generate a word piece, or COPY one from the memory block (copy head),
       |   mixed with exact n-gram statistics where they are reliable (trust head)
       v
 compiler: codes -> text; the turn is added to the conversation memory
```

Rules the design keeps: the permanent store holds **real facts only**; invented names and stories belong in the
conversation memory and in training data that teaches the model to use what a person tells it. The model should say
it doesn't know when nothing in memory supports an answer.

## How the research improves the model

Everything that works is recorded in [`recipes/BEST.json`](recipes/BEST.json): BEST (the model to beat), BASE (the
recipe new rounds build on), the data every kind of training window reads, every adopted lesson and every rejected
change with its evidence, the open fronts and the history of rounds.

One round = **one change**, tested honestly:

```bash
bash compiler/experiments/chains/train_best.sh 1 myidea_s1 --parts longconv,longdocs,alpacaqa,NEWPART   # seed 1
#   trains the base recipe plus your change (about 3 hours on an RTX 3090), runs the full suite (35 minutes),
#   and compares with the best's two seeds: runs/compare_myidea_s1.log
bash compiler/experiments/chains/train_best.sh 2 myidea_s2 --parts ...                                  # only if it wins
python research_side/compare_two_seeds.py myidea_s1 myidea_s2       # verdict: NEW BEST needs 0 losses and at least 1 win
```

A change is adopted only when two seeds beat the best's two seeds front by front (2 standard errors), with no front
lost. Then it is written into `recipes/BEST.json`, and the old best stays in its history.

Standing rules: train in bf16 (8-bit weights are checked for answering, no fp32 or 4-bit runs); one training at a time
on a GPU, using at most 3/4 of it (`--gpu-frac`); training data that works is not changed without a stated reason;
nothing above 100M parameters without Adam's agreement; results are answer accuracy on held-out questions at equal
steps and equal data, never bits per byte alone. Every run writes `results/*.json`, and every round's numbers and
verdict go into [`LOG.md`](LOG.md).

**Training needs the data folders, which are not in git** (about 2.7 GB for the best recipe). The list, sizes and
builders are in [`docs/SETUP.md`](docs/SETUP.md). The quickest way is to get them from Adam.

## What to read, in order

| file | what it is for |
| --- | --- |
| [`MASTER.md`](MASTER.md) | the project in one page: goal, rules, best recipe, current scope, file map |
| [`recipes/BEST.json`](recipes/BEST.json) | the best model and recipe, lessons, rejected changes, open fronts |
| [`LOG.md`](LOG.md) | the working log: every round, its numbers and decisions (newest at the bottom) |
| [`docs/SETUP.md`](docs/SETUP.md) | environment, training data, training, tests, as commands |
| [`docs/RESEARCH_STORY.html`](docs/RESEARCH_STORY.html) | the story from September to now: 158 tests with settings, results and lessons (open in a browser) |
| [`docs/CHAT5M_V2_DESIGN.html`](docs/CHAT5M_V2_DESIGN.html) | design of the chat model: window layout, memories, training kinds, results |
| [`docs/V22_CLEAN_DESIGN.md`](docs/V22_CLEAN_DESIGN.md), [`docs/SEARCH_ENGINE_V2.md`](docs/SEARCH_ENGINE_V2.md) | the chat22 package and the search engine |
| [`docs/TRAINING_MAP.html`](docs/TRAINING_MAP.html) | interactive map from core data to models and results |
| [`docs/RUN_LEDGER.md`](docs/RUN_LEDGER.md) | every run's settings and results, regenerated from `results/` |
| [`MODEL_EXPLAINED.html`](MODEL_EXPLAINED.html) | how the memory-and-copy model works, with real examples |
| [`BRAIN_DESIGN.html`](BRAIN_DESIGN.html), [`HYBRID_DESIGN.html`](HYBRID_DESIGN.html), [`Traditional_Hybrid.html`](Traditional_Hybrid.html), [`COMPILER_AND_MODERN_METHODS.html`](COMPILER_AND_MODERN_METHODS.html) | Adam's designs and forward-looking ideas |
| [`docs/REASONING_MAP.html`](docs/REASONING_MAP.html), [`docs/REASONING_PLAN.md`](docs/REASONING_PLAN.md) | the reasoning and maths line (paused since 12 October) |

## Repository map

```text
run_model.py          chat with the best model (menu option 3), train on a folder, device settings
chat22/               the chat model package (current)
  engine.py           answering: memories, search, window, streaming reply
  memory.py, search_v2.py, window.py, tools.py
  model.py, blocks.py, tables.py, encoder.py
  train.py, windows_train.py    training (every kind of window, --parts switches, held-out plateau stop)
  ground.py, judge.py           tested and rejected for now (grounding stage, learned judge); kept for later rounds
  data/               data builders (real datasets, fact store, any-length sessions, story worlds)
  tests/run_suite.py  the full test suite; gate22.py, scoring.py, test_engine.py, test_equivalence.py
recipes/BEST.json     the cumulative best recipe
research_side/        side studies and the two-seed comparison (compare_two_seeds.py)
compiler/             the lossless text compiler (compiler_v0.py, native/cv0.cpp) and its tests
  experiments/        core training (train_lm.py, scale_run.py), data preparation, dense keys, probes, earlier chat engine
  experiments/chains/ every queued session of the current line (42-57) and train_best.sh
results/              one JSON per run and per test
docs/                 designs, plans, ledger, research story
data_real/            README, plans and recipe of the real-data folders; the permanent fact store (LFS)
runs/                 only the best model and its core are in git (LFS)
data_chat22/, data_core/, data_mt/, data_skills/   only the held-out test files the suite reads
```

Not in the repository: the training data, other checkpoints, downloads, and `archive/` (September to mid-October
research, Version 1, the paused maths line), which stay on the research machine.

## Earlier findings that still hold

| finding | evidence |
| --- | --- |
| Memory plus copying beats a bigger network for facts | 5M network alone 13% F1 on stored questions; with memory lookup and a copy head 89% |
| Training the model with the search in the loop, and showing the right answer under reworded questions, matters most | reworded questions 5/5 and 4/5 over two seeds, against 1-3/5 before |
| Cleaner data beats more data | same steps: conversation-text error 2.02 -> 1.30 bits per byte |
| Answering speed is launch overhead, not compute, at these sizes | one CUDA graph per step: 4-7x faster, byte-identical replies |
| bf16 training and 8-bit weights cost nothing; 4-bit does | bf16 within 0.1% and 2.2x faster; 4-bit 8-13% worse |
| Shallow and wide is enough at 5M | 3 layers matched 6; 11 and 16 were worse and slower |
| Bigger helps reading, not arithmetic | SQuAD 1% -> 28% from 5M to 50M; GSM8K stayed near 2% (maths paused) |

Details for each are in `docs/RUN_LEDGER.md` and `docs/RESEARCH_STORY.html`.

## Working together

Work on your own branch and open a pull request into `main`. Keep the habits that make results comparable: one change
per round, the full suite with `--engine-opts '{"search": "v2"}'`, two seeds before claiming a win, a results JSON for
every run and a short entry in `LOG.md` (date, what changed, numbers, verdict). Model files go into git only for a new
best, through LFS (`git add -f runs/chat22/NAME.pt runs/chat22/NAME.tables.npz`), so the repository stays small.
