# Text Research Model

*Research concept and direction by Adam Barnett. Updated 20 October 2026: the current model is **C2b**.*

A 5M-parameter chat model that keeps its knowledge outside its weights. It talks with you, remembers everything said in
the conversation (any length), looks up real facts in a permanent store of 311,010 facts, answers several questions in
one message, and says when it doesn't know. It trains in about 2 hours on one RTX 3090 and answers in about a quarter of a
second. At this size it is meant for simple chat (a game character, a companion) and for finding facts quickly.

The research question: at small sizes, does splitting "know the facts" (memory) from "use the facts" (a small network
trained to read and copy from memory) give a model that is lighter, faster, easier to update and more honest than a
conventional model of the same size? The answer so far is yes for memory and facts; the network's size limits how well
it understands new wordings.

## Start here (about 15 minutes)

You need Python 3.12 and Git with **Git LFS** (`git lfs version`; included in Git for Windows). An NVIDIA GPU is
recommended; chatting also works on the CPU.

```powershell
git lfs install
git clone https://github.com/adamspencerbarnett-afk/Text-Research-Model.git
cd Text-Research-Model
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe run_model.py          # menu: 3 = chat with the best model (C2b)
```

On Linux or macOS use `.venv/bin/python`; for CPU only, `pip install torch`. The clone brings the model through Git LFS
(about 235 MB: C2b, its n-gram tables, the 5M core and the fact store). If those files are a few hundred bytes, run
`git lfs pull`. The training datasets are **not** downloaded by a normal clone (see [Reproduce C2b](#reproduce-c2b)).

The first chat downloads the MiniLM sentence encoder (~90 MB) and builds the fact store's meaning vectors once
(`data_real/shared/knowledge/knowledge_real_minilm.npy`, 0.5 GB; a few minutes on a GPU). After that it loads in under a
minute. Direct start: `python run_model.py --chat runs/chat22/combo2b_5m_s1.pt`.

### What it does: real C2b replies (fresh conversations)

```text
you> My sister Maya works as a nurse and she lives in Denver.
bot> Thanks, noted.
you> Where does my sister live and what does she do?
bot> Your sister Maya works as a nurse and she lives in Denver.
you> Actually, my sister moved to Portland.
you> Where does my sister live now?
bot> Your sister moved to Portland.

you> My dog's name is Rex.
you> I have two questions. What is my dog's name? What is my uncle's job?
bot> Your dog's name is Rex. But I don't have your uncle's job in my memory.

you> Who has won the most Wimbledon men's singles titles? Also, what is the name of the biggest city in Nigeria?
bot> To your first question: Roger Federer. To your second: Lagos.

you> My coach sent this: "Practice is on Monday at 5 pm. The locker code is 2291." When is practice and what is the locker code?
bot> To your first question: Monday at 5 pm. The locker code is 2291.

you> What is photosynthesis?
bot> Photosynthesis is the biological process by which plants use carbon dioxide, sunlight and water to create oxygen and energy.
you> What is 12 plus 30?
bot> 12 plus 30 is 42.
```

Under each reply a line shows where the answer came from (`memory: stored fact ...`, `memory: turn N ...`, a tool, or
"nothing found"), how full the window is and the time. Commands: `/add question || answer` (store a fact), `/load
file.txt`, `/forget` (new conversation), `/memory off|on`, `/info off|short|full`, `/codes N`, `/quit`.

**Weak spots you will meet:** a never-told question about a common thing ("What is my cat's name?") can pull general
knowledge instead of "I don't know"; counting or summarizing what you said is not in C2b yet (learned in tests, see
below); a single message can get an odd acknowledgement; stories wander after a few sentences; real human phrasing is
understood far less often (33%) than the casual wordings it trained on (90%).

## The current model: C2b, and how it compares

C2b = the best earlier recipe (round 5) + training on messages with several parts, answered with two search options
(below). Scores on the full test suite (held-out questions, wordings and names the training never saw); round 5 is the
previous best (two seeds pooled), C2b is one seed.

| ability | round 5 | **C2b** |
| --- | ---: | ---: |
| multi-part messages, every part right | 27% | **77%** |
| — two questions at once / one known + one unknown / two stored facts / a quoted note | 0 / 0 / 0 / 0% | 80 / 70 / 65 / 60% |
| casual wording of told facts | 80% | **90%** |
| conversation tasks (recall, correction, choosing, returning to a topic) | 92% | **98%** |
| telling similar facts apart (near-miss) | 48% | **70%** |
| questions about a story told in the chat | 76% | **87%** |
| answers right when memory is involved (1,344 questions); invented | 88% · 5.8% | 89% · 6.1% |
| facts told 100 / 1,000 / 10,000 turns earlier | 85 / 93 / 87% | 87 / 90 / 90% |
| "I don't know" when never told | 90-100% | 90-100% |
| corrections · misleading facts · two-step facts (from memory) | 93 · 92 · 99% | 95 · 91 · 100% |
| a fact inside 4,096 codes of text (twice the window), with memory | 93% | 94% |
| reading a long pasted document (memory / window) | 55 / 41% | 57 / 39% |
| personal facts told in chat · stored with /add | 39 · 67% | 47 · 69% |
| stored fact asked directly | 100% | 100% |
| story contradictions per scene | 0.00 | 0.08 |
| told fact + stored fact ("Who wrote my favorite song?") | 32% | 10% |
| first word of a reply | 20 ms | 20 ms |

C2b's two losses (story contradictions: 2 slips in 25 scenes; told fact + stored fact) are the next fixes. Adam adopted
C2b as the strongest model (20 Oct); `recipes/BEST.json` holds its recipe and evidence.

## How the chat works: one small network, two memories

```text
 your message
      |
 compiler: text -> codes (lossless, 8,192-entry dictionary)
      |
 search engine v2 (no training):
      |   question splitting: one search per question in the message
      |   fact table + fast lane: a personal-fact question goes straight to the newest stored value
      |   candidates + decide: word key + MiniLM meaning key, drop look-alikes and outdated values
      |   second hop: "my favorite song" -> the song named earlier -> the stored fact about that song
      |        |                                   |
      |   CONVERSATION MEMORY                 PERMANENT STORE
      |   every turn, kept for the whole      311,010 real facts (Wikipedia, science
      |   conversation (tested to 10,000      statements, real Q&A); /add adds yours;
      |   turns); statements become fact-     real facts only (invented material
      |   table records, corrections          stays in conversation memory)
      |   replace the old value
      v
 window (2,048 codes) = recent turns + memory block (the chosen facts and turns) + your message
      |
 5M network (6 layers, width 256): at each step generate a word piece or COPY one from the memory
 block (copy head), mixed with exact n-gram statistics where they are reliable (trust head)
      |
 reply streams out; the turn is added to the conversation memory
```

The window holds 2,048 codes (about 7,000 characters). Anything older lives in the conversation memory and comes back
through the search, so conversations of any length work; a very long single document is read only in part.

### How the two memories are trained

Training builds windows in exactly the layout the engine answers in (`chat22/window.py`), so what the model sees in
training is what it sees in use. One chat stage trains every kind together, from a pre-trained 5M core
(`runs/core_mt5m`), in shares that tests chose:

| kind | share | what it teaches | memory it trains |
| --- | ---: | --- | --- |
| recall | 12% | facts stated across 1-20 real turns, with distractors, corrections, two-step links, 25% never told; turns that scrolled out return as `[turn N]` items | conversation |
| skills | 13.5% | personal facts told in chat: near misses, updates, unknown, clarify, lists; **35% multi-part messages** (C2b) | conversation |
| any-length sessions | 11% | real dialogues of 10-10,000 turns with planted facts; the memory block is what the real search engine returned; "I don't have that" when it missed | conversation, at any length |
| Q&A | 17% | instruction Q&A with the fact (sometimes reworded) and look-alike neighbours as `[fact]` items; **15% two stored facts asked at once, with look-alike decoys** (C2b) | permanent store |
| buried evidence | 11% | the answer inside a long pasted document, with decoys | reading |
| document reading | 6% | finish a sentence from a real document | reading |
| stories | 8.5% | continue a book with its earlier pieces in memory; generated story worlds with character notes, questions, repairs | conversation |
| conversations | 11% | real dialogues (UltraChat, OASST), greetings, story requests | chat |
| plain text | 10% | whole documents, so language does not fade | - |

Each kind has held-out windows; training stops when the combined held-out score plateaus (30,000-48,000 steps).

## Reproduce C2b

The training data is in the repository but not in a normal clone (Git LFS skips `datasets/`). Fetch and unpack it (590
MB compressed, 2.75 GB unpacked; details in [`datasets/README.md`](datasets/README.md)):

```bash
git lfs pull --include="datasets/c2b_5m/*" --exclude=""
tar -xJf datasets/c2b_5m/c2b_5m_training_data.tar.xz        # from the repo root: files land where the trainer reads them
mkdir -p data && g++ -O2 -std=c++17 -static -o data/cv0.exe compiler/native/cv0.cpp   # the native encoder (once; drop -static on Linux/macOS)
bash compiler/experiments/chains/train_best.sh 1 c2b_repro_s1  # trains the BEST recipe (~2 h on an RTX 3090) and runs the full suite
```

The recipe (`recipes/BEST.json`, base.train): core `runs/core_mt5m/scale_custom_v0_8k.pt`; `--data-dir data_real/5m_mixed
--mix v22 --parts longconv,longdocs,alpacaqa,multi2`; the shares above; `--min-steps 30000 --max-steps 48000 --patience 10
--memory-opts '{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}' --gpu-frac 0.6`; answer with
`--engine-opts '{"search": "v2", "hop2": true, "split": true}'`. Tests and the verdict:

```bash
python -m chat22.tests.run_suite runs/chat22/NAME.pt --tier full --engine-opts '{"search": "v2", "hop2": true, "split": true}' --tag _ev2full
python -m chat22.tests.run_suite runs/chat22/NAME.pt --tier full --engine-opts '{"search": "v2", "hop2": true, "split": true}' --sections afterno,phrasing,probe2,reason,twomem,open,multi --tag _ev2new
python research_side/compare_two_seeds.py NAME              # front by front against BEST, 2 standard errors
```

## Continue the research

- **The loop** (`research_side/loop_5m.py`, `recipes/LOOP.json`): each candidate = BEST's recipe + changes (training
  switches, data, shares, answer-time options); seed 1, full suite + new tests, comparison; a second seed for a win with
  no front lost; a two-seed winner is written into `recipes/BEST.json`. The loop is paused at C2b; the next candidate
  (c4: counting and recaps on C2b's recipe) is queued.
- **Adding or strengthening a reasoning skill:** the method and what 13 runs taught us are in
  [`docs/CHAT5M_V2_DESIGN.html`](docs/CHAT5M_V2_DESIGN.html#c2b) (test first, a switch that leaves old recipes
  byte-identical, retrieval before training, dose and form).
- **Rules:** bf16 training; one training at a time, at most 3/4 of the GPU; the permanent store holds real facts only;
  results are answer accuracy on held-out questions; every run writes `results/*.json` and a line in `LOG.md`.
- **Train the 100M:** [`docs/TRAIN_100M.md`](docs/TRAIN_100M.md) (its own data bundle in `datasets/100m/`; the C2b recipe
  on the 109M core runs at ~0.85 s per step in 11.5 GB of GPU memory).

## What we learned (13 runs, 18-20 October)

| finding | evidence |
| --- | --- |
| Every new reasoning skill is learnable at 5M | counting, first thing told and summaries went from 0-5% to 90-100%; multi-part messages 27% -> 77% |
| The search decides what the model can combine | "Who wrote my favorite song?" never retrieved the stored fact (0/48) until the two-hop search (41/48 in the window); two questions in one message needed one search per question (0/12 -> 10/12 facts found) |
| Dose matters | counting/summary windows at 30% of recall practice cost documents (55% -> 38%) and near-miss facts; even 15% cost something |
| Form matters | every "told fact + stored fact" window form taught answering from stored facts over the conversation: story questions fell from 76% to 25-41% |
| Combined runs need attribution | a combined run that loses must be split to find the culprit; one change per run made each cause clear |
| The 5M learns patterns, not new wordings | skills 92% on training-style questions vs 39-47% on new wordings; the reason for the 100M |
| Earlier: memory + copying beats a bigger network for facts | network alone 13% F1 on stored questions, with memory and copy head 89% |
| Earlier: speed is launch overhead at this size | one CUDA graph per step: answers 4-7x faster, identical replies |

## Repository map

```text
run_model.py          chat with C2b (menu 3), train on a folder, device settings
chat22/               the chat package
  engine.py           answering: memories, search (v2, hop2, split), window, streaming
  memory.py, search_v2.py, window.py, tools.py, model.py, blocks.py, tables.py, encoder.py
  train.py, windows_train.py   the chat stage: every training kind, --parts switches, --accum, held-out plateau stop
  data/               data builders (real datasets, fact store, any-length sessions, story worlds)
  tests/              run_suite.py (full suite), suite_v2.py (multi-part, reasoning, phrasing, two memories, probe v2, open-ended, refusals), gate22.py, scoring.py
recipes/              BEST.json (the best recipe and all evidence), LOOP.json (the loop's candidates)
research_side/        loop_5m.py, compare_two_seeds.py, natural_rewording_test.py, fresh_facts_test.py
compiler/             the lossless text compiler and core training (experiments/train_lm.py, scale_run.py)
runs/                 C2b (runs/chat22/combo2b_5m_s1.*) and the 5M core (runs/core_mt5m), via Git LFS
datasets/             c2b_5m/ (training data, 590 MB) and 100m/ (the 109M core, 405 MB): fetched on request
data_real/            the permanent store (LFS), data notes and plans
data_chat22/, data_core/, data_mt/, data_skills/   held-out test files the suite reads
docs/                 SETUP.md, TRAIN_100M.md, CHAT5M_V2_DESIGN.html (design + methods), RUN_LEDGER.md, RESEARCH_STORY.html
LOG.md, MASTER.md     the working log (every run and decision) and the one-page project state
```

Adam's design pages (`BRAIN_DESIGN.html`, `HYBRID_DESIGN.html`, `Traditional_Hybrid.html`,
`COMPILER_AND_MODERN_METHODS.html`, `MODEL_EXPLAINED.html`) hold the longer-term ideas.

## Reading order

1. This README, then [`MASTER.md`](MASTER.md) (one page: goal, rules, best recipe, where it is headed).
2. [`recipes/BEST.json`](recipes/BEST.json): the recipe, every adopted lesson and rejected change with evidence.
3. [`docs/CHAT5M_V2_DESIGN.html`](docs/CHAT5M_V2_DESIGN.html): the design, C2b's recipe and methods, how to add reasoning skills.
4. [`docs/SETUP.md`](docs/SETUP.md) and [`docs/TRAIN_100M.md`](docs/TRAIN_100M.md): commands for the 5M and the 100M.
5. [`LOG.md`](LOG.md): every run, newest at the bottom.
