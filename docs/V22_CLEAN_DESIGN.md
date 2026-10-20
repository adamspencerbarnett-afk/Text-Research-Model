# Chat model v2.2: clean package design

*16 October 2026. For Adam's review before anything is built. The design is based on three read-only reviews (code path, tests, datasets) and on today's results.*

## Why a clean package

The v2.1 path reaches about 3,100 lines across 7 files (`train_chat.py`, `chat_engine.py`, `m2_qa.py`, `train_lm.py`, `dense_keys.py`, `ladder_encodings.py`, `gen_skills*.py`). Most of that code serves layouts, flags and methods we no longer use. The reviews found real defects hidden in it:

- the rehearsal-text bug: 15 MB instead of 300 MB in every run since 15 October;
- a memory-order mismatch in the new v22 layout;
- answer-time-only bans that training never sees;
- loose weight loading.

The old scripts stay untouched so past results can still be reproduced. v2.2 and every larger model after it are built from the new package only.

## 1. Package: `chat22/` (about 1,500 lines, 10 modules)

| module | contains | from |
|---|---|---|
| `encoder.py` | the dictionary encoder (V0Dict) and the EOS / SEP / memory-heading constants | wraps existing code unchanged |
| `blocks.py` | RMSNorm, rotary positions with a scale setting (for S42), ModernBlock with a cached forward pass | copied from `train_lm.py` |
| `model.py` | ChatLM: the core, the copy head and the trust head. It has one scoring function shared by training and answering, one decode step, strict weight loading (a missing part is an error) and a correct parameter count | rewrite (merges four overlapping functions) |
| `tables.py` | n-gram count tables, saved inside the checkpoint; the leave-one-out correction is applied only to text in the tables | copied, with the fix |
| `memory.py` | the word key, search, the exact-match rule, and two pools (stored facts and conversation memory) with session ids. The dense key is a plug-in: MiniLM as the reference, K2 later | rewrite |
| `window.py` | the v22 layout only; one fixed item-order rule; source labels; spans and the "nothing here" position | rewrite |
| `windows_train.py` | qa, skills, convo and greetings, recall v2, buried evidence, rehearsal, any-length conversations; every one built through `window.py` | rewrite from `train_chat.py` |
| `train.py` | the training loop, the evidence and pointer losses, the schedule, resume, held-out loss every 500 steps, and a saved config | rewrite |
| `engine.py` | load, two-pool search with gate and budget, session memory, generation with the history cached between turns, reply and verify | rewrite, no inheritance from the old QAEngine |
| `tools.py` | arithmetic and calendar | copied |

Plus `chat22/data/` (dataset builders) and `chat22/tests/` (the suite, section 4). `run_model.py` gains a "chat22 model" path that uses the new engine.

## 2. Design decisions to make on purpose

Each of these was a silent default in the old code. I propose a choice for each; the reproduction step (section 5) tests the ones marked *test*.

| # | item | old behaviour | proposal |
|---|---|---|---|
| 1 | Memory item order | v22 put the *first* item nearest the message, and training lists corrections as [old, new], so training placed the old value nearest the message, while the engine orders by search score | **Stored facts first, then conversation items by turn, newest nearest the message**, each labelled `[fact]` or `[turn N]`. Search only *selects* items; it never orders them. Training and answering use the same function, so they cannot drift apart |
| 2 | How many items | top 3 within 0.25 of the best; an exact match removed every other candidate | **Everything that passes the gate, up to the budget** (a quarter of the window, at most 160 codes per item). An exact match is placed first but removes nothing |
| 3 | Reply room | answering reserved room for the reply, training did not, so fewer turns fit at answer time | Training reserves the same room. History, memory, message and reply always fit identically |
| 4 | Banned codes | 186 codes never seen in the Q&A text were banned at answer time only, which blocks copying rare words from pasted text | **Drop the ban** (*test*) |
| 5 | Repeat ban | no 3-code phrase twice, at answer time only | Keep it for free writing, but exempt text copied from the window (*test*) |
| 6 | n-gram tables | built from the Q&A text at load time; a different data path silently changes them; leave-one-out applied to every window | Keep them (*test*: on vs off), saved in the checkpoint, leave-one-out only on table text |
| 7 | Copy head | full code-against-code attention: 2–3 GB at 2k, growing with the square of the window | At 2k: the same maths, computed in chunks of queries, only over copyable positions. Before S42: a top-k or local copy attention (*test*) |
| 8 | Positions past training | the 1,024 core was used at 2,048 with no rescaling | A scale setting in `blocks.py`, off for v2.2 and an arm in S42 |
| 9 | Session vs permanent memory | one counter (`base_n`); `/add` turned the whole session into permanent knowledge | Explicit session ids; `/add` writes to the stored facts; `forget()` clears only the session |
| 10 | Dense key | MiniLM, weight 2.0, run up to 4 times per reply | Computed once per reply; MiniLM labelled as the outside reference in every result; K2 can be plugged in |

## 3. Data: fixes and new sets

**Fixes (before any v2.2 training):**
1. **Rehearsal text:** rebuild from random *whole documents* across all P100 sources, at least 150M codes (about 550 MB). The file is named by its size and recipe, so an audit or smoke sample can never be reused again.
2. **Long-document pool for S42/S43:** every document above 8k codes (146M codes) and above 32k (70M, 87% books), stored apart from the rest.
3. **Conversations:** write the deduplicated, held-out-filtered dialogue file to disk, rather than filtering at load time.
4. **Clean evaluation text:** rebuild `eval_clean` against `data_mt`, `data_qa` and the new rehearsal sample. 32% of its ultrachat paragraphs appear in the training conversations; wiki and books, which the guard uses, are clean. Grow the clean evidence pairs from 203 to about 2,000.

**New sets (generated):**
5. **Any-length conversations:** real dialogues stitched into sessions of 10 to 10,000 turns, log-uniform in length, with:
   - planted facts, same-field distractors, corrections (some by name), two-step facts, and questions with no answer;
   - windows built with the real search, so the model trains on what memory actually returns.

   Size: about 50,000 sessions for 5M. The filler vectors are computed once and cached, so building sessions doesn't re-embed every turn.
6. **Buried evidence from real documents:** the answer is a sentence *from the document itself* (cloze-style: who, where, which), not an inserted Alpaca answer. About 100,000 windows.
7. **Stories:** a test only in v2.2. The story loop is outline, then scene, then extract facts into memory, then check, then the next scene, with contradictions counted. Story training data comes after the test shows where it fails.

**Gaps for later (decide before S44 / 100M):**
- We have 0.9B codes of core text against about 2B needed at 100M.
- Long documents above 32k are mostly 341 books.
- Dialogues are 99% UltraChat, and none is longer than 7 user turns.

Candidate sources: more Gutenberg, FineWeb-Edu, whole Wikipedia articles, more dialogue sets (licences to check). The Alpaca and QQP licences are non-commercial: fine for research, a problem for any release.

## 4. Test suite: `chat22/tests/run_suite.py MODEL --tier quick|full|long`

One runner and one scorer, writing `results/suite_<model>.json`. The guard reads only that file.

- **One scorer:**
  - a single abstain matcher with negative patterns ("you don't have to" is not an abstention);
  - whole-word answer matching;
  - a *wrong-value veto*: naming both the old and new value fails;
  - abstention scored as precision and recall, so always saying "I don't know" can't pass.
- **Paired cases:** memory on/off, and arm against arm, use identical cases. At least 50 items per reported cell, 150 per guarded front, 300 for headline totals, with 95% intervals.
- **Two seeds per arm:** if the two seeds of the same arm differ by more than the gap between arms, the comparison is called inconclusive.
- **Sections:**
  - **text:** wiki and books bits/byte on clean text; position bands per domain up to 2× the window;
  - **skills:** v2, rewriting, and v1 kept as a development set, outside the guard;
  - **conversation:** one set merging mt_probe, mt_probe2, recall and the chat probe, with three splits: new templates, real phrasing (expanded from 27 to 150+), reserved names. The chat probe's lookup items (stored, knowledge, tool) are scored separately;
  - **long:** the grid rewritten paired at n≥50, with lengths 512 to 4k for v2.2 and up to 65k for S42/S43;
  - **memory:** any-length conversations at 100 / 1,000 / 10,000 turns, judged by a flat line; a memory block with 1 / 3 / 10 / 30 items, right item at each rank;
  - **cache:** the same conversation replayed with the history cache on and off must give *identical* replies at every turn, including scroll-out, memory changes, verify and forget;
  - **latency:** cold and warm first-word time, p50/p95 over a 200-turn session, GPU and CPU.
- **Guard:**
  - floors are the baseline minus 2 standard errors of the difference;
  - a missing front counts as a failure;
  - the long-context fronts have floors too.

## 5. How we prove the package is set up correctly

1. **Equivalence tests (no training).** Load arm A and v2.1 seed 1 into the new model class:
   - the scoring function must give the same loss as the old code on the same windows;
   - a decode step must give the same next-code probabilities;
   - the old v2 window builder lives in the test folder only, as a reference.

   Any difference is a bug in the new code.
2. **Engine check (no training).** For a v2-trained model, the new engine with v2-compatible settings must give the same replies as the old engine on the old test cases.
3. **v2.2 training, two seeds,** on the fixed data, with the decision-table tests (items 4–7 above) as arms only where they matter.
4. **Judge by the full suite and the guard.** v2.2 passes if every front holds and the long-context and memory sections improve on v2.1. Only then S42 (8k).

## 6. Order of work and time

| step | what | time |
|---|---|---|
| a | data fixes 1–4, any-length conversation builder, real-document buried QA | ~3 h build, ~1–2 h CPU |
| b | package modules + equivalence tests | ~4 h |
| c | test suite + scorer + guard | ~3 h |
| d | v2.2 seed 1 and 2 training (GPU) | ~2.5–3 h |
| e | full suite on A, v2.1 and v2.2 | ~2 h GPU |

The GPU stays busy meanwhile: v2.1 seed 2, its grid and guard (session 37), then sessions 38–40 (memory to 5,000 turns, CPU stress, disk speed).

## 7. Adam's answers (16 October)

1. Name: `chat22/`.
2. Decision table: accepted as proposed.
3. More data: **wait until the 100M**. v2.2 must be comparable with the earlier 5M models to show what worked and what did not.
4. Stories: **story training starts now**, in v2.2 (section 8).
5. Asked: does the trainer stop at the right point instead of over- or under-training? It did not: a fixed 30,000 steps, no held-out loss. Section 9 is the answer.

## 8. Story training in v2.2

Two new window kinds, both built through `window.py`, so the story facts arrive the way the engine delivers them.

- **Book continuation with memory (real text):** the target is the next passage of one of the 446 whole books. The window holds the recent text. The memory block holds earlier pieces of the *same book*, found by the real search using the names and words of the recent text. Rule: the target must mention a name that appears in the memory block but not in the recent window. This teaches what long story writing needs: continue the text, and pull in older facts from memory. Held-out books stay out.
- **Generated story worlds (checkable facts):** a story bible (3–6 characters with name, role, trait, home, possession; places; a timeline of events) and an outline of scenes. Scenes are rendered from templates plus real sentences. The windows are of three kinds:
  - write the next scene, with the bible facts that apply in the memory block;
  - answer a question about an earlier scene (who, where, what happened);
  - fix a contradiction ("Mira lives in Oslo" when the bible says Bergen, which should be corrected).

  Every fact is known, so consistency is scored exactly.
- **The story test (suite):** outline, then scenes written one at a time, with facts extracted into memory after each scene. Planted facts are checked later at 2k / 8k / 32k codes of story. The score counts contradictions per 1,000 codes and planted facts kept.

## 9. Training that stops at the right point

`chat22/train.py` decides when to stop from held-out measurements, not from a fixed step count:

1. **Held-out windows of every kind:**
   - built once, from held-out sources only: held-out Q&A (1,042), reserved-name skill and recall windows, held-out dialogues, held-out books, clean wiki;
   - fixed, so every evaluation is comparable;
   - scored every 500 steps, as loss per kind plus the evidence-target accuracy.
2. **A quick task check every 2,000 steps:** a small, fast slice of the suite (about 200 items: skills, never-told, two-step, document, conversation), which measures answers, not just loss.
3. **A schedule that follows the measurements (warmup-stable-decay, already adopted for the core):**
   - **warm up,** then hold the learning rate steady while the combined held-out score improves;
   - **plateau** means no improvement of more than 0.5% over the last 3 checks (6,000 steps);
   - **then cool down** over the next 20% of the steps taken so far, and stop. That's the step most of the final gain comes from.
4. **Keep the best checkpoint,** not the last. The combined score weighs every front, so a gain on one kind can't hide a loss on another.
5. **Overfitting alarm:** if a kind's training loss keeps falling while its held-out loss rises over 2 checks, that kind is flagged and its share can be lowered. The skills loss of 0.01 in v2.1 is the case to watch.
6. **Limits:** a minimum and a maximum number of steps (for example 10,000 to 60,000 at 5M), resume after an interruption, and every check logged in the run's result file with its cost.

## 10. Questions for Adam (answered in section 7)

1. Package name `chat22/` (or another)?
2. Decision table: accept the proposals, or change any? Items 1, 2 and 4 matter most.
3. More data now or later? The 5M v2.2 run is fine with what we have plus the generated sets. 100M will need more text and more dialogue sources, and downloads mean picking sources and licences.
4. Stories: test-only in v2.2 (proposed), or generated story training data now?
