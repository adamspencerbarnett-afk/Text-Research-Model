# Multi-turn conversation: which training structure works best?

*14 October 2026. Requested by Adam: test at 5M, before adapting the larger model, how a model should be
trained to hold a conversation, comparing sequential stages, one mixed stage, and LoRA adapters, with
good tests and a dataset built for it. The answer decides the training structure of the 100M+ models and
updates HYBRID_DESIGN.html (and notes in BRAIN_DESIGN.html).*

## 1. Question

Does a model hold a conversation best when trained as **core, then Q&A stage, then multi-turn stage**
(three sequential fine-tunes), as **core, then one mixed stage** (Q&A and multi-turn windows together), or
with **LoRA adapters** on a frozen core? And does any of them lose what earlier stages taught (recall of
stored facts, language quality)?

## 2. Everything carried in from earlier results

| from | applied as |
| --- | --- |
| best 5M chat recipe (sessions 23, 25: probe 24 and 23/32) | dense retrieval in the training loop, 30% reworded self windows, bf16, gentle core rate with fast heads, answer-only loss, end code, 30% rehearsal |
| Muon + warmup-stable-decay (sessions 20, 22: -0.9% text, two seeds) | the new core's recipe |
| data variety (5M core read its 300 MB mix ~15 times) | the new core trains on the 3.2 GB P100 mix, ~1.6 passes, the same mix as the 100M core |
| precision policy | bf16 training; bf16 answers; int8 check |
| clean evaluation (13 October) | report cards on data_core/eval_clean only |
| DLI course lessons (session 30, results today) | the gap is tracked in every run (--track-gap); two-phase fine-tuning, dropout and typo augmentation are applied to every arm if session 30 shows they help |
| retrieval confidence (session 28, results today) | --sim-feature in every arm if session 28 shows it helps |
| gradient accumulation | used where memory requires; identical updates |

## 3. Data (built 14 October, `prepare_dialogue.py`, no new downloads)

- **Training threads:** UltraChat 200k train split (207,865 conversations, mean 3.2 user turns) and
  OpenAssistant English trees (top-ranked reply at each turn), kept whole, at least two user turns.
- **Held-out threads:** about 1,000 conversations from UltraChat's test split and 1% of OpenAssistant,
  kept only if none of their replies occurs in the P100 training text (the core mixes were built from every
  UltraChat file, test split included, so most test conversations were seen as single exchanges).
- **Window layout for a multi-turn example:** `[retrieved exchange] [sep] [earlier turns, newest last] User: question
  Assistant: reply`, loss on the final reply only; earlier turns are trimmed from the oldest end to fit.
- **Long, multi-part messages are kept whole** (added 14 October, at Adam's question): 22% of user turns run past
  350 characters, 37% have three or more sentences and 14% ask two or more questions (a passage then a request,
  or several requests at once). The current message is kept up to 1,200 bytes (about 330 codes), so the model
  learns to read a whole paragraph and answer every part; reading a long message and reading a conversation are
  the same skill, so they train in the same stage rather than a separate one.

## 4. The new 5M core

256 wide x 6 layers, modern block (rotary positions let the window grow to 1,024 codes, which a multi-turn
window needs), Muon + WSD, bf16, P100 mix and dictionary, window 1,024, batch 32 (the same codes per step as
every 5M core), 42,538 steps. One core feeds every arm, so the arms differ only in how stages 2 and 3 train.

## 5. Arms

| arm | stage 2 | stage 3 |
| --- | --- | --- |
| **Q** control | Q&A stage (full fine-tune) | none |
| **S** sequential | Q&A stage | multi-turn stage from Q's model (full fine-tune; 30% Q&A windows and 30% core text rehearsed) |
| **M** mixed | one stage: Q&A and multi-turn windows 50/50, same steps as Q + S together | none |
| **L** LoRA | Q&A adapter on the frozen core (rank 16 on every block matrix; heads trained in full) | multi-turn adapter on top of the frozen Q&A adapter |

LoRA adapters are merged into the weights after training, so every arm is answered and timed by the same
engine and graph decoder. The best arm is repeated with a second seed before anything is adopted.

## 6. Tests (the same for every arm)

1. **Multi-turn probe** (new, `mt_probe.py`): 27 scripted conversations, keyword-graded, in six groups:
   recall of a stated fact ("my sister is called Ana ... where does my sister live?"), reference resolution
   ("the second one"), follow-ups, corrections ("actually it is 4471"), returning to an earlier topic, and
   multi-part messages (one paragraph with two or more requests; correct only if every part is answered). Each question is also asked
   without its history, so the score shows how much the conversation itself helped.
2. **History gain** on the held-out threads: bits per byte of each final reply with and without the earlier
   turns in the window (the conversation counterpart of evidence gain).
3. **Retention:** the Q&A groups (stored, added, reworded, new, network alone) and the 32-question chat probe,
   plus its typo version, to show nothing earlier was lost.
4. **Language retention:** the clean report card.
5. **Overfitting gap** and **cost** (training time, energy, seconds and joules per answer) for every arm.

## 7. Decision rules

- An arm wins if it is best on the multi-turn probe and history gain, and stays within seed noise of the
  control on retention (stored answers within 0.02 F1, chat probe within 1 point, clean card within 0.5%).
- Between near-ties, the cheaper and safer structure wins: LoRA over full fine-tunes (no forgetting, swappable
  skills), one mixed stage over sequential stages (fewer steps).
- If **S** wins (three stages), the stage structure of HYBRID_DESIGN 6 becomes core, Q&A stage, multi-turn
  stage, with the measured rehearsal shares. If **M** wins, stage 2 is one mixed stage. If **L** wins,
  adapters become the default for every skill after the core (and the consolidation design in 3.10 and
  BRAIN_DESIGN's synapse idea gain direct evidence). The decision and its numbers are written into both HTML
  documents either way.

## 8. Code to build (all opt-in, defaults unchanged, tested like the session 30 options)

- `prepare_dialogue.py` (built): the threads above.
- `m2_qa.py`: `--mt-data DIR --mt-p P` (share of windows built from a conversation, history trimmed to fit),
  `--history-codes N`; the engine and the chat take a history of earlier turns.
- `lora.py`: rank-r adapters on the block matrices, `--lora-rank R` (core frozen, adapters and heads trained),
  merged into the weights at save.
- `mt_probe.py`: the multi-turn probe; `history_gain` in the same script on the held-out threads.
- `run_model.py`: the chat keeps the last two turns in the window (after the winner is known).

## 9. Schedule

Today: data (CPU) and code, tested on the CPU while the 100M Q&A stage, session 28 and session 30 use the
GPU. The new chain (session 31) starts after session 30 and fills in the winning options from sessions 28 and
30: about 1 h for the core, 40 min per stage, 30 min of tests per arm: about 6 h in all. The dictionary study
(session 29) stays on hold.
