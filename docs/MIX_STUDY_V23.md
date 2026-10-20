# Data mix study for v2.3 (16 October 2026)

*Purpose (Adam): decide the datasets and the share of each type for a smart, reasoning model that takes its facts from
memory, with conversation facts kept apart from uploaded facts. Do it at 5M, so the same numbers can size the datasets for
larger models. Evidence: the per-type held-out curves of v2.2 (both seeds), how often each dataset was seen, and the
five-model suite (arm A, v2.1 x2, v2.2 x2, identical cases). No new training was run for this study.*

## 1. What the learning curves say (v2.2, both seeds agree)

| type | 90% of its gain reached by step | held-out at the end | reading |
|---|---|---|---|
| skills | 8,500-9,000 | loss 0.013-0.016 | **saturated early**: the templates were learned, then repeated for ~14,000 steps |
| recall (short) | 8,000 | 0.052-0.061 | saturated early: too much of the easy kind |
| document reading | 12,500-13,500 | 0.157-0.159 | about right |
| buried evidence | 15,500-16,000 | 0.43-0.55 | **still improving**: too little |
| Q&A | 9,000-21,500 | 1.08-1.12 | about right; slow |
| conversations | 22,000 (still improving at the end) | 2.88-2.90 | **too little** (and the hardest type) |
| stories | 16,500-21,000 | 3.81-4.15 | slightly too little |
| plain text | it got **worse** during training (5.05 -> 5.09 bits/code) | | **too little to hold the core's fluency** |

Skills saturated after less than half a pass over 127,635 windows. More of the same data would not help; the generator's
templates are too few. Diversity, not share, is the lever for skills.

## 2. How much of each dataset each model saw

| | arm A (480k windows) | v2.1 (480k) | v2.2 (365k) |
|---|---|---|---|
| Q&A (50,937 exchanges) | 2.8 passes | 2.4 | 1.4 |
| skills (127,635 windows) | 0.9 | 0.75 | 0.4 |
| conversations (205,623) | 0.35 | 0.35 | 0.23 |
| plain text | **48 passes over 4.1M codes** (the 15 MB bug) | 36 | **0.6 over 155M** |
| generated kinds | fresh every window | fresh | fresh |

## 3. Across models: what each share bought (identical cases)

- **Plain text: variety beats share.** Wiki bits/byte: A 1.353 (20% share, 48 passes over 15 MB), v2.1 1.340 (15%, 36 passes),
  v2.2 **1.321** (13%, 0.6 pass over 555 MB). But every chat model still reads worse than its core (1.260): the share is too
  low to stop forgetting. **Raise it to about 17%.**
- **Buried evidence: very sensitive to share.** A fact in a pasted document, memory off: v2.1 (15%) 66-74%, v2.2 (10%, plus 7%
  sentence-ending) **22-32%**. The sentence-ending task does not replace it. **Back to 15%**, sentence-ending down to 4%.
- **Fact format beats share.** v2.2 (15% skills, all told) told 38%, stored 39-42%; earlier models (20-25%, all stored)
  stored 62%, told 18-22%. **12% split half told, half stored**, plus diversity.
- **Conversations are slow learners.** A (15%) generated probe 58%; v2.2 (13%, with crowded memory) 28-36%; real phrasing is
  low for every model (3-6/27). **12%, plus the any-length kind**, whose filler is real dialogue turns.
- **Easy recall saturates; hard recall matters.** v2.1's recall training broke recall at 40 turns (5/15); v2.2's helped
  1,000-10,000 turns. **Move recall into the any-length kind (18%) with the real search**; keep 2% short recall.
- **Q&A** reached its plateau with 1.4-2.8 passes; **12%**, with look-alike facts in the memory block half the time.

## 4. The v2.3 mix

| type | v2.2 | **v2.3** | source of memory items |
|---|---|---|---|
| **any-length conversations (real engine search)** | - | **18** | conversation memory + uploaded facts |
| buried evidence | 10 | **15** | pasted document (the window) |
| plain-text rehearsal (555 MB) | 13 | **17** | - |
| Q&A (look-alike facts half the time) | 20 | **12** | uploaded facts |
| skills (half told, half stored) | 15 | **12** | conversation memory / uploaded facts |
| conversations (greetings, story requests) | 13 | **12** | - |
| stories | 10 | **8** | story notes (uploaded) + earlier scenes |
| document reading | 7 | **4** | the window |
| recall (short) | 12 | **2** | conversation memory |

## 5. Conversation facts and uploaded facts, kept apart

Two memories, as the engine already has them, now also two kinds of training material:

| memory | what goes in | label in the window | training material |
|---|---|---|---|
| **conversation** (temporary) | what the user said and the assistant replied, this conversation | `[turn N]` | told facts, corrections (newest wins), two-step links |
| **uploaded / stored** (permanent) | facts added with /add, Q&A knowledge, **uploaded documents split into pieces** | `[fact]` | stored skill facts, Q&A exchanges, document pieces |

Rules taught in training: the user's own statements about themselves come from conversation memory; the newest statement
wins; uploaded facts answer general and document questions; when neither has it, say so. The any-length kind asks about both:
about half conversation facts, a third uploaded facts (including document pieces), the rest never told.

## 6. Datasets to rebuild for v2.3

1. **Uploaded-facts pool** (`data_chat22/facts_pool.jsonl`): Q&A exchanges (train split), stored skill facts, and pieces of
   real documents from the long-document pool (as if uploaded). The engine's stored-facts pool and the source of the look-alike
   facts in training.
2. **Any-length sessions** (`data_chat22/longconv_*.jsonl`): real dialogues stitched into sessions of 10-10,000 turns, with
   planted facts, corrections, two-step links, look-alikes and never-told questions; for each question, the memory block the
   real engine search returns. About 70,000 question windows (18% of ~370k windows is ~67k).
3. Unchanged: rehearsal (155M codes, about 0.8 pass at 17%), conversations, Q&A text. Skills stay at v4c for v2.3 at a lower
   share; **a more varied skills set (v5) is needed for larger models** (more fields, phrasings and fact kinds).

## 7. What this means for larger models

Rule of thumb from the curves: real (non-generated) data should be seen **at most about 2 times**; generated kinds need more
*variety*, not more volume. For a chat stage about 4 times longer (about 1.5M windows, a guess for 100M):

| type | share | windows | unique data needed (<= 2 passes) | have now |
|---|---|---|---|---|
| Q&A | 12% | 180k | ~90k exchanges | 51k (need ~2x) |
| conversations | 12% | 180k | ~90k threads | 206k (enough) |
| plain text | 17% | 255k x 2,049 codes | ~260M codes | 155M (need ~2x) |
| skills | 12% | 180k | far more varied templates | v4c (needs v5) |
| any-length, buried, stories, document reading | 45% | 675k | generated; needs more source documents and dialogues | long pool 146M codes; 206k dialogues |

These numbers are estimates from one 5M recipe; v2.3's curves will refine them before any larger model is trained.
