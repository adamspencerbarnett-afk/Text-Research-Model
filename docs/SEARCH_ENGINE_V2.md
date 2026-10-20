# Search engine v2: design and plan

*16 October 2026. Adam's brief: a search that gathers far better from memory than today's and, if at all possible, is
faster; if a part must be slower, say why and what it buys. Built on today's retrieval study and speed measurements
(research_side/retrieval_study.json, research_side/engine_v2_speed.json); nothing here is a guess where a number exists.*

## 1. What today's engine gets right and wrong (retrieval study, 400 conversation cases + held-out stored facts)

| | today |
|---|---|
| right item reaches the block: one fact / distractor / correction | **100% / 100% / 100%** |
| two-step (both linked facts) | 88% (55% without name-following) |
| stored skill facts found | 98% (word key alone 93%, K2 95%) |
| a look-alike's value also in the block (distractor) | **49%** |
| the outdated value also in the block (correction) | **99%** |
| never told: a look-alike in the block | **52%** |
| items per block, of which stored trivia | 11.8, 6.0 |
| time per search, CPU | word 9-10 ms, word + MiniLM 34-40 ms; flat from 100 to 10,000 entries |

**Finding is solved; choosing is not.** The engine hands the model the right fact plus the outdated value, the look-alike
and six unrelated stored facts. Step 1 showed what that costs: the same model goes from 43% to 74% on single facts and from
20% to 76% on "never told" when the engine simply lets less in.

## 2. Architecture: a write path and a read path

The expensive work moves to the moment something is stored (after the reply is on screen), so each question only reads
what was prepared.

**Write path** (each new turn, reply, uploaded fact or document):

| step | what | status |
|---|---|---|
| W1 | split into pieces (sentences, ~200 bytes) | have it |
| W2 | add to the word index | have it; becomes a sparse matrix (vectorised) |
| W3 | meaning vector computed once (MiniLM now, K2 when better) | have it |
| **W4** | **fact table**: records pulled from statements: (who, attribute, value, turn); a correction replaces the value and keeps the old one as history; "my sister is called Kim" is a link | new |
| W5 | uploaded documents: pieces with their title as the source | have it (v2.3 pool) |

**Read path** (each question):

| step | what | why here | measured / estimated time |
|---|---|---|---|
| **R0 understand** | normalise; find **who** (my sister, Kim, "she" = the last person mentioned) and **what** (city, job, price); the question type (personal fact / general / chit-chat / tool) | everything after works better knowing who and what is asked | ~1 ms (rules first) |
| **R1 fast lane** | fact table lookup by (who, what): the newest value and its turn; a known person without that attribute = a **never-told signal**; links answer two-step questions | exact, instant; solves the three worst cases (corrections 99% outdated-in-block, look-alikes 49-52%, two-step 88%) | **0.0005 ms** |
| **R2 candidates** | word search (sparse matrix) + meaning key over both memories, top ~50, merged | catches what the fact table does not hold: uploaded documents, general Q&A, reworded questions | word **3.95 ms** (vectorised; today 9-10); meaning 12.8 ms GPU / 21 ms CPU |
| **R3 judge** | a small model of our own reads the question and each candidate together and scores "does this answer it?", one batch | precision: "Kim's city" vs "Eli's city", "my grandmother's school" vs "a school uniform" | 2 layers x 40 candidates: **3.8 ms GPU**, 59 ms CPU (6 layers: 10.4 / 171) |
| **R4 decide** | keep what clears a **calibrated** bar; newest first; linked facts added; fit the budget; **an empty block when nothing clears it** | a short clean block keeps a small model grounded (side test: 11/12 vs 0/12 correct "I don't know") | < 1 ms |
| R5 | window + model (unchanged) | | |

The fast lane skips R2 and R3 when it is certain (an exact who-and-what hit with no conflicting record).

## 3. Speed: faster than today on the GPU, slower on a CPU-only machine unless trimmed

| path | today | engine v2 (GPU) | engine v2 (CPU only) |
|---|---|---|---|
| personal fact, fast lane certain | ~22 ms GPU / 34-40 ms CPU | **~2 ms** (R0 + R1 + R4) | **~2 ms** |
| full path (documents, general, reworded) | ~22 ms GPU / 34-40 ms CPU | **~18 ms**: word 4 ms on the CPU while the GPU does the meaning key (12.8 ms), then judge 3.8 ms, decide < 1 | ~85 ms (judge 59 ms); **~45 ms** with the judge on the top 10 only; or skip the judge and use the calibrated meaning score |

**Honest summary:** on this PC (GPU), engine v2 is faster than today on every path: about 10x faster for personal
facts and slightly faster on the full path, while choosing far better. On a machine with no GPU, the judge is the cost: the
full path becomes slower than today unless the judge looks only at the top 10 candidates (then roughly level).

**Further speed, later:** the MiniLM call is mostly fixed overhead (tokeniser, Python); a K2 head on our own core would
cost a few ms on the GPU. An approximate-nearest-neighbour index only matters past a few million memory items.

## 4. The parts in detail

- **Fact table (W4, R1).** Schema: who (normalised: "my sister kim", plus the link "my sister" -> "kim"), attribute (the
  fact generator's fields: city, job, birthday, pet, allergy, phone, ... and event fields: when, where, price), value,
  turn, history of replaced values. Rules first, taken from the statement patterns of our generators (60,000 training /
  3,000 held-out statements with ground truth already built); a small learned extractor later if the rules miss real
  phrasing.
- **Question understanding (R0).** Rules first (the generators' question patterns + type cues); a small learned tagger
  later (60,000 / 3,000 questions with ground truth built: personal fact, general, chit-chat, tool; who; attribute).
  Pronouns: "she", "he", "it", "that one" resolve to the last person or thing mentioned in the recent turns.
- **Candidates (R2).** The word index as a sparse matrix (one product per question); the meaning key on the GPU; both
  memories, top 50 each. Today's recent-turns search is dropped (the study: look-alikes 52% -> 38%, two-step 88% -> 95%
  without it); name-following is replaced by the fact table's links (the study: without links two-step falls to 55%).
- **Judge (R3).** Our core's first 2 blocks (initialised from the 5M core) plus a scoring head; input
  `Question: <q>\nMemory: [source] <item>`; trained on (question, item, label) pairs with hard negatives from what the
  real search returns (look-alikes, outdated values, same-topic stored facts, near-miss skill facts). 2 layers first
  (3.8 ms GPU); 6 layers only if 2 is not precise enough.
- **Decide (R4).** The bar set on held-out pairs so that look-alikes rarely pass and real facts rarely fail (target below);
  newest wins among records of the same who and attribute; a two-step answer brings its link; the window rule places
  stored facts first, then conversation items oldest to newest.

## 5. Datasets

| set | size | source | status |
|---|---|---|---|
| fact extraction (statements -> records, corrections, links) | 60,000 train / 3,000 test (held-out names and phrasings) | fact generator ground truth | **built** (data_chat22/search_v2/extract_*.jsonl) |
| question understanding (type, who, attribute) | 60,000 / 3,000 | generator + Alpaca + chit-chat + tool questions | **built** (query_*.jsonl) |
| judge pairs | ~1-2 million pairs expected | the any-length sessions (what the real search returned, labelled), skills records, Q&A own vs neighbours | **building**: the sessions are being generated now (4 x 1,250); the pairs follow automatically |
| evaluation | the retrieval study's 400 cases + held-out sessions + held-out stored facts | as above | in place (research_side/retrieval_study.py) |

## 6. Targets (engine v2 must beat today on every line)

| | today | target |
|---|---|---|
| right item in the block (one / distractor / correction / two-step / stored) | 100 / 100 / 100 / 88 / 98% | >= 99 / 99 / 99 / 95 / 98% |
| look-alike in the block (distractor) | 49% | < 5% |
| outdated value in the block (correction) | 99% | < 5% |
| never told: empty or look-alike-free block | 48% | > 90% |
| items per block | 11.8 | 1-4 typical |
| time per question, GPU: fast lane / full | ~22 ms | ~2 ms / <= 20 ms |
| end to end (suite): memory-on single fact / never told | 43% (default) / 74% (stricter) ; 20% / 76% | above stricter on both |

## 7. Build order and time

1. Fact table + question understanding (rules) + decide; vectorised word search (no training): ~3-4 h build, measured with
   the retrieval study the same day.
2. Judge: pairs from the sessions, a short training run from our core (~1 h GPU), calibrate the bar on held-out pairs.
3. End to end: the suite on the v2.3 model with engine v2 (no retraining needed); then v2.4 trains on engine v2's blocks.
4. Later: K2 as our own meaning key in R2; a learned extractor and tagger if the rules miss real phrasing.

Engine v2 is a separate component (the chat core uses it); it stays within the retrieval research, not a new core line.
