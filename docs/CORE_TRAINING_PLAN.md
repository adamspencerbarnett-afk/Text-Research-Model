# What to train the core on, and how to feed it

*10 October 2026. Research concept and direction by Adam Barnett. Written from the results in
`docs/RUN_LEDGER.md`, `docs/CHAT_NOTES.md` and `docs/ENCODER_RESEARCH_PLAN.md`. This defines the
large-scale test for this machine (one RTX 3090, 64 GB) and the data it needs.*

## 1. The core's job in this design

In a conventional language model the network must hold all knowledge. In ours, knowledge lives
in memory (added in seconds), skills in adapters, exact arithmetic in the calculator. So the core
has three jobs, and its training data should be chosen for them, in this order:

1. **Understand and write modern English**, including conversation, because every question and
   answer passes through it.
2. **Use evidence placed in front of it**: read a retrieved passage or exchange and carry the
   right part into the answer. This is what makes memory useful; it is where every gain so far
   came from (copy head: 0.13 → 0.88 F1 on stored questions).
3. **Represent questions well**, so its own state can later serve as a retrieval key that
   catches rewordings and synonyms (the main gap in the chat probe).

Broad factual knowledge in the weights is a bonus, not a requirement.

## 2. Lessons from the research so far that decide the data

| # | Lesson | Evidence | What it means for core training |
| --- | --- | --- | --- |
| 1 | A lossless encoder adds no knowledge; compression is not intelligence | compiler = BPE ±1%; phrase codes shorter and worse | Choose data and feeding for content, not for compression. Keep codes stable. |
| 2 | Language first, task second | Adapted core beat every Q&A-only core in every group; a 4.7M core trained on 19 MB of Q&A memorised it in minutes | Stage 1 needs a large amount of general text; task data is small and comes after. |
| 3 | The core learns to use evidence only if training shows it evidence | Copy head learned from windows where the retrieved exchange was the answer (50% self windows) | Put evidence windows into core training itself, not only into Q&A fine-tuning. |
| 4 | Count tables help a small core and hurt a larger one | +10% at 0.8M, 0 at 4.7M, −1% at 19M | Train the 50M+ core without tables for prediction; keep them as a reading store only. |
| 5 | What the data lacks, the model cannot do | Small talk 1/6 (Alpaca has none); arithmetic 0–2% at every size | Conversation and worked procedures must be in the mix. |
| 6 | How exchanges are cut decides what is learned | Paragraph split cut 6.6% of answers; SQuAD rambling because no window showed an answer's end | Keep documents and exchanges whole, with an explicit, learned end. |
| 7 | Duplicated text is memorised, not learned | SQuAD passages repeated 3–5 times: validation rose from minute 1 | Deduplicate; put the loss on what should be learned (answers, not repeated passages). |
| 8 | Formatting noise costs capacity | Wikipedia split is raw markup (9 `[[` per KB); books hard-wrapped at 54 bytes per line | Clean text: strip markup, unwrap paragraphs, normalise whitespace. |
| 9 | Retrieval by literal words misses rewordings | 1/10 reworded questions in the probe | Train the core on paraphrase pairs so its states can serve as keys. |
| 10 | Equal time is not equal training | 19M got 6,694 steps against 15,565 for 4.7M in the same 600 s | Budget by tokens seen (about 20 per parameter), not minutes. |
| 11 | Seed noise is ±0.07 F1 on 60 answers | two seeds of the best model | Two seeds before trusting a data change; bigger eval sets for the large test. |
| 12 | The tools that kept results honest | leave-one-out, train-only fitting, JSON per run, the ledger, the probe battery | Keep all of them for the large test. |

## 3. What to train the core on

Proportions by codes seen in stage 1. Sources are public with licences that allow research use;
the first two are already on this machine.

| Share | Kind | Sources | Why (lesson) |
| ---: | --- | --- | --- |
| 35% | Encyclopedic prose, cleaned | Wikipedia: `enwik9` on disk (0.93 GB raw, about 0.6 GB clean), then a full English dump (about 20 GB clean) | Modern, factual, varied vocabulary (1, 8) |
| 25% | Educational web text | FineWeb-Edu sample (ODC-By) | Modern explanatory English, close to how people ask and answer (1) |
| 10% | Books, unwrapped | corpus v2 books with paragraphs joined | Long-range coherence, narrative (1, 8) |
| 10% | Conversation | OpenAssistant oasst1 (Apache 2.0), Dolly 15k (CC BY-SA), UltraChat (MIT) | Small talk and multi-turn did not exist in training (5) |
| 15% | **Evidence windows** (built by us) | Wikipedia and web paragraphs with a related paragraph retrieved by our memory placed before them; plus SQuAD and Natural Questions passage–question–answer | Teaches the core to read and use what retrieval gives it, at scale (3) |
| 5% | Worked procedures | GSM8K, SVAMP, MetaMathQA (MIT) in step-by-step form with `Answer:` lines | Arithmetic set-up as a skill the core has at least seen (5) |

The evidence windows are the part that makes this core ours rather than a generic small model.
The idea is from retrieval-augmented pre-training: a model trained with related text in its
window learns to rely on that text. We have the retrieval already (the exchange memory works on
paragraphs as well as on questions), so building these windows costs only preprocessing.

## 4. How to feed it

1. **Clean.** Strip wiki markup and XML, unescape HTML entities, drop tables, references and
   templates; unwrap hard-wrapped lines inside paragraphs (keep blank-line paragraph breaks);
   normalise whitespace; keep UTF-8. Each source gets a manifest with hashes, as now.
2. **Deduplicate.** Exact and near-duplicate paragraphs removed across sources (hash of
   normalised text, then shingle overlap); the held-out split is chosen by document, so no
   held-out text appears in training in any form.
3. **Refit the dictionary on the final mix.** The current dictionary was fitted on 19th-century
   books plus wiki markup. Refit at 8k on the cleaned mix; test 16k at 50M (a larger core can
   afford a larger vocabulary). Consider the scanner fixes from the first review at the same
   time (punctuation runs and UTF-8 characters as single units), since a refit is needed anyway.
4. **Mark boundaries.** Documents are separated by a reserved code; chat exchanges end with
   `\n\nUser:` or an end code that is in the loss, so the model learns where an answer stops.
5. **Put the loss where the learning is.** Plain text: every code. Evidence windows: the target
   paragraph or answer only, not the retrieved evidence. Conversation and Q&A: the assistant's
   turns only.
6. **Curriculum in three stages.** Stage 1: the mix above. Stage 2: the Q&A design (memory, copy
   head, trust head) on Alpaca + Dolly + oasst + SQuAD with reworded-question training (R0b).
   Stage 3: skill adapters (BRAIN_DESIGN B2).
7. **Budget by tokens.** About 20 codes per parameter in stage 1; equal steps when comparing
   sizes; plateau stopping only as a guard, never as the comparison.

## 5. The large-scale test

| | Core A (control) | Core B (ours) |
| --- | --- | --- |
| Size | 50M (then 125M) | same |
| Stage-1 data | the same cleaned text, plain only | the same text with 15% evidence windows and 10% conversation |
| Codes seen | 1B (2.5B at 125M) | same |
| Time on the 3090 | about 4 hours (about a day at 125M) | same |
| Stage 2 | identical Q&A design, two seeds each | same |

Measured on both, with the tools already in the repository:

- bits per byte on held-out clean text (books, Wikipedia, web) and on the old v2 held-out books;
- the chat probe battery (graded), the Q&A groups (stored, added, reworded, new, no memory);
- SQuAD exact match and F1; GSM8K/SVAMP final-number accuracy; reply speed;
- the same at 4.7M with the new data, so the data effect and the size effect separate.

Pass conditions for moving on: Core B beats Core A on reworded questions and SQuAD by more than
seed noise, without losing more than 1% in bits per byte; the 50M result beats 4.7M on the
no-memory floor and the probe. If Core B does not beat Core A, the evidence-window idea is wrong
for this design and the mix reverts to plain text plus conversation.

## 6. Order of work

1. Data pipeline: cleaners for Wikipedia (`enwik9` first, already on disk), the books, FineWeb-Edu,
   the dialogue sets; deduplication; manifests. CPU work, can run beside the current GPU queue.
2. Evidence-window builder using `ExchangeMemory` over paragraphs.
3. Refit the dictionary on the clean mix; check the compiler tests.
4. A 4.7M pilot on the new data (an hour) to catch feeding bugs cheaply before the 50M run.
5. The 50M A/B, then 125M on the winner.
