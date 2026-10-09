# Reasoning plan: from "repeats what it was told" to "answers what it was not told"

*9 October 2026. Research concept and direction by Adam Barnett. This plan follows
`DESIGN_V2.md` and the results logged in `ENCODER_RESEARCH_PLAN.md` section 8. Every estimate
below is an estimate; the measured number replaces it as soon as the run exists.*

## 1. Where we stand, in the terms that matter

| Ability | Mechanism | Measured | Ceiling reached? |
| --- | --- | --- | --- |
| Return a stored answer to its question, including one stored seconds ago | exchange memory (lookup, no parameters) | 99% exact | yes |
| Compose a reply that carries the stored answer | copy head over the retrieved exchange, 33k parameters | word F1 0.48 / 0.62 (two seeds); exact 3–12% | no; see 2.1 |
| Answer a reworded question | nearest-key retrieval, then the copy head | 0.20–0.23 F1; retrieval itself caps at 0.27 | no; retrieval-bound |
| Answer a question with nothing in memory | the network alone | 0.13–0.16 F1 (fluent, generic) | core-size-bound |
| Solve an arithmetic word problem (GSM8K, unseen) | none yet | 1.0% right | not started |
| Predict text (books), 4.7M core | tables + network | 1.4115 bits/byte, tie with the traditional model | tables no longer help at this size |

Two conclusions drive the plan. First, the copy head was the missing piece and it is cheap; its
0.48 is not the head's limit but the limit of how it is fed and trained (section 2.1). Second,
nothing in the design reasons yet: the 1% on GSM8K is the true starting point, and the measure
from here on is accuracy on questions with a checkable answer, never F1 alone.

## 2. What to change, and why

### 2.1 The copy head: fix how it is fed before building a bigger one

Why 0.48 and not 0.9 when the exact answer is in the window: four causes, all visible in the
runs, none of them "the head is too small".

1. **The evidence is truncated.** The retrieved exchange enters as its last 110 codes of a
   256-code window. An Alpaca exchange averages about 100 codes and many exceed 110, so the
   head often cannot see the *start* of the answer it is meant to produce. Fix: context 512,
   retrieved span up to 300 codes (the whole exchange). Expected: seen F1 0.48 → 0.7 or more.
2. **Training teaches a compromise.** Half the training windows show the exchange itself
   (copying is right), half show the nearest *other* exchange (copying is wrong), and the trust
   head learns an average. Fix: give the trust head the retrieval similarity as an input
   feature, and train on three window kinds (self, a degraded self with words dropped, an
   unrelated exchange) so it learns *when* to copy. Expected: higher exact rate, and the
   prediction score stops getting worse when the head is added (today 1.30 → 1.43).
3. **Sampling drifts.** Answers are sampled at temperature 0.5, top-k 10; one wrong code and
   the pointer loses its place ("becomeplomatic"). Fix: greedy decoding for scored answers, and
   a pointer that re-anchors (attend over the span, not only the last position). Expected: exact
   3–12% → 25–40% on seen and added questions.
4. **One retrieved exchange.** Rewordings often have the right answer in the second or third
   nearest exchange. Fix: place the top 3 in the window (fits at 512 codes with full spans).
   Expected: paraphrase F1 0.23 → 0.3 or more; the retrieval ceiling rises from 0.27.

Only after these four should the head itself grow (multi-head pointer, per-layer keys). The
order matters because each of the four is a day's work and measurable alone.

### 2.2 Bigger core, bigger head, or better training: which when

- **Seen and added questions** (knowledge the memory holds): the core size barely matters; the
  mechanism does. Spend here on 2.1, not on parameters.
- **Reworded questions**: retrieval quality first (keys, top-3), then the core, because the
  network must bridge the wording gap.
- **Questions memory has nothing for**: only the core and its training data help. The scale
  run says a 4.7M core predicts books at 1.41 bits/byte against about 1.8 for the 0.8M core; on
  Q&A the no-memory floor should rise from 0.13 to about 0.2–0.25 F1 at 4.7M–19M, which is as
  far as open-answer F1 can usefully go (reference answers are one of many valid ones).
- **Reasoning**: neither size nor head. It needs the number channel, the executor and the
  check loop (2.3), and a retrieval that returns *structurally* similar problems.

So: better feeding of the head now; a 4.7M core for Q&A as soon as the GPU is free (tonight);
a bigger head only if 2.1 stalls; 19M only for the reasoning sets once the mechanism works at
4.7M.

### 2.3 Reasoning: the number channel, the executor, the check loop, and template copying

What the GSM8K replies show: the model copies numbers from the *retrieved* problem, not from
the one asked; it writes digit runs with no sense of value; nothing verifies anything.

1. **Copy from the question** (`--copy-question`, running now): operands live in the question.
2. **The executor writes results** (`--calculator`, running now): the model writes `48/2 =` and
   the executor writes `24`. The model never has to do arithmetic, only to set it up. The GSM8K
   authors' calculator did the same and roughly doubled small models' accuracy.
3. **Answer check** (rule 2 of the check loop, running now): the stated `Answer:` must be a
   computed value, or the last computed value replaces it.
4. **Template copying.** Retrieval returns the problem whose *solution structure* matches (same
   operations in the same order), not the one with the most shared words. Key: the sequence of
   operators in the stored solution plus the question's content words; a second index. The copy
   head then copies the solution skeleton while copying operands from the current question. This
   is reasoning by analogy over a store of worked examples, and it is the design's own idea
   (deterministic key, inspectable store, small network) applied to procedures instead of facts.
5. **Number channel in the compiler.** A number unit becomes one code plus a value; the
   network sees "a number here", the executor sees the value. Removes the digit-run failure.
6. **The rest of the check loop**: verify against memory (contradiction with a stored fact),
   refine (re-sample the failing span), restart, or "I do not know" with the evidence. Measured
   by how often "unknown" is right, not only by accuracy.

### 2.4 The tables after the scale result

At 4.7M the count tables no longer improve prediction and hurt off-domain. They keep their
role as the store that learns by reading (one book added, 0.6% better on the author with no
training), so the change is to the trust head, which must learn to distrust them when the
confidence features look in-domain but the text is not. Two cheap screens: a domain feature
(retrieval similarity of the window to the training text) as a trust-head input; and tables
counted per domain (books, Wikipedia, Q&A) with the head choosing among them. If neither
restores a gain at 4.7M, the tables stay for learning-by-reading only and are off for
prediction at larger cores. The 19M pair of the scale run (tonight) decides whether the trend
continues.

## 3. The research sequence, with the measure and the estimate for each

Each step is a screen (CPU, 600 s, or GPU, 10–30 min), run with two seeds when it wins,
written into `ENCODER_RESEARCH_PLAN.md` section 8 with its cause.

| # | Step | Data | Measure | Estimate (honest) |
| --- | --- | --- | --- | --- |
| R0 | Copy head fed properly: context 512, full spans, similarity feature, three window kinds, greedy scoring | Alpaca | seen / added F1 and exact | F1 0.7+, exact 25–40% |
| R1 | Q&A design on the 4.7M core (GPU) | Alpaca | all groups | no-memory floor 0.13 → 0.2–0.25; seen as R0 |
| R2 | Copy from a passage: answer is a span of given text | SQuAD 1.1 | exact match | 30–50% at 4.7M (pointer models reach 60–70% at far larger size); the cleanest test of the head |
| R3 | Copy from question + executor + answer check (running now at 0.8M) | GSM8K, SVAMP | final number exact | GSM8K 1% → 3–8%; SVAMP 10–25% |
| R4 | Template copying: retrieval by solution structure | GSM8K, SVAMP | final number exact | GSM8K 10–25% if the store covers the templates, which is the open question; SVAMP 30–50% |
| R5 | Number channel in the compiler | GSM8K, books | exact; bits/byte on digit-heavy text | removes digit-run failures; +2–5 points on R4 |
| R6 | Facts in the window, one-letter answers | ARC, OpenBookQA, CommonsenseQA | accuracy | chance is 25%; 30–40% at 4.7M with the "book" retrieved; the hardest of the sets for this design |
| R7 | Deterministic reasoning over a given story (deduction, counting, paths) | bAbI (to fetch) | accuracy per task | 60–90% on the copy-and-count tasks; this is V1's ground and where the design should shine |
| R8 | Full check loop with "unknown" | all exact-answer sets | accuracy, and precision of "unknown" | fewer wrong answers at equal right ones |
| R9 | 19M core on the winning design | the sets above | all | decides the "smaller at equal quality" claim for reasoning |

## 4. Expected outcome, stated plainly

If the sequence goes as estimated: a model with a 4.7M-parameter network, a memory updated in
seconds, and a deterministic executor that answers stored questions almost exactly, answers
reworded ones about a third of the time, pulls exact answers out of a given passage about
half the time, and solves a tenth to a quarter of grade-school arithmetic problems it has
never seen, saying "I do not know" for most of the rest. That would be far below a large
language model and far above anything a 5M network does on its own, and it would be built from
inspectable parts. The failure that would stop it: retrieval not finding structurally similar
problems often enough (R4), which would cap arithmetic near the R3 level and move the work to
the number channel and a larger core.

## 5. Rules that stay

Checkable accuracy over F1; two seeds before building on a win; equal budgets; every run a
JSON and a dated log line with the cause; report failures plainly.
