# The 50M large-scale test

*11 October 2026. Research concept and direction by Adam Barnett. Every setting below is a result
from the 5M screens logged in `docs/RUN_LEDGER.md`; the reason is given beside it.*

## Question

Does the full design (compiler, exchange memory, copy head, trust head, calculator) improve when the
core grows from 4.7M to 49M parameters trained on the best data and recipe, and by how much on each of
the three jobs (fluent modern English, using evidence, question representations) and on answers?

## Stage 1: the core

| Setting | Value | Why |
| --- | --- | --- |
| Shape | width 640, 10 layers, 10 heads (49M in the blocks, 55M total) | the planned 50M size |
| Window | 1,024 codes, the whole run | 512 beat 256 on prose and evidence, and whole exchanges lifted memory answers 7-9 F1 points; 1,024 holds every Q&A exchange, retrieved answer and worked solution |
| Dictionary | 8k, refitted on the P50 mix | the dictionary must match the training text (lesson from the old books + markup dictionary) |
| Count tables | none in stage 1 | +10% at 0.8M, 0 at 4.7M, -1% at 19M |
| Data | P50 mix, 2.4 GB: web 32%, Wikipedia 25%, conversation 12%, evidence pairs 12%, worked math 8%, books 7%, question rewordings 0.8% | D1-D3e: modern clean text, conversation and evidence each improved the core |
| Packing | document separator code; random windows | random windows gave better fluency than paragraph-aligned ones; separators mark document ends |
| Amount | about 1B codes (about 1.5 passes) | about 20 codes per parameter; repeating good data up to about 4 times costs little |
| Optimiser | AdamW, peak 1e-3 (scaled down if unstable), warm-up, cosine decay to 10%, bf16, clip 1.0 | the stage-1 recipe that worked at 5M |
| Time | about 3-4 hours on the RTX 3090 | measured 640w x 10L speed, 1,024-code windows |

## Stage 2: the Q&A design

The S2b recipe, confirmed on two seeds at 5M: core lr 1e-4 and new heads 1e-3, warm-up and cosine,
loss on answers only, end-of-answer code, 30% rehearsal of stage-1 text, retrieved exchange up to 400
codes and own exchange up to 623 in the 1,024 window, 15,565 steps, 200 questions per scoring group.

## Measured

| Job | Measure | 5M best today |
| --- | --- | --- |
| fluency | report card, bits/byte on held-out Wikipedia, web, conversation | 1.595 / 1.585 / 1.296 |
| evidence | evidence gain; SQuAD exact match | 34.8%; 7% |
| question representations | reworded retrieval by core state, rank 1 | 51% |
| answers | Q&A groups (stored, added, reworded, new, no memory); chat probe | 0.89 / 0.86-0.88 / 0.26-0.30 / 0.27-0.29 / 0.20-0.24; 21-23/32 |
| reasoning | GSM8K new problems, before and after verified self-training | 2.5% -> 3.5% (round 1) |
| cost | reply speed, learned parameters, memory size | |

Pass condition for going further (125M or 1,024 to 2,048 codes): clear gains on the no-memory floor,
reworded and new questions and GSM8K, not only on stored ones (which memory already answers).

## Order and timeline

1. Evidence pairs for the P50 mix (CPU, running; slow: up to about 10 hours).
2. P50 mix, dictionary refit (CPU, under an hour).
3. Stage 1 (GPU, 3-4 hours), report card.
4. Stage 2 on Alpaca, chat probe; GSM8K stage 2; verified self-training round.
5. Write-up and the decision on 125M.

While step 1 runs, the GPU finishes the 5M verified self-training rounds.
