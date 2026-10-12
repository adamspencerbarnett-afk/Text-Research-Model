"""The run ledger: every Q&A-design run and every scale run, settings beside results, in one
Markdown file, regenerated from the JSON results so it is never out of date.

    python compiler/experiments/run_ledger.py            # writes docs/RUN_LEDGER.md

The queue of planned runs and the findings per session are the two lists at the top; edit
them here. Everything else is read from results/.
"""
from __future__ import annotations

import glob
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# ----------------------------------------------------------------------------- planned runs
QUEUE = [
    ("R1b", "Alpaca, 4.7M core adapted from the scale-run checkpoint (--init, --dict), window 256, 600 s GPU", "does a core that knows language raise the no-memory floor and the copy scores?", "done: yes, best run so far"),
    ("R1c", "same model, greedy decoding, evaluation only (--budget 0 --greedy)", "exact-answer rate without sampling drift", "done: +5-10 points exact"),
    ("R3b", "GSM8K, 4.7M core trained from scratch, copy from question + executor + check, 1200 s GPU", "set-up quality against depth (6 layers)", "done: 0/198 unseen"),
    ("R3c", "GSM8K, 4.7M core adapted from the scale-run checkpoint", "set-up quality with a core that knows language", "done: 1/198 unseen; stored 43-45%"),
    ("R1b-s2", "R1b with seed 2, greedy", "confirm the winner", "done: confirmed (seen 0.88 / 52% exact)"),
    ("R1d", "Alpaca, 19M core (scale run checkpoint) adapted, greedy", "does the floor keep rising with core size once the core knows language?", "done: better prediction, weaker copying (stored 8% exact vs ~50%)"),
    ("R0b", "Alpaca, 0.8M: retrieval similarity as a trust-head input; three window kinds in training", "learn when to copy; stop the prediction score worsening", "queued"),
    ("R0c", "Alpaca, 0.8M: top-3 retrieved exchanges in the window", "paraphrase group above the single-retrieval ceiling (0.27)", "queued"),
    ("R2b", "SQuAD with loss on answers and replies cut at the span", "copy-from-passage", "done: 5M 0.07 F1 / 1% EM; 50M 0.42 F1 / 28% EM"),
    ("D1-D3r", "5M stage-1 data tests: clean modern text, + conversation, + evidence/math/rewordings, aligned vs random windows", "which data makes the core best at its three jobs", "done: big core gains, not visible after stage 2"),
    ("S2a", "stage 2 from the D3r core at lr 3e-4 and 1e-4, with and without 30% stage-1 rehearsal", "does a gentler stage 2 keep the core's gains?", "done: S2a-3 (1e-4 + rehearsal) best of the series; S2b recipe adds head rate, answer-only loss, end code"),
    ("S2b", "full industry-style Q&A recipe at 256 and 512 windows", "the training-recipe fix and the cut-off fix", "done: S2b-512 best model (stored 0.89/56%, added 0.86/57%)"),
    ("S2b-s2", "S2b-512 with seed 2", "confirm the winner", "done: confirmed"),
    ("K1", "contrastive question key: a projection head trained on Quora reworded pairs", "a learned memory key that beats the literal one", "done: +3.5 points in domain; no Q&A gain (out-of-domain key)"),
    ("E1", "evidence builder: relaxed thresholds, multiple use, longer windows", "enough evidence pairs for the 50M mix (got 20.6k, needed 110k)", "done: 148,500 pairs; D3e core +1.3 points evidence gain; Q&A stage running"),
    ("STaR", "verified self-training on GSM8K, two rounds", "does the model learn to choose operations from its own checked solutions?", "done: 2.5% -> 3.5% -> 4.0%; confirmations +22%"),
    ("50M", "the large-scale test (docs/PLAN_50M.md)", "does the recipe scale?", "Q&A (60% of steps): new 0.33, network alone 0.27, probe alone 11/32; gate test and full run queued"),
    ("R2", "SQuAD 1.1: copy a span from a given passage, exact match", "the cleanest test of the copy head", "done (failed: rambling, memorised passages)"),
    ("R4", "GSM8K/SVAMP: retrieval by solution structure (operator sequence key), template copying", "reasoning by analogy over worked examples", "next after R2 (design written)"),
    ("R5", "number channel in the compiler (number unit → code + value)", "removes digit-run failures", "needs compiler change"),
    ("R6", "ARC / OpenBookQA / CommonsenseQA, one-letter answers with retrieved facts", "accuracy vs chance (25%)", "needs --format"),
    ("R7", "bAbI deduction / counting / path tasks", "V1's ground; expect the design to shine", "needs download mirror"),
    ("R8", "full check loop: verify, refine, restart, 'unknown'", "precision of 'unknown'", "after R4"),
    ("R9", "19M core on the winning design", "decides the 'smaller at equal quality' claim for reasoning", "after R4"),
    ("T1", "scale: domain feature for the trust head; per-domain tables (REASONING_PLAN 2.4)", "can the tables be made harmless at 19M?", "queued, low priority"),
]

# ----------------------------------------------------------------------------- findings per session
FINDINGS = [
    ("9 Oct, session 1 (GPU set-up)", [
        "CPU and GPU reproduce the same numbers to four decimals; bf16 reproduces them too; the GPU is 2x faster at 20M+ with the ragged tables.",
        "Windows CRLF in the corpus scripts changed every book's hash; fixed, corpora rebuilt from the manifests.",
    ]),
    ("9 Oct, session 2 (copy head)", [
        "The learned copy head is the missing mechanism: network answers on stored questions 0.13 -> 0.48 / 0.62 (two seeds); no change where memory holds nothing, so the gain is entirely from carrying what memory returns.",
        "An evidence bigram over the retrieved span and key folding add nothing; an answer-only copy span is no better than the full exchange.",
        "A target leak appeared when the copy span included the current position (pointer could point at itself); fixed by restricting the pointer to strictly earlier positions. Earlier copy-head results are unaffected.",
    ]),
    ("9 Oct, session 3 (scale run)", [
        "The exact-table gain reverses with core size: +10% at 0.8M, 0 at 4.7M, -1% at 19M (equal time), and 9-12% worse on Wikipedia. The tables are a crutch for a small core; the trust head does not reject them off-domain.",
        "The tables stay as the store that learns by reading; they are not the mechanism for 'lighter and faster at equal quality'. The memory and the copy head are.",
    ]),
    ("11 Oct, session 8 (50M, speed, depth)", [
        "50M math (GSM8K stage on the P50 core, 8,000 steps, calculator): new problems 1.5% (3/198), no better than 5M (2.5-4.0%); stored 47% / F1 0.90; calculator-confirmed answers 524 (5M: 348-424); validation 0.66 (5M: 1.03). The steps are well formed and the operations wrong ('27 in 5 years: 27 * 5 = 135'): ten times the parameters at 20 codes each did not teach step choice. Practice with the verifier and depth are the open levers; queued.",
        "50M verified self-training, round 1 (generation: 2,658 of 7,473 training problems solved at least once with the own answer hidden, 36% against 11% at 5M; 4,469 checked solutions; fine-tune 8,000 steps): new problems 1.5% -> 2.5% (3 -> 5 of 198, inside noise), stored 47% -> 57%, confirmed answers 524 -> 572, but added-after-training 46.5% -> 28%. The high training-problem solve rate is retrieval of near-duplicate templates (GSM8K reuses structures), which does not transfer to held-out problems: template retrieval by solution structure (R4) is the lever, and full fine-tunes on self-solutions cost copying of new facts (an argument for adapters).",
        "50M reading comprehension (SQuAD, passage in the question, answer-only loss, end code, 10,000 steps): held-out passages F1 0.418 / 28.1% exact match (5M: 0.068 / 1.0%); stored 0.593 / 51%. The first large gain from size on a reasoning-type task: the 50M core reads a passage and selects the answering span ('lightning', 'more ATP than NADPH'); the 5M core returned short but wrong spans. Fine-tuned 100M-class models reach 80-90 F1; the gap is size and training amount, not mechanism.",
        "Modern block (RoPE + RMSNorm + SwiGLU) against the best 5M core, same mix, dictionary, 512 window and 42,538 steps, 2% fewer parameters: lower bits/byte on every held-out set (Wikipedia 1.296 -> 1.279, web 1.329 -> 1.312, worked math 0.644 -> 0.625, UltraChat 0.998 -> 0.983, question pairs 0.804 -> 0.746), evidence gain 33.5% -> 33.9%. Cost in this first implementation: training 2,672 -> 3,927 s (rotation tables recomputed every step, three matrix products in the gated block) and generation 254 -> 144 B/s; both are engineering. Adopted for the next core once the speed is recovered.",
        "Idle-GPU answer speed after the device-side fix: 32 probe answers in 18.3 s at 50M and 16.7 s at 5M (replies identical with and without the KV cache; cache 0.96-0.97x). Size barely matters at batch 1, so about 8 ms per code of per-step overhead remains outside the network (Python repeat blocking, decoding the reply each step, a GPU sync per step, the table and copy-head mixing). The compute advantage of the small core cannot show in wall-clock time until that loop is rebuilt.",
        "Answer speed: a profile of a 50M answer showed about 9 ms per code in the network and about 70 ms around it (the window rebuilt on the GPU every step, masks moved every step, a sync per step); fixed so the window grows by one code on the device. Absolute 50M timings must be taken on an idle GPU (session 15 shared the card); queued.",
        "Memory gate: with the similarity as a trust-head input (S2c) the model gates itself and the hard threshold changes nothing (21/32 at 0.4, 0.45, 0.5); on the earlier S2b model the 0.5 gate gave 24/32. Both configurations are kept: S2c best on the Q&A groups, S2b + gate best for chat.",
    ]),
    ("10 Oct, session 7 (what to train the core on)", [
        "Data deep dive: 82% of stage-1 text was 1800s fiction (14.4 archaic words per 10k, 0.2 modern), the rest raw wiki markup (half the bytes), no conversation, no evidence windows. Manual review found more: conversations 5x longer than the 256-code window, FineWeb paragraph breaks invisible, MetaMath in LaTeX, Gutenberg front matter. All rewritten (prepare_core_mix.py).",
        "Better data transforms the core itself (5M, equal steps, D0 -> best new): conversation fluency 2.02 -> 1.30 bits/byte (UltraChat) and 2.12 -> 1.49 (OpenAssistant), Wikipedia 2.06 -> 1.68, web 2.02 -> 1.66, math 2.23 -> 0.91; evidence gain 22.7% -> 31-32%; reworded-question retrieval by the core's own state 0.38 -> 0.50 at rank 1, 0.59 -> 0.78 in the top 10. Conversation data gave the largest single jump. Old books got worse (expected: D0 trained 82% on them).",
        "None of it shows in the Alpaca stage-2 scores: every core lands at stored 0.62-0.69 F1, reworded about 0.2, no-memory 0.09-0.14, within seed noise. Cause: stage 2 is a full fine-tune at lr 1e-3 for 15,565 steps on 19 MB, which overwrites stage-1 differences, and Alpaca F1 mostly measures copying. Stage 2 must be gentler (lower lr, rehearsal of stage-1 data) to keep what stage 1 built.",
        "Paragraph-aligned windows: mixed. Random windows give slightly better fluency (Wikipedia 1.68 vs 1.73, math 0.91 vs 1.17); aligned slightly better evidence gain (32.4% vs 31.2%) and reword retrieval (0.50 vs 0.49). Use random windows for prose and aligned windows only for structured units. The evidence test was incomplete: the builder found 20.6k pairs, so evidence was 3% of D3 instead of 18%.",
        "Learned memory key: the core's state alone finds 50% of reworded questions at rank 1 against 74% for the literal key; combined, 75%. A useful learned key needs training on rewordings (contrastive), not the language model's states as they are.",
        "S2a-1 (Q&A stage at lr 3e-4 instead of 1e-3, D3r core): validation 1.039 -> 0.971 and the no-memory floor doubles (0.10 -> 0.20 F1, equal to the 19M core), but exact copying drops (stored 30% -> 10%): the fresh copy and trust heads learn too slowly at the core's gentle rate. Fix adopted: two learning rates (--head-lr), core gentle, new heads full.",
        "S2a-2 (lr 3e-4 + 30% rehearsal of stage-1 text): best validation yet (0.925) and no-memory floor 0.22, but exact copying near zero (stored 3%, added 0%). The transcript shows why: answers do not stop ('William Shakespeare' then ': William Shockpeare : William Wordsworth ...'). Nothing in stage-2 training ever marked the end of an answer (the window after it is padding, outside the loss), and rehearsal text teaches that more text always follows. Fix adopted: an end-of-answer code trained as the last code of every exchange (--eos), the industry's end-of-turn token; generation stops at it.",
        "S2a-3 (lr 1e-4 + rehearsal): the best balance of the series. Validation 0.896 (from 1.039 at 1e-3), no-memory floor 0.23 (from 0.10), and copying back near the old level (stored 0.65 F1 / 27% exact, added 0.63 / 20%); probe 22/32 with held-out instructions 4/5. The 3e-4 + rehearsal run (3% exact) is the outlier, not the trend; the stage-2 recipe moves to 1e-4 for the core with rehearsal, plus the head rate, answer-only loss and the end code (S2b).",
        "S2b-256, the full industry-style Q&A recipe (core lr 1e-4 + heads 1e-3, cosine, answer-only loss, end-of-answer code, rehearsal; D3r core; 200 questions per group): best Q&A model of the research in every group. Stored 0.82 F1 / 60% exact, added after training 0.77 / 51%, reworded 0.28, new 0.29, no memory 0.24 (old best: 0.79-0.88 / 47-52%, 0.25-0.34, 0.26, 0.15-0.20, on 60-question groups). Answers now stop when done ('The capital of Japan is Tokyo.'). Remaining failures in the probe: synonyms (graphics card / GPU), arithmetic, small talk, and wrong retrievals (hairstyle -> sneakers).",
        "Core D3s512 (5M, 512-code window, document separators, random windows, 42,538 steps = 2x the codes of D3r in 2,672 s): better on running prose, network alone (report card) Wikipedia 1.683 -> 1.603, web 1.664 -> 1.589, evidence gain 31.2% -> 33.5%; with tables, Wikipedia 1.338 -> 1.296. Part of the gain is the doubled text seen, not only the window; the Q&A test (S2b-512) decides whether whole exchanges matter.",
        "S2b-512 (same recipe on the 512-window core; whole exchanges up to 261 codes and whole retrieved exchanges up to 250): best model. Stored 0.89 F1 / 56% exact, added 0.86 / 57% (S2b-256: 0.82 / 60%, 0.77 / 51%); reworded 0.26, new 0.29, no memory 0.23, unchanged. Not cutting answers improves answering from memory by 7-9 F1 points; reworded and new questions depend on the core's reading and on retrieval, not on the window. Second seed queued.",
        "K1 in use (S2b-512 re-scored with literal + learned key, no retraining): no gain (reworded 0.259 -> 0.260, new 0.290 -> 0.281, probe 20 -> 19). Retrieval changes in both directions (gravity and night-sky questions find better matches; 'Italian dishes' finds 'Italian actors'). Two causes: the key was trained on Quora questions, not instructions; and the reworded group's ceiling is low, since the retrieved answer itself scores only 0.23 F1 against the reference (held-out questions are rarely true rewordings), so the model at 0.26 is already above it. The probe's reworded and taught sections are the better test of rewording. Key kept as an option, off by default; retrain on in-domain pairs before the next try.",
        "S2b-512 confirmed on seed 2: stored 0.887 / 63%, added 0.881 / 58%, reworded 0.30, new 0.27, no memory 0.24, probe 21/32 (seed 1: 0.890 / 56%, 0.862 / 57%, 0.26, 0.29, 0.23, 20/32). Saved as models/qa_best_5M_512.pt with its dictionary beside it (.cv0d) and its stage-1 core (models/core_5M_512_stage1_D3s.pt); chat with it through run_model.py.",
        "GSM8K under the new recipe (S2b-gsm, 512 core with MetaMath in stage 1, copy from question, calculator): new problems 2.5% (5/198) against 1.0% with the old recipe; stored problems 42% (F1 0.88 against 0.80); the calculator confirmed 348 stated answers and replaced 1,216. Steps now read as solutions ('The price of the magazines is 11/8*72 = 99'), but the choice of operations is still mostly wrong. Next: verified self-training (STaR).",
        "E1 evidence builder: 148,500 pairs at similarity >= 0.10 (from 20,600 at 0.15); core D3e512 (18% evidence instead of 3%, otherwise D3s512): evidence gain 33.5% -> 34.8%, bits/byte with related evidence 1.203 -> 1.176, math 0.910 -> 0.884, other fluency equal or slightly better, reword retrieval unchanged. A modest real gain; the Q&A stage on it (S2b-512e) decides whether it reaches answers.",
        "STaR round 1 generation: with its own stored solution hidden, the GSM model solved 797 of 7,473 training problems (10.7%) at least once in 4 tries; 972 checked solutions kept (about 3% of 29,880 tries).",
        "STaR round 1 (fine-tuned 8,000 steps on GSM8K + its 972 checked self-solutions): new problems 3.5% (7/198) against 2.5% (5/198), inside noise; stored 40% (from 42%); calculator-confirmed answers 348 -> 377. First steps are visibly better set up ('27-5 = 22 years old; in 8 years 22+8 = 30'). Round 2 decides whether it compounds.",
        "S2b-512e (Q&A stage on the 18%-evidence core): Q&A groups within seed noise of S2b-512 (stored 0.878 / 57%, added 0.840 / 51%, new 0.298), network alone a little lower (0.195 vs 0.23-0.24); chat probe 23/32, chat 3/3. Read in the transcript: asked for an email asking for a day off, it adapted the retrieved 'raise' email to 'a day off' where S2b-512 copied it unchanged; asked what to cook, it gave steps. Evidence share kept at 18% for 50M: no measurable cost, small gains in the core and in adapting retrieved answers.",
        "STaR round 2 (the round-1 model solved 2,395 training problems; 3,408 checked solutions): new problems 2.5% -> 3.5% -> 4.0% over base, round 1, round 2 (5 -> 7 -> 8 of 198, each step inside noise); calculator-confirmed answers 348 -> 377 -> 424 (+22%), the steadier signal; stored problems 48%. Steps increasingly right ('11/8*$72 = 99. The profit is 99-$72 = 27', then one wrong extra step). Verified self-training works at 5M, slowly; next on the 50M core.",
        "Parallel evidence builder: 4.0M paragraphs indexed in 463 s, 10 workers at about 80 paragraphs/s each (about 21 minutes for all), a partner for about a quarter of paragraphs at the 900-character limit; the serial builder needed about 4.5 hours for 150k pairs.",
        "50M core (P50 mix, 640w x 10L, 1,024-code window, 30,500 steps, 3.47 GB seen, 8,564 s): network-alone report card against the best 5M cores, Wikipedia 1.595 -> 1.148 (-28%), web 1.585 -> 1.165 (-27%), conversation 1.296 -> 0.831 (-36%) and 1.476 -> 1.020, worked math 0.884 -> 0.402 (-55%), old held-out books 2.03 -> 1.50, bits/byte with related evidence 1.176 -> 0.809; relative evidence gain 34.8% -> 31.6% (the core predicts much better without evidence too); reworded-question retrieval by its own state 50.5% -> 58.4% at rank 1, 78% -> 84% in the top 10. The 5M core saw more text (4.9 GB), so the gain is size and window, not data volume. Q&A stage running.",
        "50M Q&A stage (S2b recipe, stopped by the 2-hour cap at 9,247 of 15,565 steps): stored 0.893 / 60%, added 0.878 / 60% (5M best 0.89 / 56-63%, 0.86-0.88), reworded 0.272, new 0.332 (5M 0.27-0.29), network alone 0.271 (5M 0.23-0.24), policy 0.312; chat probe 21/32 with memory, 11/32 for the network alone (5M: 4-5). The 50M network answers on its own ('12 + 30 = 42', '5 - 2 = 3 apples left', a proper day-off email, photosynthesis), and weak retrievals now hurt it ('12 plus 30 is 100' with memory): the trust head still trusts weak matches. Next: memory only for strong matches (probe, no retraining), and a full-length Q&A stage with the similarity as a trust-head input and 20% of windows with nothing retrieved.",
        "Memory gate on the 50M model (no retraining; memory used only for exact hits or nearest matches with similarity >= 0.5): chat probe 21 -> 24/32, the best yet (held-out 5/5, chat 2/3, arithmetic 1/3). Weak matches now go to the network alone ('5 - 2 = 3 apples left', a proper day-off email). One loss: a reworded taught fact at similarity 0.476 fell below the bar. The full-length 50M Q&A run gives the trust head the similarity so it can learn the boundary.",
        "50M full-length Q&A (S2c: 15,565 steps, similarity as a trust-head input, 20% of non-self windows with nothing retrieved): best 200-question numbers so far, stored 0.906 / 68% exact, added 0.879 / 61%, new 0.325, network alone 0.290 (short S2b run: 0.893 / 60%, 0.878, 0.332, 0.271); reworded 0.234 (38 questions, noise +-0.07). Chat probe 21/32, unchanged by the 0.5 gate (the similarity input makes the model gate itself, less sharply than the hard rule), network alone 7/32 against 11 for S2b. Near-tie overall: S2b + hard gate is the best chat configuration (24/32), S2c the best on the Q&A groups.",
        "19M with equal steps (R1f): no better than 4.7M on Q&A, and copies less (stored exact 8% against 22%). Reading comprehension (R2b): the copy head now returns short spans from the passage, but the wrong ones (unseen F1 0.07). Both point at the same limit: the core does not yet read well enough to choose what to copy.",
    ]),
    ("10 Oct, session 6 (power cut, seed check, reasoning with the adapted core)", [
        "Power cut lost R3c during scoring; the Q&A script now saves weights before scoring and run chains skip finished runs.",
        "R1b confirmed on a second seed: seen 0.79 / 0.88 F1, 47% / 52% exact; added 0.75 / 0.81, 53% / 48%; no-memory floor 0.20 / 0.15. The adapted 4.7M core is the base.",
        "GSM8K with the adapted core (R3c): 1 of 198 unseen. Three cores (0.8M scratch, 4.7M scratch, 4.7M adapted) all score 0-2%: core size and language knowledge do not fix arithmetic set-up. The replies are now well-formed step-by-step solutions whose individual steps are correct arithmetic (the executor computes them: 157 stated answers confirmed, 628 replaced) but whose choice of operations is wrong. This is the retrieval-of-procedure problem (R4), not a capacity problem.",
        "R1d (19M core adapted): best prediction yet (0.958 bits/byte against 0.98-1.05 at 4.7M) and a slightly higher no-memory floor (0.22 F1 against 0.15-0.20), but copying from memory is much weaker: stored questions 0.76 F1 / 8% exact against 0.79-0.88 / 47-52% at 4.7M. Cause: equal time is not equal training. The 19M core is 2.3x slower per step, so the same 600 s gave it 6,694 steps against 15,565, and the fresh copy head and trust head got less than half the updates. One seed; the fair re-run gives 19M the same number of steps (about 1,400 s).",
        "A small executor defect: after the executor writes a value the model can append digits to it ('17 - 17 = 0.5'); fix by closing the number after the executor writes it.",
    ]),
    ("9 Oct, session 5 (adapted core)", [
        "Adapting the scale run's 4.7M core (trained on 286 MB of books + Wikipedia) to the Q&A design for ten minutes beats every earlier run in every group: seen 0.80 F1 / 42% exact, added 0.74 / 43%, paraphrase 0.29, unseen 0.27, no-memory floor 0.19 (from 0.13). A core that knows language is the base from now on; training a core from scratch on Q&A data is dropped.",
        "Greedy decoding raises exact match by 5-10 points at no F1 cost (seen 47%, added 53%); adopted for scored answers. The 3-nat retrieval bias now lowers exact match (22% vs 42%): superseded by the copy head, dropped.",
        "GSM8K with a 4.7M core from scratch: 0 of 198 unseen; the executor only helps where the problem is stored (42-50%). The adapted core on GSM8K (R3c) is the next test.",
    ]),
    ("9 Oct, session 4 (reasoning baseline)", [
        "GSM8K unseen: 1% right at 0.8M. With copy-from-question, the executor and the answer check: 2% (inside noise). The machinery works (operands come from the question, results are computed); the 0.8M core cannot choose the operations. Set-up, not computing, is the problem.",
        "A bigger window (512, whole exchange) did not help the copy head (truncation hypothesis dropped).",
        "A 4.7M core trained from scratch on 19 MB of Q&A memorises it in minutes (validation flat from minute one, no-memory floor unchanged at 0.13). The core must learn language from the big text and be adapted to Q&A; plateau stopping with best-weights restore added to m2_qa.",
    ]),
]


def num(v, d=3):
    return "" if v is None else (f"{v:.{d}f}" if isinstance(v, float) else str(v))


def core_of(setup: dict) -> tuple[str, int]:
    w, l, h = setup.get("width", 128), setup.get("layers", 4), setup.get("heads", 4)
    params = l * 12 * w * w
    return f"{w}w {l}L {h}h", params


def flags_of(s: dict) -> str:
    on = []
    for k, label in (("copy_head", "copy"), ("evidence", "evid"), ("fold_keys", "fold"), ("copy_answer_only", "ans-only"),
                     ("copy_question", "copy-q"), ("calculator", "calc"), ("greedy", "greedy")):
        if s.get(k):
            on.append(label)
    if s.get("init"):
        on.append("init:" + Path(s["init"]).stem)
    return " ".join(on) or "none"


def qa_rows() -> list[dict]:
    rows = []
    for path in sorted(glob.glob(str(ROOT / "results" / "m2_*.json")) + glob.glob(str(ROOT / "results" / "r[0-9]*.json")),
                       key=os.path.getmtime):
        j = json.load(open(path))
        if "groups" not in j:
            continue
        s, g, t = j["setup"], j["groups"], j["training"]
        core, params = core_of(s)
        data = "GSM8K" if "gsm" in s.get("data", "") else "Alpaca"
        seen, added = g["seen_network_with_own_exchange_in_context"], g["added_network_with_added_exchange_in_context"]
        para, unseen, nomem = g["paraphrase_network_with_nearest"], g["unseen_network_with_nearest"], g["heldout_network_without_memory"]
        rows.append({
            "run": Path(path).stem.replace("m2_qa_", "").replace("m2_", ""), "data": data, "core": core, "params": f"{params/1e6:.2f}M",
            "ctx": s.get("ctx"), "span": s.get("neighbor_codes"), "batch": s.get("batch"), "budget": int(s.get("budget", 0)),
            "steps": t["steps"], "device": s.get("device", "cpu"), "seed": s.get("seed"), "self": s.get("self_context_p"),
            "flags": flags_of(s), "valid": t.get("best_valid") or (t["history"][-1]["valid_bits_per_byte"] if t["history"] else None),
            "seen_f1": seen["f1"], "seen_ex": seen.get("exact"), "seen_bias": seen.get("f1_bias"),
            "added_f1": added["f1"], "para_f1": para["f1"], "para_n": para["n"], "unseen_f1": unseen["f1"], "nomem_f1": nomem["f1"],
            "num_unseen": unseen.get("num_exact"), "num_seen": seen.get("num_exact"), "stop": t.get("stop_reason", "budget"),
        })
    return rows


def scale_rows() -> list[dict]:
    out = []
    for path in [ROOT / "results" / "scale_run.json", ROOT / "results" / "scale_check.json"]:
        if not path.exists():
            continue
        j = json.load(open(path))
        runs = j.get("runs") or j.get("configs") or {}
        for key, r in runs.items():
            if "model" not in r:
                continue
            m, t, ev = r["model"], r["training"], r["eval"]
            wiki = ev.get("ood", {}).get("enwik_head_2MiB", {}).get("bits_per_byte")
            out.append({"file": path.stem, "run": key, "core": f"{m['width']}w {m['layers']}L {m['heads']}h", "params": f"{m['params']['total']/1e6:.1f}M",
                        "tables": "exact 2,3-gram x%d" % m["ngram_tables"][0]["topk"] if m.get("ngram_tables") else "none",
                        "ctx": m["ctx"], "batch": m["batch"], "budget": int(t["budget_s"]), "steps": t["steps"], "seen_gb": t["bytes_seen"] / 1e9,
                        "amp": t.get("amp", False), "books": ev["heldout_overall"]["bits_per_byte"], "valid": ev.get("validation", {}).get("bits_per_byte"),
                        "wiki": wiki, "ood": ev.get("ood_overall", {}).get("bits_per_byte"), "gen": r["generation"]["bytes_per_s"], "stop": t.get("stop_reason", "budget")})
    return out


def table(headers: list[tuple[str, str]], rows: list[dict]) -> str:
    head = "| " + " | ".join(h for _, h in headers) + " |\n|" + "|".join(" ---: " if k not in ("run", "data", "core", "flags", "device", "stop", "tables", "file") else " --- " for k, _ in headers) + "|\n"
    body = "".join("| " + " | ".join(num(r.get(k)) for k, _ in headers) + " |\n" for r in rows)
    return head + body


def main() -> int:
    qa = qa_rows()
    sc = scale_rows()
    md = ["# Run ledger\n", "*Regenerated by `compiler/experiments/run_ledger.py` from `results/`. Settings on the left, results on the right. "
          "F1 is word overlap with the reference answer (0 to 1); `ex` is the exact-match rate; `num` is the final-number exact-match rate on GSM8K; "
          "`valid` is bits per byte on held-out exchanges with the nearest exchange in the window (lower is better). Seed noise on the F1 columns is about ±0.07 (60 sampled answers).*\n",
          "\n## Q&A design runs (M2 memory + M1 tables + core + trust head, with the flags listed)\n",
          table([("run", "run"), ("data", "data"), ("core", "core"), ("params", "core params"), ("ctx", "ctx"), ("span", "span"), ("batch", "batch"),
                 ("budget", "budget s"), ("steps", "steps"), ("device", "dev"), ("seed", "seed"), ("self", "self-ctx p"), ("flags", "flags"),
                 ("valid", "valid b/B"), ("seen_f1", "seen F1"), ("seen_ex", "seen ex"), ("seen_bias", "seen F1 +bias"), ("added_f1", "added F1"),
                 ("para_f1", "para F1"), ("para_n", "para n"), ("unseen_f1", "unseen F1"), ("nomem_f1", "no-mem F1"), ("num_seen", "num seen"), ("num_unseen", "num unseen"), ("stop", "stop")], qa),
          "\nFlags: `copy` learned copy head over the retrieved exchange; `evid` bigram table over the retrieved span; `fold` inflection folding in the key; "
          "`ans-only` copy span limited to the answer; `copy-q` copy head may point into the current question; `calc` executor writes `a op b =` results and checks the final answer; "
          "`greedy` argmax decoding; `init:` core initialised from that checkpoint. Runs 2 and 3 on GSM8K carry the target leak (fixed in 2b/3b).\n",
          "\n## Scale runs (language modelling, books + Wikipedia; traditional BPE against compiler + exact tables)\n",
          table([("file", "file"), ("run", "run"), ("core", "core"), ("params", "params"), ("tables", "tables"), ("ctx", "ctx"), ("batch", "batch"), ("budget", "budget s"),
                 ("steps", "steps"), ("seen_gb", "text seen GB"), ("amp", "bf16"), ("books", "books b/B"), ("valid", "valid b/B"), ("wiki", "wiki b/B"), ("ood", "ood b/B"), ("gen", "reply B/s"), ("stop", "stop")], sc),
          "\n## Queue of planned runs\n", "| id | run | question it answers | status |\n| --- | --- | --- | --- |\n",
          "".join(f"| {i} | {r} | {q} | {st} |\n" for i, r, q, st in QUEUE),
          "\n## Key findings by session\n"]
    for title, items in FINDINGS:
        md.append(f"\n**{title}**\n\n" + "".join(f"- {it}\n" for it in items))
    out = ROOT / "docs" / "RUN_LEDGER.md"
    out.write_bytes("".join(md).encode("utf-8"))
    print(f"wrote {out} with {len(qa)} Q&A runs and {len(sc)} scale runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
