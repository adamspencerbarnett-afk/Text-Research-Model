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
    ("R4", "GSM8K/SVAMP: retrieval by solution structure (operator sequence key), template copying", "reasoning by analogy over worked examples", "paused 12 October: math gets its own research line later"),
    ("R5", "number channel in the compiler (number unit → code + value)", "removes digit-run failures", "paused 12 October (math line)"),
    ("R6", "ARC / OpenBookQA / CommonsenseQA, one-letter answers with retrieved facts", "accuracy vs chance (25%)", "needs --format"),
    ("R7", "bAbI deduction / counting / path tasks", "V1's ground; expect the design to shine", "needs download mirror"),
    ("R8", "full check loop: verify, refine, restart, 'unknown'", "precision of 'unknown'", "paused 12 October (math line)"),
    ("R9", "19M core on the winning design", "decides the 'smaller at equal quality' claim for reasoning", "after R4"),
    ("T1", "scale: domain feature for the trust head; per-domain tables (REASONING_PLAN 2.4)", "can the tables be made harmless at 19M?", "queued, low priority"),
    ("DEPTH", "depth at equal parameters and steps: 384x3, 192x11, 160x16 (128x24 dropped 12 October: 11 layers already lost to 6 at equal steps)", "does depth buy language or step choice?", "done: 3 = 6 > 11 > 16 layers on language; GSM8K within noise at every depth; 256x6 stays"),
    ("MQA", "the modern 256x6 core through the Alpaca Q&A stage, chat probe and GSM8K stage (session 18)", "does the modern block's lower error reach answers?", "done: no; Q&A and math level, probe 17/32 vs 20-21, slower; legacy stays at 5M, re-test at 50M"),
    ("PREC", "precision (session 19): a 5M core trained in fp32 at equal steps against the bf16 core; the best 5M and the 50M Q&A models with weights rounded to bf16, int8 and int4 (precision.py), scored on the Q&A groups, the chat probe and the core report card", "does bf16 training cost anything, and how few bits can the weights keep without losing answers?", "done: bf16 training = fp32 (0.1%), 2.2x faster; int8 weights free at 5M and 50M; int4 -13% stored F1 at 5M, -8% new at 50M"),
    ("GRAPH", "CUDA-graph answer step and real low-bit kernels (bitsandbytes)", "can the small model answer fast?", "done: 3.6-9.3x faster, replies identical; low-bit kernels slower at this size"),
    ("RECIPE", "session 20: Muon, warmup-stable-decay, and both, on the 5M core at equal steps; report card, Q&A stage, probe (graph)", "does the modern training recipe raise text and answer quality?", "done: Muon + WSD best text core (-0.9%), WSD -0.8% at no cost; answers unchanged at 5M; bf16 Q&A control and seed 2 in session 22"),
    ("RETR", "retrieval upgrade: dense learned question keys trained contrastively in-domain, top-k retrieval with reranking, top-k exchanges sampled in training", "reworded questions (0.21-0.30 F1) are the weakest group: is it retrieval?", "done: dense keys adopted (retrieved-answer F1 +11-13%; reworded 0.259 -> 0.276, new 0.290 -> 0.299; probe 22/32); core reranking and bad evidence next"),
    ("S22", "session 22: bf16 Q&A-stage control on the baseline core; bad-evidence training (10% nothing, 10% a random wrong exchange, --distractor-p); second seed of the Muon + WSD core", "was session 20's answer dip bf16 or the recipes; does the model learn when not to copy; is the Muon + WSD text gain real?", "done: not bf16 (control 0.891); bad evidence: network alone +20%, stored best at 0.901 / 62%; Muon + WSD confirmed on seed 2"),
    ("S23", "session 23: reranking by the core at answer time on the dense-trained model; dense retrieval in training plus --perturb-self 0.3 (copy under rewording)", "can the core rerank its own candidates; does training on reworded self windows make it copy from matches worded differently?", "running (started 05:56, 13 October)"),
    ("S24", "session 24: dense retrieval in training + copy under rewording (--perturb-self 0.3) + bad evidence (10% none, 10% wrong), one Q&A stage; probes plain, int8 and with core reranking", "do the two training lessons combine: copy from imperfect matches, refuse wrong ones?", "queued after session 23"),
    ("RERANK", "rerank the top 10 with the core itself (evidence gain, zero new parameters) against a cross-encoder; 30% bad-evidence windows (distractor, none, conflicting) in the Q&A stage; gate trained on outcomes (HYBRID_DESIGN 3.3, 3.6, section 11)", "does the model stop copying from wrong matches, and can the core rerank its own candidates?", "after session 21 (session 22)"),
    ("DRAFT", "copy-as-draft decoding inside the CUDA graph: verify k copied codes per replay (HYBRID_DESIGN 3.7)", "how fast can copied answers get, replies identical?", "session 22, code only"),
    ("PASSAGE", "passage memory tier (200-500 word chunks, SQuAD-style), top-3 in the window with an evidence cache of unrotated keys; retrieval key from the core's own state trained in-domain; multi-turn probe; session memory in the chat (HYBRID_DESIGN 3.2, 3.3, 3.7, 3.8)", "reading from retrieved passages at small size; conversation state; retrieval without an outside model", "session 23"),
    ("DATA", "modern dialogue data for the Q&A stage (SmolTalk, Apache 2.0; Tulu 3 SFT, ODC-BY; licences checked) and multi-turn windows with turn history", "conversation and instruction quality beyond Alpaca", "session 24"),
    ("REASON", "program tier with exact execution (state, count, compare, date, unit, find/filter/join; synthetic verified traces), thinking codes (THINK..END, adaptive) and self-requested SEARCH (two hops), trained by verified practice; support check and 'I don't know' (HYBRID_DESIGN 3.8, 3.9, 3.11)", "can a small model reason in steps over retrieved evidence and programs, and know when it cannot?", "session 25"),
    ("DPO", "preference tuning (direct preference optimisation) on better/worse answer pairs", "tone, staying on topic, not rambling (the probe's chat section)", "session 26"),
    ("MEMLAYER", "product-key memory layer inside the core (trainable form of the count tables); consolidation and nightly practice", "capacity without compute; should not reverse with size", "session 27"),
    ("SIZE", "100M, 200M, 300M shallow-wide cores with the winning recipe, trained across nights (WSD), int8 release; 16k and 32k dictionaries at 100M", "at what size does reading become good, and does a larger dictionary make answers faster per byte?", "session 28"),
    ("BASELINE", "a 0.5B instruct model with the same retriever and the same questions, both at their best precision; accuracy and joules per answer side by side", "does the design hold against a conventional RAG system?", "session 29, last"),
    ("COST", "record cost beside accuracy for every run: actual training seconds, steps/s, peak GPU memory, answer seconds per question, GPU power (joules per answer); ledger columns for train seconds and answer seconds", "is the model lighter and faster, in measured numbers?", "done: cost.py meters power, energy, memory and steps/s for every run; ledger cost table; probes record s and J per answer"),
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
        "Depth sweep 2 of 4, deep-narrow 192x11 (6.56M total; same mix, window and 42,538 steps; 3,487 s against 2,606-2,672 s for the shallower shapes): held-out bits/byte 0.5-0.9% worse than 256x6 on every source except question pairs (overall 1.1990 vs 1.1915; wiki 1.305 vs 1.296; math 0.647 vs 0.644; books 1.754 vs 1.734), generation 245 B/s. At equal steps, eleven narrow layers learn language slightly slower than six and cost 30% more time per step; nothing so far says depth buys fluency at this scale. Report card and GSM8K stage follow.",
        "Depth sweep 2 of 4, 192x11 report card and GSM8K stage: evidence gain 35.6% (384x3 34.7%, 256x6 33.5%; deeper cores use shown evidence a little better), reword recall@1 0.495 (0.504, 0.509). GSM8K (8,000 steps, calculator): new problems 2.0% (4/198; 256x6 2.5%, 384x3 0%), stored 44.5% final-number exact (44.5%, 41.5%), F1 0.889, added 45.5%, calculator-confirmed 533 (516, 348), validation 0.651. Across 3, 6 and 11 layers every math measure is within noise: depth changes nothing on GSM8K at 5M, in either direction. One more point (160x16) closes the sweep.",
        "Depth sweep 3 of 3, 160x16 core (6.33M; same mix, window and 42,538 steps; 3,936 s, the slowest shape, 1.5x the 256x6 time): held-out overall 1.2032 (256x6 1.1915, 384x3 1.1905, 192x11 1.1990), worse on every source except question pairs and math, which are flat; generation 113 B/s (256x6 254, 384x3 330); evidence gain 35.7% (the one measure that rises with depth: 33.5 -> 34.7 -> 35.6 -> 35.7), reword recall@1 0.484 (falls with depth: 0.509 -> 0.504 -> 0.495 -> 0.484). Language at equal steps: 3 = 6 > 11 > 16 layers. 128x24 dropped; the GSM8K stage closes the sweep.",
        "Modern block through the Alpaca Q&A stage (S2b recipe, 15,565 steps, same data and settings as S2b-512): validation 0.911 (legacy seed 1 0.929, seed 2 0.874), but every answer group inside the seed spread of the legacy core: stored 0.887 F1 / 56% exact (legacy 0.887-0.890 / 56-63%), added 0.866 / 57% (0.862-0.881 / 57-58%), reworded 0.227 (0.259-0.302), new 0.289 (0.271-0.290), network alone 0.245 (0.226-0.243). The 1-7% lower language error does not reach the answers at 5M, where copying from memory sets the score; reworded is the one group below both legacy seeds and needs a second seed before it counts. First metered training run: 1,469 s (25% more than legacy), 379 W average, 557 kJ, peak 3.9 GB, 36 J per step. Probe and GSM8K stage follow.",
        "Modern core chat probe: 17/32 (legacy 20 and 21/32): knowledge 4 -> 3, held-out 4 -> 3, taught 5 -> 4, the rest equal; 1.01 s and 56 J per answer on an idle GPU (the modern block's generation is half the legacy speed). With the Q&A groups level, the modern block at 5M is no better on answers, slower and a little worse in conversation; legacy stays the 5M default, and the block is re-tested at 50M, where the network alone does more of the work.",
        "Session 22 A, the bf16 control: the baseline core through the unchanged S2b stage in bf16 gives stored 0.891 F1 / 56%, added 0.858, new 0.286, network alone 0.229, probe 21/32 (int8 21), level with both fp32 seeds (stored 0.887-0.890). bf16 in the Q&A stage is not the cause of session 20's dip: the Muon and WSD cores themselves are slightly worse starting points for the Q&A stage at 5M (stored 0.872-0.876), even though they are better text models. At 5M, better stage-1 text does not reach answers and can cost a little copying; the recipe question returns at the larger cores, where the network does more of the work.",
        "Session 22 B, bad-evidence training (10% of windows with nothing retrieved, 10% with a random wrong exchange; otherwise A's recipe): network alone 0.229 -> 0.274 F1 (+20%, the best network-alone score at 5M; previous best 0.245), stored 0.891 -> 0.901 F1 and 56% -> 62% exact (best stored at 5M), added 0.858 -> 0.868, new 0.286 -> 0.287, reworded 0.242 -> 0.215 (small group), probe 20/32 (int8 20). Training on misleading evidence makes the network answer better on its own and copy correct evidence more precisely; reworded questions fall, consistent with the exact-match copying finding (wrong evidence teaches the model not to copy from imperfect matches). Adopted, to be combined with copy-under-rewording training (session 23 B) so both lessons are taught together.",
        "Session 22 C, Muon + WSD core seed 2: held-out overall 1.1826 bits/byte (seed 1 1.1810; baseline 1.1915), books 1.700 (1.700), evidence gain 35.0% (35.0%), reword recall 0.497 (0.490). The -0.8 to -0.9% text gain is confirmed on two seeds (spread 0.0016). Adopted for stage 1 of the 100-300M cores.",
        "Training-signal finding (13 October, smoke test of core reranking): with dense retrieval in training the right stored pair is retrieved for 'Which graphics card is in the research computer?', yet the model still does not copy its answer. In stage 2 the copy pointer only ever sees a correct answer to copy in self windows, where the stored question is word-identical; retrieved neighbours are other exchanges whose answers are wrong to copy. The model has learned 'copy only on an exact match'. Session 23 trains with the right answer under a reworded question (--perturb-self, never used in a run before).",
        "Dense retrieval (session 21; literal key + 2 x all-MiniLM cosine). Retrieval itself improves: the retrieved answer's own F1 against the reference rises 0.231 -> 0.261 on reworded questions (+13%) and 0.196 -> 0.217 on new ones (+11%). R1, answer time only, best 5M model unchanged (bf16 weights): reworded 0.214 -> 0.239, new 0.280 -> 0.285, stored and added unchanged (0.876, 0.860), probe 21 -> 22/32. R2, dense neighbours in the training loop (S2b recipe, bf16 Q&A stage), against S2b-512 seed 1: reworded 0.259 -> 0.276, new 0.290 -> 0.299 (the best new-question score at 5M), stored 0.889 (0.890), added 0.865 (0.862), network alone 0.221 (0.226), probe 22/32, int8 probe 21. The answer gains are small and the reworded group is small (inside the two-seed spread 0.259-0.302); the retrieval gain is clean. On reworded questions the model now scores above the answer it retrieves (0.276 vs 0.261), so it adds to copying. Adopted: dense keys for retrieval at answer time and in training.",
        "R2 also bears on session 20's dip: it trained its Q&A stage in bf16 and kept stored answers at 0.889 (fp32 seed 1: 0.890), so bf16 in the Q&A stage is unlikely to be the cause of the recipe arms' 0.872-0.876; session 22's control on the baseline core settles it.",
        "Training recipe on the 5M core (session 20; same mix, dictionary, window and 42,538 steps; bf16; fused AdamW): held-out overall bits/byte AdamW + cosine 1.1915 (baseline), Muon + cosine 1.1878 (-0.3%), AdamW + warmup-stable-decay 1.1825 (-0.8%), Muon + WSD 1.1810 (-0.9%; books 1.700 vs 1.734 -2.0%, math -3.7%, question pairs -5.5%, OpenAssistant -1.2%, UltraChat -1.0%). Muon raises evidence gain (35.1% and 35.0% vs 33.5%) and lowers reword recall@1 (0.48-0.49 vs 0.51); WSD alone keeps recall (0.515). Cost: Muon 11% more time per step (2,972 / 2,906 s vs 2,634 s fused AdamW), about 1.1-1.2 MJ per core. WSD wins on text at no cost and makes runs extendable; Muon + WSD is the best text core. One seed each.",
        "The recipe gains did not reach answers at 5M (Q&A stage unchanged, AdamW + cosine, now in bf16): stored 0.872-0.876 F1 (baseline seeds 0.887-0.890), added 0.849-0.858 (0.862-0.881), new 0.275-0.298 (0.271-0.290), network alone 0.183-0.236 (0.226-0.243), probe 18-20/32 (20-21), int8 probe equal or one lower. Same pattern as the modern block: about 1% better text at stage 1 does not survive the Q&A stage at 5M, where copying from memory sets the score. A confound must be removed before reading it as 'no effect': every recipe arm trained its Q&A stage in bf16 for the first time, and all three sit about 1.5 points below both fp32 baseline seeds on stored answers. Session 22 runs the baseline core through a bf16 Q&A stage as the control, and a second seed of the Muon + WSD core.",
        "Clean answer cost on an idle GPU with the CUDA-graph step and bf16 weights (session 20 part A, chat probe, 32 answers): 5M 0.081 s and 10.2 J per answer at 125 W (eager fp32 earlier: 0.45 s, 29 J), probe 21/32 unchanged; 50M 0.147 s and 24.5 J per answer at 167 W, probe 21/32 unchanged. The graph step cuts time per answer 5.5x and energy per answer 2.8x at 5M with the same replies.",
        "Training precision, 5M core at equal data, window and 42,538 steps: fp32 against bf16 autocast (D3s512). Held-out overall 1.1901 vs 1.1915 bits/byte (0.1%), every source within 0.1-0.4% except question pairs (0.774 vs 0.804, the fp32 core better) and books (1.745 vs 1.734, bf16 better); evidence gain 33.6% vs 33.5%; reword recall@1 0.498 vs 0.509. bf16 trained 2.2x faster (2,672 vs 5,868 s; fp32 run 411 W, 2.4 MJ, 6.6 GB peak). bf16 training costs nothing measurable: confirmed on our own data, closing the question.",
        "Weight precision at answer time, 50M Q&A model (weights rounded; same eval as S2b-50m): bf16 stored 0.883 F1 / 58% exact, added 0.878 / 60%, reworded 0.272, new 0.329, network alone 0.271, probe 21/32; int8 0.883 / 58%, 0.878 / 60%, 0.279, 0.332, 0.271, probe 21/32 (free); int4 0.882 / 58%, 0.874 / 57%, 0.264, 0.305, 0.246, probe 21/32. int4 costs far less at 50M than at 5M (new -8%, network alone -9%, stored -0.1%, against -13% stored at 5M): larger models tolerate low bits better. Policy unchanged: bf16 and int8 only. The 50M probe energy readings (48-79 J per answer) were taken while speed tests shared the GPU and are upper bounds.",
        "Weight precision at answer time, best 5M Q&A model (weights rounded, compute unchanged; precision.py): bf16 identical to fp32 on every measure (probe 21/32, stored 0.876 F1 / 56%, added 0.860 / 57%, new 0.280, network alone 0.244, Wikipedia 1.6027 vs 1.6026 bits/byte). int8 within noise (probe 20/32, stored 0.875, added 0.858, reworded 0.235, new 0.277, alone 0.238, Wikipedia 1.6034): int8 weights are free at 5M. int4 (groups of 32) costs real accuracy: stored 0.763 F1 / 40% exact (from 0.876 / 56%), added 0.753 / 40%, alone 0.200, Wikipedia 1.658 (+3.5%), evidence gain 32.5%, probe 19/32. Floor at 5M: 8 bits. The 50M model follows.",
        "CUDA-graph answer step (fast_decode.py): the eager loop launched about 300 small kernels per generated code; the step is now recorded once with fixed-size buffers (attention cache, window, copy-head keys) and replayed with one launch. Replies identical to the eager loop on all 64 probe answers for four models (5M legacy, 5M modern, 50M, 50M with the similarity feature). Speed with session 19 sharing the GPU: 5M 11.7 -> 1.75 ms per code (6.7x), 50M 14.6 -> 4.1 (3.6x), 5M modern 21.8 -> 2.4 (9.3x). Two exact fixes in the eager loop as well (count tables looked up for the last position only; the stop marker checked on the last codes instead of decoding the whole reply), worth 2%.",
        "Real low-bit kernels (bitsandbytes 0.50.2, installed 12 October) on the 5M model are slower, not faster: int8 (LLM.int8, threshold 6) 42.6 ms per code eager vs 11.4 fp32, NF4 16.2, bf16 autocast 12.3; and they change most replies (bf16 16/32 identical, int8 7/32, NF4 0/32), since they quantise activations or compute in lower precision, unlike the weight rounding above. bitsandbytes int8 cannot be captured in a CUDA graph. At 5-50M parameters the GPU is launch-bound, not bandwidth-bound, so low-bit kernels cannot help; they matter for the 0.5B baseline. fp32 + graph is the answer path.",
        "Precision policy from 12 October: every new test trains in bf16 (cores with --amp since the scale run; the Q&A stage with the new m2_qa --amp) and answers with bf16 weights, with an int8 check of each new Q&A model; no further fp32 or int4 runs (int4 lost 13% of stored-answer F1 at 5M). The CUDA-graph answer step computes in fp32 with the bf16-rounded weights, because bf16 kernels were slower at this size. Earlier fp32 Q&A stages remain the baselines; bf16 reproduced fp32 to four decimals on the core stage.",
        "Dense question keys (dense_keys.py; all-MiniLM-L6-v2, pretrained sentence encoder, Apache 2.0, used in retrieval only) on the report card's 2,000 QQP rewordings: recall@1 literal 75.1%, dense 80.4%, literal + 2 x dense 80.9% (top 10: 96.7% -> 98.6%). In the chat, 'Which graphics card is in the research computer?' now retrieves the taught 'What GPU does the research computer use?' (literal similarity 0.40), but the 5M network trained on literal neighbours does not copy from a match that shares few words; session 21 measures it at answer time only and with dense neighbours in training.",
        "Correction to the 'train longer' plan: the 5M core already sees about 200 codes per parameter (42,538 steps over a 92M-code mix, about 15 passes), so longer training is a 50M lever (about 20 per parameter), not a 5M one; at 5M the limit is the size of the mix.",
        "Modern core GSM8K stage (8,000 steps, calculator; 750 s, 290 kJ): new problems 2.0% (4/198), stored 44.5% final-number exact, F1 0.897, added 42.5%, calculator-confirmed 444, validation 0.632. Level with the depth-sweep cores at the same 8,000 steps (stored 44.5%, confirmed 516-533, new 0-2%). Correction to the depth comparison: the 256x6 legacy GSM8K reference (S2b-gsm-512, stored 41.5%, confirmed 348) ran 15,565 steps, not 8,000, so it is not an equal-steps point; the conclusion stands (every 8,000-step core is within noise), but the 256x6 row is not like for like. Session 18 closed: the modern block is level on math and Q&A at 5M, below on the probe, and slower.",
        "Depth sweep closed, 160x16 GSM8K stage: new problems 0% (0/198), stored 44.5% final-number exact, F1 0.899, added 43.5%, calculator-confirmed 526, validation 0.629 (the best validation of the four shapes, as the evidence gain was). Final table, 3 / 6 / 11 / 16 layers at equal parameters and steps: new problems 0 / 2.5 / 2.0 / 0% (all noise), stored 44.5 / 41.5 / 44.5 / 44.5%, confirmed 516 / 348 / 533 / 526, language overall 1.1905 / 1.1915 / 1.1990 / 1.2032, generation 330 / 254 / 245 / 113 B/s. Depth does not move math at 5M and costs language and speed past six layers; the shape question is settled for this size (256x6 stays, 384x3 is the fast alternative), and the math lever remains template retrieval (R4).",
        "Cost instrumentation (cost.py): every training run now records GPU board power (nvidia-smi, 2 s samples), kilojoules, peak allocator memory and steps per second in training.cost; the chat probe records seconds and joules per answer. Readings are only valid when nothing else is on the GPU: the first probe reading (qa_best_5M_512: 0.48 s and 67 J per answer at 142 W) was taken while the 160x16 GSM8K stage was training and is an upper bound. Older runs keep their train_s only.",
        "Memory gate at 5M (qa_best_5M_512, same model, probe with and without --min-sim 0.5): 21/32 without the gate, 19/32 with it (reworded 3 -> 2, taught 5 -> 4): the 5M network alone is too weak to be handed the questions the gate takes away from memory, so the gate is a 50M-only setting (21 -> 24/32 there). run_model.py defaults: gate off, window 512, modern block, bf16; menu option 3 opens the best model in git.",
        "Depth sweep 1 of 4, GSM8K stage on the 384x3 core (8,000 steps, S2b recipe, calculator): new problems 0% (0/198) against 2.5% (5/198) for the 256x6 core; stored problems 44.5% final-number exact (256x6: 41.5%), F1 0.90 (0.88), added-after-training 45% (42%), calculator-confirmed answers 516 (348), validation 0.644 (0.660). Three layers copy and set up steps at least as well as six, and the only loss is on new problems, where both are at the noise floor (0-5 of 198). Depth below six does not remove step choice; the deep shapes decide whether depth above six adds it.",
        "Depth sweep 1 of 4, shallow-wide 384x3 (8.66M total, 5.32M core; same mix, dictionary, 512 window, 42,538 steps, 2,606 s): held-out bits/byte within 0.3% of the 256x6 legacy core on every source (overall 1.1905 vs 1.1915; worked math 0.638 vs 0.644; question pairs 0.829 vs 0.804, the one loss), evidence gain 34.7% vs 33.5%, state recall@1 0.504 vs 0.509; generation 330 vs 254 B/s because there are half the layers. Three layers at width 384 match six at width 256 on language at equal steps; the modern 256x6 block is still the best core (1.1724). The GSM8K stage decides whether depth matters for step choice.",
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


def cost_rows() -> list[dict]:
    """Cost beside accuracy: every result that recorded training time (and, from 12 October, power and memory),
    plus the chat probes' seconds and joules per answer."""
    out, probes = [], {}
    for path in glob.glob(str(ROOT / "results" / "probe_*.json")):
        j = json.load(open(path, encoding="utf-8"))
        c = j.get("cost") or {}
        probes[Path(j.get("model", "")).stem] = {"ans_s": c.get("s_per_unit"), "ans_j": c.get("j_per_unit"),
                                                 "probe": sum(v["ok"] for v in j["summary"].values())}
    for path in sorted(glob.glob(str(ROOT / "results" / "*.json")), key=os.path.getmtime):
        try:
            j = json.load(open(path, encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(j, dict):
            continue
        items = []
        if isinstance(j.get("training"), dict) and "train_s" in j["training"]:
            items.append((Path(path).stem, "Q&A" if "groups" in j else "stage", j["training"], j.get("setup", {})))
        for key, r in (j.get("runs") or {}).items():
            if isinstance(r, dict) and isinstance(r.get("training"), dict) and "train_s" in r["training"]:
                items.append((f"{Path(path).stem} {key}", "core", r["training"], {}))
        for name, kind, t, setup in items:
            c = t.get("cost") or {}
            pr = probes.get(Path(setup.get("save") or "").stem, {})
            out.append({"run": name, "kind": kind, "steps": t.get("steps"), "train_s": None if t.get("train_s") is None else int(round(t["train_s"])),
                        "steps_s": round(t["steps"] / t["train_s"], 2) if t.get("train_s") and t.get("steps") else None,
                        "w": c.get("gpu_avg_w"), "kj": c.get("gpu_kj"), "mem": c.get("peak_mem_mb"),
                        "ans_s": pr.get("ans_s"), "ans_j": pr.get("ans_j"), "probe": pr.get("probe")})
    return out


def table(headers: list[tuple[str, str]], rows: list[dict]) -> str:
    head = "| " + " | ".join(h for _, h in headers) + " |\n|" + "|".join(" ---: " if k not in ("run", "data", "core", "flags", "device", "stop", "tables", "file", "kind") else " --- " for k, _ in headers) + "|\n"
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
          "\n## Cost beside accuracy\n",
          "*Training seconds for every run that recorded them; GPU watts, kilojoules and peak memory from 12 October (cost.py; valid only when the GPU was otherwise idle); "
          "seconds and joules per answer from the chat probe of the saved model, with its probe score out of 32.*\n\n",
          table([("run", "run"), ("kind", "kind"), ("steps", "steps"), ("train_s", "train s"), ("steps_s", "steps/s"), ("w", "GPU W"), ("kj", "kJ"), ("mem", "peak MB"),
                 ("ans_s", "s/answer"), ("ans_j", "J/answer"), ("probe", "probe /32")], cost_rows()),
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
