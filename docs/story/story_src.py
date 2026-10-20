"""Source of the research story (docs/story/story.json) and the test list (docs/story/tests.json, from the extraction
in merged.json). Every number below is copied from the records (LOG.md, run_ledger.py, the result files, the design
pages); nothing was re-run for this page.   python docs/story/story_src.py MERGED_JSON && python docs/story/build.py"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

PHASES = {   # extraction phase -> chapter shown in the sidebar
    "September research: compact state models to Semantic V1": "1 · Where it started: September research",
    "Compiler v0 and repository review": "2 · A compiler between language and the model",
    "Encoding ladders and tables (Version 1 of the compiler line)": "3 · Encoding ladders and count tables",
    "Talking-bot test and Design v2 (memory + copy head)": "4 · The bot that could not answer",
    "Local GPU, scale run and reasoning baseline": "5 · A real GPU and the first hard truths",
    "Session 6: seeds, larger core, SQuAD, data fixes": "6 · Seeds, a larger core, reading the transcripts",
    "Session 7: what to train the core on, and the stage-2 recipe": "7 · What to train on, and a gentler recipe",
    "The 50M large-scale test": "8 · The 50M test",
    "Answer speed, modern block and depth": "9 · Shape, precision and speed",
    "Phase 1": "9 · Shape, precision and speed",
    "Phase 2": "10 · Training recipe and retrieval",
    "Phase 3": "11 · Data integrity and the 100M core",
    "Phase 4": "12 · The 100M model and the side labs",
    "Multi-turn structure and own retrieval": "13 · Skills and conversation",
    "Skills v4, K2 v2 and grounded scorecard (session 34)": "13 · Skills and conversation",
    "Chat model v2: arms A/B/C/D (session 35)": "14 · Chat model v2: four versions",
    "Long-context review and fixes": "15 · Long context: the window is not the answer",
    "Overnight stress, stall and limits": "15 · Long context: the window is not the answer",
    "v2.1 and the three audits": "16 · Audits and a clean start",
    "chat22 clean package": "16 · Audits and a clean start",
}

EXTRA_TESTS = [   # added after the extraction, from this session's records (LOG.md 16 Oct, results/long_grid_*, chat_tests_*)
    {"id": "V21-FULL", "title": "Chat model v2.1 on two seeds: full grid and guard", "date": "16 Oct 2026 (done 11:00)",
     "phase": "16 · Audits and a clean start",
     "question": "Do v2.1's long-context gains repeat on a second seed and on the full GPU grid, and does every other front hold?",
     "settings": "5M (256 x 6), window 2,048, 30,000 steps of 16, lr 1e-4 / heads 1e-3, --stage v21 --aux 0.5 --point 0.5, bf16; seeds 1 and 2 (74 and 77 min)",
     "data": "v2.1 mix: Q&A 25, skills v4 20, conversations 15, recall v2 10, buried evidence 15, rehearsal 15 (the 15 MB rehearsal sample found later by the dataset audit)",
     "results": ["grid (GPU, 8 per cell), window-only share: two-step 0.02 (A) -> 0.85 / 0.84, never told 0.01 -> 0.96 / 0.99, document 0.50 -> 0.71 / 0.69",
                 "overall at 512 / 1,024 codes: A 0.56 / 0.50 -> v2.1 0.93 / 0.93 (both seeds)",
                 "one fact, distractor, correction: 2-6 points below A on both seeds",
                 "clean text: wiki 1.340 / 1.340 bits/byte (A 1.354)",
                 "guard held: chat probe 27 / 26 (A 26), rewriting 32, held-out 118 / 120 (A 119)",
                 "guard lost: skills v1 67 / 70 (A 72), clarify 4 / 3 (A 7), conversation probe v2 108 / 109 (A 110), wrong refusals 8 / 7 (A 5)",
                 "recall from session memory at 2/5/10/20/40 turns: 15/12/10/9/5 and 15/11/7/6/4 of 15 (A 15/15/14/15/14)",
                 "read at 40 turns: the right fact retrieved in every case, the reply 'I don't have that in my memory' in 3 of 6"],
     "verdict": "rejected (gains carried into v2.2)",
     "lesson": "Training for 'never told' must be balanced against cases where the fact is only in memory; otherwise the model learns that a recalled fact means 'I don't know'. v2.2's recall data was checked for this balance (223 answer vs 108 abstain per 1,500 windows).",
     "sources": ["LOG.md 16 Oct", "results/long_grid_chat5m_v21_s1.json, _s2.json", "results/chat_tests_chat5m_v21_s1.json, _s2.json", "compiler/experiments/gate_check.py output"]},
]

STORY = {
    "eyebrow": "Text Research Model · September to 16 October 2026",
    "title": "From compact semantics to a model that remembers",
    "lede": ("This is the story of the research so far: what we set out to find, what each test showed, where we were wrong, "
             "and where it has led. Every test has its own page in the list on the left, with its settings, data, results and "
             "lesson, taken from the records. Nothing on this page was re-run to write it."),
    "chapters": [
        {"id": "start", "title": "1. Where it started", "dates": "September 2026",
         "paras": [
             "The project began as Adam Barnett's <b>Compact Semantic Reasoning</b> research. The question was practical and set by local hardware: could text be represented more compactly, with less redundant input, so that a <b>smaller, faster and more accurate</b> model could be trained on one machine?",
             "Over phases V6 to 30 the work separated three ideas that had been treated as one: storage compression, input representation, and <b>semantic compilation</b> (turning language into explicit entities, roles, state changes and queries). Precision kept following semantic compilation. The decisive moment was Phase 13: a learned byte-level parser reached only 36.5% exact parses, while a deterministic controlled-language parser reached 100%, with 99.1% of answers right. Much of the earlier accuracy had come from correct parsing. <b>Getting from text to structure was the real problem.</b>",
             "<b>Semantic V1</b> (released 15 September) was a parser that turned controlled accounting and ordering text into typed event rows. It was 100% on its regression set, but one of three seeds fell to 76% on unfamiliar wording. The records say plainly that it was a narrow specialist, not a language model. Its lessons carried forward: structure makes a small model precise; the boundary with real language is the main risk; a right answer can hide wrong internal reasoning; and exact tools belong beside the network. The September design also stated the idea that runs through everything after it: <b>keep knowledge outside the weights</b>, so new facts are usable at once without retraining."],
         "tests": ["SEP-P13", "SEP-V1", "SEP-P18-30"]},
        {"id": "compiler", "title": "2. A compiler between language and the model", "dates": "early October; review 8 October",
         "paras": [
             "In early October the work moved to a <b>deterministic text compiler</b> (Compiler v0): one frozen dictionary, words with their leading space, rare words spelled in pieces and then bytes, lossless round trips, and a C++ encoder byte-identical to the Python reference. On the first measurements plain v0 matched an in-house BPE tokenizer (1.865 vs 1.866 bits per byte on a count-model proxy), and encoded at 112.5 MB/s.",
             "On 9 October Adam set the direction: the compiler is the go-between in both directions. Training text is compiled to codes, the model learns on codes, and the user's words are compiled the same way at chat time. The hope was a much smaller, much faster model, built from a small compute core plus large lookup tables."],
         "tests": ["CV0", "REVIEW", "CALIB"]},
        {"id": "ladders", "title": "3. Encoding ladders and count tables", "dates": "9 October",
         "paras": [
             "The first ladders answered the encoding question quickly. Compiled text beat raw bytes by 8% at equal compute and talked back 2.5 times faster, but <b>matched a standard BPE tokenizer to within 1%</b>. Phrase codes made sequences 18% shorter but predictions 2.5-3.4% worse. The lesson, written into the Design v2 notes: <b>a lossless encoder cannot add knowledge</b>; it only changes speed, stability and what structure is exposed.",
             "The real gain came from what stable codes made possible: <b>exact count tables</b> mixed into the output. Done with leave-one-out counting, four minutes of training beat every ten-minute run (1.881 bits per byte with 1.9M learned parameters), and the hybrid was 9.5-10.9% better than BPE in a one-time side-by-side. The gain held at 200 MB (9.3%), lived in the middle and rare vocabulary, and let the model learn by reading: adding a book's counts improved a related novel with no retraining."],
         "tests": ["L1", "L2d", "L2d-LOO", "SBS", "SCALE-CHECK", "LBR"]},
        {"id": "talk", "title": "4. The bot that could not answer", "dates": "9 October",
         "paras": [
             "Then the first talking-bot test changed the course. Trained on question-and-answer text, the tables predicted 19% better than a plain model, yet <b>neither answered any of ten questions</b>. The tables gave the same opening to every question, because after 'Assistant:' their context is identical. Predicting text and answering a question are different skills.",
             "Design v2 split answering into two jobs: <b>find</b> the right stored exchange (a memory keyed on the question answered 97% of what it held, including 200 pairs added after training), and <b>copy</b> from it. A small network did not learn to copy on its own (0.13 F1). A learned <b>copy head</b>, which points at the retrieved answer and carries it into the reply, lifted stored questions to 0.48-0.62 F1. This became the design the project still follows: a small network with its knowledge in a memory it can read and copy from."],
         "tests": ["TALK", "DV2-S1", "DV2-S2"]},
        {"id": "gpu", "title": "5. A real GPU and the first hard truths", "dates": "9-10 October",
         "paras": [
             "The RTX 3090 reproduced the CPU numbers to four decimals and ran 5.5 times faster. The scale run then delivered an uncomfortable result: the exact-table gain <b>reverses with core size</b> (+10% at 0.8M, level at 4.7M, 1% worse at 19M, and 9-12% worse on Wikipedia). Count tables are a crutch for a core too small to learn the statistics. Memory and the copy head, not tables, were the mechanism to scale.",
             "Two more lessons came the same day. On GSM8K grade-school maths, copying numbers from the question and an exact calculator worked, but the model chose the wrong operations: <b>setting up the steps, not computing them, was the problem</b>. And <b>language first, task second</b>: adapting a core pretrained on general text beat every Q&A-only core in every group (stored 0.80 F1)."],
         "tests": ["GPU", "SCALE", "GSM-2-3", "R1b", "BUG-LEAK-COPY"]},
        {"id": "s6", "title": "6. Seeds, a larger core, and reading the transcripts", "dates": "10 October",
         "paras": [
             "A second seed confirmed the adapted core. A 19M core predicted better but copied worse at equal time, and was no better at equal steps. SQuAD reading at 5M failed (1% exact match). The replies rambled past the answer, and passages were memorised.",
             "The most important event of the day was not a score. <b>Reading the chat-probe transcripts found a parsing bug</b>: 3,450 Alpaca answers (6.6%) with more than one paragraph had been cut to their first paragraph. No metric had shown it. From then on, reading the model's actual replies became part of every evaluation."],
         "tests": ["R1b-s2", "R1d", "R2", "BUG-PARSE", "R1f"]},
        {"id": "s7", "title": "7. What to train on, and a gentler recipe", "dates": "10 October",
         "paras": [
             "A deep look at the core's training text was sobering: 82% was 1800s fiction and the rest raw wiki markup, with no conversation at all. Rebuilding it with clean modern text, real conversations, evidence pairs and worked maths cut prediction error on conversation from 2.02 to 1.30 bits per byte and on maths from 2.23 to 0.91. <b>Better data beat more data.</b>",
             "But a harsh second stage erased the difference. Through S2a and S2b the project found an industry-style recipe: a low core learning rate, a separate head rate, rehearsal of plain text, loss only on the answer, an end-of-answer code (S2a-2's answers never stopped), and whole exchanges in the window. <b>S2b-512 reached stored 0.89 F1 / 56% exact</b>, the result in the README's headline (network alone 13%, with memory and copying 89%). Verified self-training on maths moved 2.5% to 4.0%, slowly."],
         "tests": ["DATA-AUDIT", "D1-D3r", "S2a-2", "S2b-512", "S2b-s2", "STaR", "E1"]},
        {"id": "50m", "title": "8. The 50M test", "dates": "11 October",
         "paras": [
             "A 55M-parameter core asked whether the recipe scales. It did for language: Wikipedia error fell 28%, conversation 36%, and <b>SQuAD reading went from 1% to 28% exact match</b>, the first large gain from size on a reasoning-type task. <b>Maths did not move</b>: 1.5% on new problems, with well-formed steps and wrong operations. Size did not teach the model to set up the steps.",
             "Later checks qualified the 50M numbers: an encoder bug meant it trained on only the first 2.15 GB of its mix, and its evidence-gain figures were inflated by a leak found on 13 October."],
         "tests": ["50M-core", "50M-SQuAD", "50M-GSM", "GATE-50M", "SPEED"]},
        {"id": "shape", "title": "9. Shape, precision and speed", "dates": "11-12 October",
         "paras": [
             "The next questions were about cost. A modern block (rotary positions, RMS normalisation, gated feed-forward) lowered text error 1-7% but did not improve answers at 5M. At equal parameters, <b>3 layers matched 6 and 11-16 were worse</b>. Training in <b>bf16 equalled fp32</b> (0.1% apart) at 2.2 times the speed; <b>8-bit weights were free</b> and 4-bit cost 13%. Most answer time had been spent outside the network, and a <b>CUDA-graph answer step</b> made replies 3.6-9.3 times faster with byte-identical output.",
             "Adam made three decisions on 12 October: <b>maths is paused</b> (to get its own line later), <b>precision is bf16 plus int8 only</b>, and the focus is <b>text, Q&A and conversation</b>."],
         "tests": ["S16-MODERN", "DEPTH", "PREC", "GRAPH", "S18-MQA", "COST"]},
        {"id": "recipe", "title": "10. Training recipe and retrieval", "dates": "12-13 October",
         "paras": [
             "Overnight sessions tested how to train and how to retrieve. <b>Muon with warmup-stable-decay</b> gave the best text core (-0.9%), confirmed on a second seed. <b>Dense retrieval</b> (a MiniLM sentence key beside our word key) lifted retrieved-answer F1 by 11-13%.",
             "Session 23 found the most useful change of the series: <b>train the copy head on reworded questions</b>, so it learns to copy when the stored question only nearly matches. With dense retrieval in training this became the 5M chat recipe (chat probe 24 and 23 of 32 on two seeds). Reranking by the core itself hurt (20/32). Combining all the lessons at once cancelled out (22/32), an early warning that wins do not simply add up at 5M."],
         "tests": ["RECIPE", "RETR", "S22", "COPY-EXACT", "S23", "S24", "S25", "S26"]},
        {"id": "integrity", "title": "11. Data integrity and the 100M core", "dates": "13 October",
         "paras": [
             "Preparing the 100M core exposed two problems that changed how results are trusted. The native encoder had been <b>silently stopping at 2 GiB</b>: the 50M core lost its last 184 MB, and the first 100M start would have lost 1.06 GB. And held-out evaluation text was <b>leaking into training</b> through evidence pairs (20-23% of held-out Wikipedia in the large mixes). A <b>clean evaluation set</b> with zero leaked paragraphs replaced it. The clean evidence gain at 5M was 8.7-9.2%, not the 33.5% reported earlier.",
             "The 100M core ran for 18 hours on 2B codes. Twice it slowed in a GPU memory squeeze, fixed by gradient accumulation with an identical update."],
         "tests": ["ENC2GB", "LEAK", "CLEAN", "SQUEEZE-1", "S27-CORE", "S29-1"]},
        {"id": "100m", "title": "12. The 100M model and the side labs", "dates": "14 October",
         "paras": [
             "The <b>100M Q&A model became the best model so far</b>: chat probe 25/32 at bf16 and int8, 11/32 without memory, and new questions 0.358 F1. It answered in 0.28 s and 55 J. A correction the same evening noted that it used MiniLM dense retrieval and reworded-copy training, the full adopted recipe. Feeding retrieval confidence to the trust head (session 28) and the DLI course lessons (session 30) were both rejected.",
             "Side labs ran on the CPU beside the queue. A KV cache made CPU answers 2-28 times faster. A <b>100-question skills test</b> of invented facts showed that many failures had the right fact retrieved but not copied (baseline 33/100). <b>K2</b>, our own learned retrieval key, was built because Adam directed that retrieval should become our own. Adam also set a rule: <b>no model above 100M without his go-ahead</b>. A 50-question common-sense rating, done by reading the answers (by Claude, not Adam), showed the 5M network alone writes fluent-looking nonsense: 9.9 of 100, against 47.0 for the 100M."],
         "tests": ["S27-QA", "S28", "S30", "INF-A", "SKILLS-1", "SKILLS-23", "K2-SETUP", "QA-RULES", "RATE50", "MT-SETUP"]},
        {"id": "skills", "title": "13. Skills and conversation", "dates": "15 October",
         "paras": [
             "Six times more skill data fixed most of the over-refusal. The <b>100,000-example skills arm</b> took the skills test from 33 to 58/100 (65 with three candidates), with held-out 112-128/200 against 75-77, in seven minutes of GPU training. Skills v4 taught answering in the form asked (short, yes/no, lists, 'your' for 'my'): rewriting went from 4 to 29/40.",
             "The multi-turn study settled how to train for conversation: <b>one mixed stage</b> beat sequential stages (history gain 11.2% vs 9.3% on both seeds). K2 came close to MiniLM on finding reworded questions (78.9% vs 80.9%) but cost 2-4 chat-probe points, so MiniLM stayed as the outside reference."],
         "tests": ["SKILLS-100K", "S34-SKV4", "S31", "S33", "S32", "S34-K2V2", "S34-GROUNDED", "INF-B"]},
        {"id": "v2", "title": "14. Chat model v2: four versions", "dates": "15 October",
         "paras": [
             "Chat model v2 was built from every result that held up: the new 5M core, the Q&A recipe, skills v4, real conversations, long recall, a 2,048-code window and one mixed stage. Four versions tested modern reasoning methods one at a time. <b>A</b> was the baseline (skills 72/100, held-out 119/200, rewriting 32/40, chat probe 26/32; the old 5M chat model scored 33, 75-77, 4 and 23). <b>B</b> added intermediate targets (which fact, which words) and copied better but lost 'I don't know' and clarify. <b>C</b> added a verifier (+3 skills at 3.4 times the time). <b>D</b> added a round of self-training and was level with B.",
             "The first live conversation found the gaps the scores hid: refusing greetings and statements, missing corrections by name, a verifier that preferred short fragments."],
         "tests": ["S35-AUDIT", "S35-A", "S35-B", "S35-C", "S35-D", "S35-SUITE", "S35-LIVE"]},
        {"id": "long", "title": "15. Long context: the window is not the answer", "dates": "15-16 October",
         "paras": [
             "Adam asked how far the context could be pushed, compared with million-token models. A review of a long-context training guide against our results found eight gaps. One was a plain bug: <b>session memory skipped facts that had just left the window</b>, so recall after 10 turns was 0/15. The fix took it to <b>14-15 of 15 at every distance</b> from 2 to 40 turns. Another was a design gap: evidence had only ever appeared in a short slot at the start of the window, so the model had never learned to find a fact buried in long input.",
             "The overnight stress test showed what a raw window really does. The 5M ran up to 262,144 codes (55 s before the first word) and the 100M to 65,536. The hidden fact was <b>never found at any length, even inside the trained 2,048</b>, and quality collapsed past the trained window (5.0 bits per code inside, about 10 at 100k). The test also froze for 9.5 hours when GPU memory spilled into system RAM. That led to a stall check every 30 minutes and Adam's limits: <b>at most 3/4 of the GPU, long context capped at 65,536 codes, a 32k target for the 5M first</b>.",
             "Adam's conclusion set the direction: the goal is not just a longer window. <b>Every turn goes into a temporary conversation memory</b>, looked up with the stored facts, and used if it applies. A fixed window fed by memory can cover a conversation of any length. v2.1 trained for exactly this, with buried evidence, harder recall and a 'nothing here' target. On an early check (seed 1, CPU, 6 per cell), 'I don't know' when never told went from 2% to <b>91%</b>, two-step questions from 0% to <b>76%</b>, and a fact inside a pasted document from 4% to <b>59%</b>."],
         "tests": ["CTX-TRIAL", "LONG-REVIEW", "MEMFIX", "GRID-A-QUICK", "V21-BUILD", "S36", "STRESS-TOOLS", "MEM-ENGINE", "V21-S1"]},
        {"id": "audits", "title": "16. Audits and a clean start", "dates": "16 October",
         "paras": [
             "Before building on v2.1, Adam asked for a deep audit of the whole script. Three read-only reviews (the code path, the tests, the datasets) found problems that two weeks of added options had hidden. <b>Every chat model since 15 October had trained on 15 MB of plain text instead of 300 MB</b>, a small audit sample reused by mistake. The new memory layout would have taught the model to prefer old values over corrections. Some answer-time rules were never seen in training. And several tests could be passed the wrong way, for example by always saying 'I don't know'.",
             "The answer was a <b>clean package, chat22</b>: about 1,500 lines in ten modules, with one window builder shared by training and answering. It has two memory pools (stored facts and this conversation), a trainer that stops on a held-out plateau and keeps the best checkpoint, rebuilt data (555 MB of whole documents, a long-document pool, deduplicated conversations), story training, and one test suite with a guard that fails on any lost front. It was proved before training: loaded with arm A's weights it computes what the old code computes (within 0.00003), replies are identical with and without the history cache, and the plateau stop works."],
         "tests": ["V21-FULL", "V22-LAYOUT", "AUDIT-DATA", "AUDIT-CODE", "AUDIT-TESTS", "GATE", "CHAT22-DATA", "CHAT22-EQ", "CHAT22-AUDIT", "CHAT22-ENGINE", "CHAT22-TRAINER"]},
        {"id": "now", "title": "Where we are now, and what comes next", "dates": "16 October, morning",
         "paras": [
             "<b>v2.1 is measured on both seeds.</b> Its long-context gains repeat (two-step questions 2% to 85%, never told 1% to 96-99%, a fact in a document 50% to 70%). But it lost ground elsewhere, and in one way that matters for the goal: recalling a fact from memory once the conversation is long (5 of 15 at 40 turns, against 14 for arm A). It retrieved the right fact and said it did not have it. It is not adopted. Its gains and that lesson go into v2.2, whose recall training was checked for the balance v2.1 lacked.",
             "<b>v2.2 is training now from the clean package</b> on two seeds, and stops when its held-out score levels off. The suite then compares it with v2.1 and arm A on identical cases, including memory at 100, 1,000 and 10,000 turns. After that come the session-memory test to 5,000 turns, the CPU stress test and the disk speed test.",
             "The road ahead is set: if v2.2 holds every front and improves long context and memory, the window grows in measured steps, from 8k to 32k on the 5M, then 32k on the 100M. Only then come larger models (with Adam's go-ahead) and more data. The goal is unchanged since September: <b>a small, fast model that keeps its knowledge outside its weights</b>, now with a memory that lets it hold a conversation of any length."],
         "tests": []},
    ],
    "lessons_intro": ("The findings that held up across the tests, grouped by theme. Each links to the tests that taught it."),
    "lessons": [
        {"title": "Memory and copying", "items": [
            {"text": "Predicting text and answering questions are different skills. Answering needs <b>find, then copy</b>: a memory keyed on the question and a learned copy head (network alone 13% F1, with memory and copying 89%).", "tests": ["TALK", "DV2-S2", "S2b-512"]},
            {"text": "Training the copy head on <b>reworded</b> questions teaches it to copy when the match is only close; the model had learned to copy only on exact matches.", "tests": ["COPY-EXACT", "S23", "S25"]},
            {"text": "Session memory must skip <b>only</b> the turns actually in the window; assuming a fixed number dropped recall to 0/15.", "tests": ["MEMFIX"]},
            {"text": "Stored facts and conversation memory must be ranked <b>separately</b>; in one ranking, short general facts beat the conversation piece that holds the answer.", "tests": ["MEM-ENGINE"]}]},
        {"title": "Data", "items": [
            {"text": "<b>Better data beat more data</b>: clean, modern, whole documents with conversation cut conversation error from 2.02 to 1.30 bits per byte at the same size and steps.", "tests": ["DATA-AUDIT", "D1-D3r"]},
            {"text": "Read the data and the replies, not only the scores: the parse bug, the 2 GiB encoder bug, the evaluation leak and the 15 MB rehearsal bug were all invisible in the metrics.", "tests": ["BUG-PARSE", "ENC2GB", "LEAK", "AUDIT-DATA"]},
            {"text": "Six times more skill data fixed most over-refusal; teaching the form of an answer (short, yes/no, list) needs its own examples.", "tests": ["SKILLS-100K", "S34-SKV4"]}]},
        {"title": "Training recipe", "items": [
            {"text": "<b>Language first, task second</b>: adapt a pretrained core, with a low core learning rate, rehearsal, answer-only loss and an end-of-answer code.", "tests": ["R1b", "S2a-2", "S2b-512"]},
            {"text": "Muon + warmup-stable-decay is the best text recipe; bf16 training is free; int8 weights are free; 4-bit is not.", "tests": ["RECIPE", "PREC"]},
            {"text": "Lessons do not simply add up at 5M: combined wins cancelled, and several 'modern' additions (intermediate targets, self-training, course lessons) were level or traded one skill for another.", "tests": ["S24", "S30", "S35-B", "S35-D"]},
            {"text": "One mixed training stage beats sequential stages for conversation.", "tests": ["S31", "S33"]},
            {"text": "Training should stop on held-out measurements, not a fixed step count (the v2.1 skills loss of 0.01 suggests memorised templates).", "tests": ["CHAT22-TRAINER"]}]},
        {"title": "Size and shape", "items": [
            {"text": "Count tables help only a core too small to learn the statistics; their gain reverses with size.", "tests": ["SCALE"]},
            {"text": "Size helps reading (SQuAD 1% to 28%), not choosing maths steps (1.5-4.0% at every size and depth).", "tests": ["50M-SQuAD", "50M-GSM", "DEPTH"]},
            {"text": "The 100M Q&A model is the best so far (25/32), and the 5M network alone has almost no common sense: memory carries the 5M.", "tests": ["S27-QA", "RATE50"]}]},
        {"title": "Long context and memory", "items": [
            {"text": "A window is room, not use: the model never found a fact by attention alone, even inside its trained window, until it was trained to (buried evidence, v2.1).", "tests": ["S36", "V21-S1"]},
            {"text": "Positions past the trained window are noise; grow the window by training in steps, capped at 65,536 codes.", "tests": ["S36", "LONG-REVIEW"]},
            {"text": "A fixed window fed by memory can cover any length of conversation; judge it by a flat accuracy line from 100 to 10,000 turns.", "tests": ["V21-S1", "CHAT22-ENGINE"]}]},
        {"title": "Measuring honestly", "items": [
            {"text": "Two seeds before building on a win; equal steps and equal text in every comparison.", "tests": ["R1b-s2", "S25", "S33"]},
            {"text": "Tests can be gamed: always abstaining, naming both values, template phrasing. Use a single scorer with a wrong-value veto, paired cases and intervals.", "tests": ["AUDIT-TESTS"]},
            {"text": "Watch every long run: a frozen job looks alive to a simple check; 9.5 hours were lost before a 30-minute stall check existed.", "tests": ["S36", "STRESS-TOOLS"]}]},
    ],
}


def main() -> int:
    merged = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    tests = []
    for t in merged["tests"]:
        t = dict(t); t["phase"] = PHASES.get(t["phase"], t["phase"])
        tests.append(t)
    tests += EXTRA_TESTS
    missing = [i for c in STORY["chapters"] for i in c["tests"] if i not in {t["id"] for t in tests}]
    missing += [i for g in STORY["lessons"] for it in g["items"] for i in it["tests"] if i not in {t["id"] for t in tests}]
    if missing:
        print("chapter/lesson links to unknown tests:", sorted(set(missing)))
    (HERE / "tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "story.json").write_text(json.dumps(STORY, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tests)} tests, {len(STORY['chapters'])} chapters, phases: {len(set(t['phase'] for t in tests))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
