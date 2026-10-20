"""Adds the research of 16 October (afternoon) to 17 October to the story (docs/story/tests.json and story.json), then
rebuild with  python docs/story/build.py . Every number is copied from the records (LOG.md, results/*.json, research_side/*,
docs/CHAT5M_V2_DESIGN.html#v24, data_real/README.md); nothing was re-run for this page. Safe to run twice."""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
P17 = "17 · Memory at scale, and choosing what to remember"
P18 = "18 · Any length, and what it cost"
P19 = "19 · Repair, real data and the merged model"


def T(id, title, date, phase, question, settings, data, results, verdict, lesson, sources, type="experiment"):
    return {"id": id, "type": type, "title": title, "date": date, "phase": phase, "question": question, "settings": settings,
            "data": data, "results": results, "verdict": verdict, "lesson": lesson, "sources": sources}


NEW_TESTS = [
    T("S38-MEM", "Session memory to 5,000 turns", "16 Oct 2026 (done 17:54)", P17,
      "Does conversation memory still find a fact told in turn 1 after thousands of turns, and how fast?",
      "Arm A and the 100M Q&A model; facts told once, then 10, 100, 1,000 and 5,000 filler turns; GPU",
      "real dialogue turns as filler; 10 planted facts per length",
      ["arm A recalled 10/10 at 10, 100, 1,000 and 5,000 turns",
       "5,000 turns = about 2.05M codes / 7.2M characters in session memory; lookup 18.3-18.7 ms; storing 58 s per conversation",
       "100M Q&A model: 0/10 at every length (never trained to recall what a user said); lookup 16.5-21.1 ms"],
      "adopted (memory, not the window, carries long conversations)",
      "Memory scales where the window cannot: thousands of turns cost milliseconds per lookup. A model must still be trained to use what memory returns.",
      ["LOG.md 16 Oct 18:15"]),
    T("S39-CPU", "CPU stress test: how long a window a CPU can read", "16 Oct 2026 (done ~18:35)", P17,
      "How long can the raw window grow on the CPU before waits become unusable?",
      "16 CPU threads, at most 3/4 of RAM; window doubling from 2k codes; 5M and 100M",
      "held-out text; a planted 'needle' at several depths",
      ["5M, wait / ms per code / peak RAM: 2k 0.24 s / 9 / 0.9 GB; 16k 3.8 s / 23 / 1.5 GB; 32k 12.9 s / 42 / 2.5 GB; 65k 50 s / 77 / 4.7 GB; 131k 206 s / 148 / 8.1 GB; 262k 898 s / 282 / 10.8 GB",
       "100M: 2k 1.3 s / 40 / 1.7 GB; 16k 16 s / 111 / 6.4 GB; 32k 49 s / 199 / 8.8 GB; 65k 170 s / 378 / 12.5 GB; 131k 703 s / 735 / 23.2 GB",
       "the needle was never found at any length (neither model trained on long windows)"],
      "measured (no window growth without training)",
      "A CPU can hold very long windows, but the wait grows faster than the window, and an untrained model does not use the extra room.",
      ["LOG.md 16 Oct 18:40", "results/context_stress_cpu_5M.json, _100M.json"]),
    T("S40-DISK", "Window cache on GPU, RAM or drive", "16 Oct 2026 (done 18:42)", P17,
      "Could a long window's attention cache live in RAM or on the NVMe drive instead of the GPU?",
      "per written code, GPU vs RAM vs NVMe; 5M at 8k-262k codes, 100M at 16k-65k",
      "synthetic cache of the real model shapes",
      ["5M ms per code, GPU / RAM / NVMe: 8k 12 / 14 / 43; 32k 16 / 32 / 134; 262k 92 / 234 / 988 (cache 1.6 GB)",
       "100M: 16k 16 / 61 / 330; 65k 42 / 231 / 1,304 (2.1 GB)",
       "drive: 1.39 GB/s sequential, one read at a time"],
      "adopted: the window cache stays on the GPU; the drive suits memory storage (small lookups)",
      "RAM is 2-5x and the drive 4-31x slower per code than the GPU for the working window; large knowledge belongs in memory, not in a giant window.",
      ["LOG.md 16 Oct 18:45", "results/disk_kv_bench.json"]),
    T("MEM-ONLY", "Side test: answering from memory alone", "16 Oct 2026 (18:47)", P17,
      "Can the conversation window be removed entirely, so the model reads only what memory returns?",
      "arm A and v2.1 seed 1, old engine; 12 cases per cell; window + memory vs memory only; CPU, 6 threads",
      "long-context grid cases (one fact, two-step, distractor, never told, correction) at 1,024 and 4,096 codes",
      ["memory only kept distractor and never-told cases; lost single facts (12 -> 6-7 at 1,024), corrections (12 -> 7-9) and two-step (v2.1 12 -> 1)",
       "arm A, never told: window + memory 0/12, memory only 11/12"],
      "measured (supports a memory-first input; not adopted on its own)",
      "A short, clean input of retrieved items keeps a small model grounded; a long window of filler pulls it into inventing. Memory-only needs a search that gathers every relevant piece (linked facts, newest values).",
      ["LOG.md 16 Oct 18:47", "research_side/memory_only_*.json"]),
    T("V22", "Chat model v2.2 on two seeds (clean package)", "16 Oct 2026 (seed 2 done 15:02)", P17,
      "Does the clean chat22 package keep v2.1's gains and fix its memory recall?",
      "5M, window 2,048, plateau stop (both seeds levelled at 19,000, best 22,800), lr 1e-4, batch 16; chat22 suite on identical cases",
      "v2.2 mix: Q&A 20, skills 15, conversations 13, recall 12, buried 10, document reading 7, stories 10, plain text 13 (555 MB of whole documents)",
      ["best of every chat model on clean text (wiki 1.322 / 1.321), told facts (38 / 37% vs 18-22%), never told in the window (99 / 100%), recall at 1,000 / 10,000 turns (14 / 15 and 20 / 11 of 30)",
       "with memory on it fell 20-40 points: single fact 43 / 42%, never told 20 / 18%, document 32 / 22%",
       "cause: the engine filled every memory block with six look-alike stored facts, and the model copied them ('Pablo is a large oil painting')"],
      "rejected (strong model, flooded memory)",
      "A model is only as good as the memory block it is handed: precision of what the engine lets in matters as much as training.",
      ["LOG.md 16 Oct", "results/suite_v22_s1.json, _s2.json"]),
    T("STEP1", "Step 1: stricter memory selection, no retraining", "16 Oct 2026 (done 21:31)", P17,
      "How much does v2.2 gain if the engine simply lets less into the memory block?",
      "v2.2 both seeds; settings default, strict, stricter (word 0.30 / meaning 0.60 for conversation, 0.50 / 0.80 for stored facts, at most 2 facts and 4 conversation items), no stored facts; memory-sensitive suite sections",
      "the same suite cases as v2.2",
      ["single fact with memory on: default 43 / 42% -> stricter 74 / 76% -> no stored facts 90 / 91%",
       "never told: 20 / 18% -> 76 / 76% -> 98 / 99%; distractor 38 / 34% -> 66 / 69%; two-step 67 / 71% -> 80 / 87%",
       "stored skills: stricter 65 / 67%, but no stored facts 22% (the model loses stored knowledge)"],
      "adopted: stricter (both seeds agree)",
      "Letting less in is the cheapest gain we have measured: +31 points on single facts with no training.",
      ["LOG.md 16-17 Oct", "results/suite_v22_s*_strict*.json, _stricter.json, _nofacts.json"]),
    T("RETR-STUDY", "Retrieval study: finding vs choosing", "16 Oct 2026 (done 20:06)", P17,
      "Is the engine failing to find the right memory, or failing to choose it?",
      "CPU, no training: 400 conversation cases (5 tasks x 1,024 / 4,096 codes x 40, held-out names), held-out skills facts among 30,000 stored facts; 9 search variants",
      "facts pool 137,944 (Q&A, stored skill facts, document pieces)",
      ["finding: single fact, distractor, correction 100%; two-step 88%; stored skills 98% (word key alone 93%, K2 95%)",
       "choosing: a look-alike's value in the block 49%, the outdated value 99%, a look-alike in never-told blocks 52%; 11.8 items per block, 6 of them stored trivia",
       "recent-turns search off: look-alikes 52 -> 38%, two-step 88 -> 95%",
       "speed: word key 9-10 ms, word + MiniLM 34-40 ms on the CPU, flat from 100 to 10,000 entries"],
      "measured (led to search engine v2)",
      "Finding is solved; choosing is not. The model was handed the right fact plus the outdated value, the look-alike and six unrelated facts.",
      ["LOG.md 16 Oct 20:06", "research_side/retrieval_study.json"]),
    T("ENGINE-V2", "Search engine v2, part 1: fact table, fast lane, decide", "16-17 Oct 2026", P17,
      "Can a structured search choose better than today's engine, and faster?",
      "rules only (no training): records (who, attribute, value, turn, history) from statements; a fast lane for exact who-and-what matches; candidates filtered by a decide step; engine v2 study on the retrieval study's cases",
      "the retrieval study's cases and held-out stored facts",
      ["first study: two-step 72 -> 100%, distractor wrong value 56 -> 0%, correction outdated value 100 -> 8%, never told clean 36 -> 75%, 4.6 items per block, 0.2 / 0.6 ms vs 27 / 40 ms",
       "after the night's fixes ('moved to Lima' read as a city; a stored match counts only when one person matches): correction 80/80 with no outdated value, never-told look-alikes 20 -> 4 of 80, items 18.1 -> 0.6, about 0.1 ms per search",
       "stored skill facts found 105 -> 97 of 120 (the test stores conflicting values; the old engine 'found' them by showing 18 items)"],
      "adopted (rules); part 2, the judge, below",
      "Structured records solve corrections (newest wins, the old value kept as history) and two-step links; rules first, learned parts later.",
      ["docs/SEARCH_ENGINE_V2.md", "research_side/engine_v2_study.json", "LOG.md 16-17 Oct"]),
    T("JUDGE", "Search engine v2, part 2: a learned judge", "17 Oct 2026 (02:48)", P17,
      "Can a small model of our own (the core's first 2 blocks) decide which memory items answer a question?",
      "2 layers from the 5M core + a scoring head; 8,262 steps of 128 (216 s); bar = lowest threshold with held-out precision >= 95%",
      "1,057,554 training / 18,625 test pairs from the real search's results (positives 24.8%)",
      ["AUC 0.941, but at the 95% precision bar it kept only 26% of right items (bar at the grid's 0.95 ceiling)",
       "with the judge on, stored skills fell 63 -> 24%"],
      "rejected (not usable yet; engine v2 rules win)",
      "A learned filter must reach its precision bar at a useful recall before it may filter anything.",
      ["results/chat22_judge.json", "results/suite_v23_s*_ev2jreal.json"]),
    T("REAL-STORE", "The permanent store: real facts only", "16-17 Oct 2026", P17,
      "Does replacing the invented stored pool with real facts change what the model answers?",
      "Adam's rule: lasting knowledge = true facts about real things; invented material only in conversation memory. Store: Wikipedia lead sentences ('What is X?') and clean pieces with the article title",
      "data_chat22/knowledge_wiki.jsonl: 94,276 facts from 25,000 Wikipedia articles (the old pool: 20,001 invented skill facts, instruction Q&A, web pieces)",
      ["on v2.3, real store vs the invented pool: two-step 78 -> 84% / 60 -> 74% (seed 1 / 2); stored skills 42 -> 61% / 36 -> 58%",
       "'What is a fumarole?' answered with the Wikipedia sentence from memory"],
      "adopted",
      "Invented facts in the permanent store were getting in the way; real facts helped on every line measured.",
      ["LOG.md 16 Oct 23:30", "results/suite_v23_s*_real.json"]),
    T("V23", "Chat model v2.3: the any-length study, two seeds", "17 Oct 2026 (done 02:44)", P18,
      "Can the 5M remember a conversation of any length through memory, while keeping its other skills?",
      "5M, window 2,048, plateau stop; ten changes from v2.2 at once (any-length kind 18%, look-alike stored facts, skills told and stored, varied memory budget, 1,024-code memory, stricter memory settings...); seed 1 31,800 steps (2.9 h), seed 2 stopped at 14,400",
      "v2.3 mix: any-length 18, buried 15, plain text 17, Q&A 12, skills 12, conversations 12, stories 8, document reading 4, short recall 2",
      ["recall of 30 facts flat from 100 to 10,000 turns: 22 / 22 / 24 and 21 / 25 / 24 (best earlier model 15 at 10,000)",
       "never told in the window 99% -> 69 / 56%; documents 21-34%; skill updates 5/40; told facts wrongly refused 81 / 87 times",
       "best fluency yet (wiki 1.309 / 1.299); story questions 51 / 40 of 75 (earlier best 30)",
       "seed 2 stopped at 14,400 steps on noisy checks while seed 1 kept improving to 31,800"],
      "rejected overall (the memory design works; other skills lost)",
      "Memory at any length works at 5M. Changing ten things at once made the losses impossible to attribute without a diagnosis.",
      ["results/suite_v23_s1.json, _s2.json, gate22_v23_s*.json", "LOG.md 17 Oct 07:20"]),
    T("V23-EV2", "Engine v2 on the real store, with v2.3", "17 Oct 2026 (03:12-05:13)", P18,
      "Does search engine v2 beat today's search end to end, on the store the model will really use?",
      "v2.3 both seeds; memory sections of the suite; today's search vs engine v2 rules vs engine v2 + judge; real Wikipedia store",
      "the suite's memory cases",
      ["two-step 84 -> 96% / 74 -> 88%",
       "recall at 100 / 1k / 10k turns 63/80/70 -> 87/90/83% and 63/77/80 -> 83/87/90%",
       "never told 60/70/90 -> 90/80/100% and 60/50/80 -> 80/60/100%; wrong values 9/5/6 -> 4/2/3"],
      "adopted (engine v2 rules)",
      "A better search improves every memory line with no retraining.",
      ["results/suite_v23_s*_real.json, _ev2real.json, _ev2jreal.json"]),
    T("LR2", "Learning rate 2e-4 vs 1e-4", "17 Oct 2026", P18,
      "Does a doubled learning rate reach the same model faster?",
      "v2.3 recipe, seed 1, lr 2e-4 (22,200 steps, 1.9 h) vs lr 1e-4 (31,800 steps, 2.9 h); quick suite",
      "v2.3 mix",
      ["same held-out score (0.561 vs 0.559) in 65% of the time",
       "quick suite: wiki 1.340 vs 1.309, conversations 88 vs 110 of 120, memory one fact 5 vs 8 of 12, two-step 8 vs 11 of 12, grounding right 65 vs 74 of 112; better only on rewriting (13 vs 7 of 40)"],
      "rejected (keep 1e-4)",
      "An equal training score can hide broad losses: recipes are decided by the full suite, never by the curve.",
      ["results/chat22_train_v23_lr2.json", "results/suite_v23_lr2_quick.json, suite_v23_s1_quick.json"]),
    T("DIAG-V23", "Diagnosis: why v2.3 lost ground", "17 Oct 2026 (morning)", P18,
      "Did the data change, the percentages, an overwrite, or the search engine and tests?",
      "config diff v2.2 vs v2.3; window lengths measured from the trainer; skills tests under several memory settings; any-length windows counted by how they end",
      "the v2.2 and v2.3 trainers and datasets",
      ["datasets unchanged; the shares changed: short recall 12 -> 2%, document reading 7 -> 4%, Q&A 20 -> 12%, + any-length 18%",
       "short recall trains in-window facts and 'never told' (25%) and is the main long-history kind (68% of its windows > 1,700 codes): its cut explains the window losses",
       "30% of any-length windows end in 'I don't have that', all with memory blocks: v2.3 refused told facts 72-81 times under every setting (v2.2: 0-1)",
       "documents fell at v2.1 -> v2.2 (90/82/82 -> 57/49/22% at 512/1,024/1,900 codes), not at v2.3",
       "not the cause: the search engine and test settings (window tests run with memory off)"],
      "measured (led to variants A-F)",
      "A kind whose held-out loss has levelled off is not finished: cut it and its skill fades. Change one thing at a time.",
      ["LOG.md 17 Oct ~08:00", "results/suite_*_dx_*.json"], type="analysis"),
    T("V24A", "Variant A: v2.2's recipe + any-length conversations", "17 Oct 2026 (trained 09:30, tested 10:04)", P19,
      "Can the memory gains be added to v2.2 without its losses?",
      "5M, seed 1, floor 20,000 steps, patience 10; 26,400 steps (1.5 h); switches: any-length kind 12%, full-length buried documents",
      "v2.2 shares scaled to make room for any-length 12%",
      ["best or tied-best on 26 of 39 suite fronts across all chat models (v2.1 seed 1: 20)",
       "memory on: one fact 87%, never told 98%, two-step 90%; window never told back to 99%",
       "recall 25 / 24 / 25 of 30 at 100 / 1k / 10k turns; told skills 79/180, stored 113/180; grounding 79% right, 10% invented, 1.3% wrong memory",
       "still open: documents 26-32%, skill updates 8/40, story consistency"],
      "adopted (best model so far)",
      "Restoring the kinds v2.3 cut, while keeping the any-length kind, kept both kinds of gain.",
      ["results/suite_v24a_s1.json, gate22_v24a_s1.json"]),
    T("V24A-UPD", "Skill updates on A: setting or model?", "17 Oct 2026", P19,
      "Are A's lost skill updates caused by the strict memory setting?",
      "variant A, skills section under strict, loose and strict with 8 conversation items",
      "held-out skills test v2",
      ["updates 8, 9 and 7 of 40 under the three settings (v2.2: 19)",
       "strict remains A's best total (79 vs 65 / 75); A says 'I don't know' correctly 30/40 but refuses 38 told facts"],
      "measured (the model, not the setting)",
      "Refusing and inventing trade against each other; the balance must be trained, not tuned at answer time.",
      ["results/suite_v24a_s1_dx_def.json, _dx_k8.json"]),
    T("GPU-OVERFLOW", "Three trainings side by side overflowed the GPU", "17 Oct 2026 (08:10)", P19,
      "Can three 5M variants train at once to save time?",
      "variants A, B, C started together on one RTX 3090",
      "-",
      ["one run takes about 11.4 GB of GPU cache; together 21.7-23.8 of 24.6 GB, 98% busy; B and C never logged a step, A stalled",
       "caught within about 15 minutes; A recovered at once when the others stopped",
       "fix: --gpu-frac memory cap (an error, never paging), one training at a time"],
      "bug fixed", "Every unattended job needs a hard memory cap; a full card stalls silently, as session 36 did for 9.5 hours.",
      ["LOG.md 17 Oct 08:10-08:30"], type="bug"),
    T("V24B", "Variant B: A + look-alike facts + varied memory budget", "17 Oct 2026 (trained 12:43, tested ~14:15)", P19,
      "Do v2.3's memory-hardening changes add anything on top of A?",
      "as A plus look-alike stored facts in half the windows, varied memory budget, 1,024-code memory; 33,000 steps (3.2 h)",
      "A's shares",
      ["significantly worse than A on 5 fronts: stored skills 113 -> 44/180, correct 'I don't know' 30 -> 10/40, documents 53 -> 33 (window) and 65 -> 44 (memory) of 204, story contradictions 0.68 -> 1.52",
       "better on 1: reply time p95 34 -> 28 ms",
       "its Q&A, conversation and story held-out scores peaked at 7-17k steps, then worsened"],
      "rejected (these changes stay out of the merged recipe)",
      "Hardening tricks that looked sensible hurt general skills; each change must earn its place on the suite.",
      ["results/suite_v24b_s1.json, gate22_v24b_s1.json"]),
    T("V24C", "Variant C: all v2.3 changes, shares restored", "17 Oct 2026", P19,
      "Do v2.3's changes work once short recall, buried evidence and document reading get their v2.2 shares back?",
      "all v2.3 switches, recall 12%, buried 15%, document reading 5%; floor 20,000 steps",
      "C's shares", ["training (step 16,500 at 15:20); suite to follow"], "in progress", "-", ["runs/chat22_train_v24c_s1.log"]),
    T("DATA-REAL", "data_real: real, non-fiction datasets for testing", "17 Oct 2026", P19,
      "What in our data is invented, and can we build a real, modern American English set to test against it?",
      "audit of every dataset; a test folder beside the training data (nothing replaced); American spelling filter; held-out paragraph check; 5M and 100M size folders; trainer --data-dir switch",
      "on disk + downloads approved by Adam: Dolly, SQuAD, ARC, OpenBookQA, CommonsenseQA, FineWeb-Edu, OASST; HotpotQA, Natural Questions (open), SciQ, BoolQ",
      ["audit: 1800s Gutenberg novels are 13% of the plain text and 42% of the long-document pool; Alpaca is machine-written and partly invented; skills, any-length facts and story worlds invented on purpose",
       "real Q&A 292,675 + 29,470 held-out (Alpaca: 50,965); fact store 215,602; plain text from web documents the cores never saw (81% of the file's first documents were already in core training)",
       "5M plan 30-48k steps (Q&A 1.7 passes in the draft mix before the downloads); 100M plan 25-40k steps",
       "two leaks caught while building: 1,361 documents shared with held-out text, and SciQ training questions in its held-out set"],
      "in progress (built; trained only after the current queue)",
      "Real data must be checked for leaks into the test text at every step; whether it trains a better model is still to be measured.",
      ["data_real/README.md", "data_real/<size>/plan.json", "LOG.md 17 Oct"]),
]

NEW_CHAPTERS = [
    {"id": "memscale", "title": P17, "dates": "16 October, afternoon and evening",
     "paras": [
         "<b>Memory scales; the window has a cost.</b> A fact told in turn 1 was still recalled 10 of 10 times after 5,000 turns, with a lookup of about 18 ms. The CPU can hold very long windows, but waits grow faster than the window, and the attention cache belongs on the GPU: RAM is 2-5 times and the drive 4-31 times slower per code. A side test showed that a short, clean input of retrieved items keeps a small model grounded: arm A said 'I don't know' correctly 11 of 12 times from memory alone, and 0 of 12 times with a long window of filler.",
         "<b>v2.2, the first model from the clean package, was strong and badly served.</b> It was the best of every chat model on clean text, told facts and in-window reasoning, but with memory on it fell 20-40 points. The engine was filling every memory block with six look-alike stored facts, and the model copied them. Step 1 needed no training: letting less into the block raised single facts from 43% to 74% and never-told answers from 20% to 76%.",
         "<b>The retrieval study put a name to it: finding was solved, choosing was not.</b> The search found the right fact every time, but handed it over with the outdated value, the look-alike and six unrelated facts. Search engine v2 answers with structure: records of who, what and when, the newest value first, links for two-step questions, and an empty block when nothing qualifies. A learned judge was tried and was not yet precise enough, so rules decide for now. On Adam's rule, the permanent store became real facts only (Wikipedia), and that helped on every line measured."],
     "tests": ["S38-MEM", "S39-CPU", "S40-DISK", "MEM-ONLY", "V22", "STEP1", "RETR-STUDY", "ENGINE-V2", "JUDGE", "REAL-STORE"]},
    {"id": "anylength", "title": P18, "dates": "16-17 October, night",
     "paras": [
         "<b>v2.3 answered the main question: memory works at any length.</b> Recall of 30 planted facts stayed flat from 100 to 10,000 turns, and with engine v2 it reached 83-90%. Reply time stayed at about a quarter of a second whatever the length: length costs storage, not speed.",
         "<b>But it lost ground elsewhere</b>, and it had changed ten things at once. A morning of diagnosis found the causes. The data had not changed; the percentages had. Short recall, cut from 12% to 2%, had been quietly teaching the window skills and supplying most long histories. The new any-length data ended 30% of its examples with 'I don't have that', and the model learned to refuse told facts. Documents had been weak since v2.2. The search engine and the tests were not the cause. A doubled learning rate looked equally good on the training curve and was worse on the suite."],
     "tests": ["V23", "V23-EV2", "LR2", "DIAG-V23"]},
    {"id": "repair", "title": P19, "dates": "17 October",
     "paras": [
         "<b>The first repair worked.</b> Variant A put v2.2's recipe back and added the any-length conversations at 12%. It became the best model of the whole research: best or tied-best on 26 of 39 test fronts, with the best memory results yet and the window skills back. B showed that v2.3's memory-hardening tricks hurt; C, D, E and F test the rest, and the gaps that remain (documents, skill updates, story consistency).",
         "<b>A plan to merge every gain.</b> One merged training stage with every specialised kind at the share the evidence supports, then a grounding stage of automatically graded answer pairs, with engine v2 and the real-facts store at answer time. A test folder of real, modern datasets was built and audited, with the downloads Adam approved, to be trained only once the current queue is done."],
     "tests": ["V24A", "V24A-UPD", "GPU-OVERFLOW", "V24B", "V24C", "DATA-REAL"]},
]

NOW = {"id": "now", "title": "Where we are now, and what comes next", "dates": "17 October, afternoon",
       "paras": [
           "<b>The best model is variant A</b>: memory that works at any length (recall flat to 10,000 turns), the window skills back, and grounded answers (79% right with memory on, 10% invented). It is one seed; a second seed confirms it before it becomes the base. Search engine v2 and the real-facts store are adopted for answering.",
           "<b>Running now:</b> variant C, then D and E (documents) and F (long windows). Their results set the shares of the merged model: one chat stage on two seeds, then the grounding stage. The real datasets (data_real) are built and wait until the current queue finishes; they are unproven until trained and tested like everything else.",
           "<b>Further ahead:</b> the window-removed and 'lots of information' tests, a 16k window study on the winner, then the 100M with the merged recipe, only once the 5M study is complete. The goal is unchanged since September: <b>a small, fast model that keeps its knowledge outside its weights</b>, and remembers a conversation of any length."],
       "tests": []}

NEW_LESSONS = {
    "Memory and copying": [
        {"text": "<b>Finding is easy, choosing is hard</b>: a short, precise memory block beats a generous one (single facts 43% to 74% with no training).", "tests": ["STEP1", "RETR-STUDY", "V22"]},
        {"text": "Structured records solve corrections and two-step questions: newest value first, the old one kept as history, names linked to relations.", "tests": ["ENGINE-V2", "V23-EV2"]},
        {"text": "The permanent store holds <b>real facts only</b>; invented material belongs in conversation memory.", "tests": ["REAL-STORE"]}],
    "Training recipe": [
        {"text": "<b>What leaves the mix fades</b>, even a kind whose held-out loss looked finished; change one thing at a time.", "tests": ["V23", "DIAG-V23", "V24A"]},
        {"text": "Refusing and inventing trade against each other: balance the 'never told' examples against told facts.", "tests": ["V23", "V24A-UPD"]},
        {"text": "Hardening tricks must earn their place: look-alike facts and a varied memory budget hurt general skills.", "tests": ["V24B"]}],
    "Long context and memory": [
        {"text": "Memory works at any length: recall flat from 100 to 10,000 turns, reply time flat; the window cache stays on the GPU.", "tests": ["S38-MEM", "V23", "V24A", "S40-DISK"]}],
    "Measuring honestly": [
        {"text": "A matching training score can hide broad losses: decide recipes by the full suite.", "tests": ["LR2"]},
        {"text": "Check new data for leaks into the test text at every step, and cap the GPU memory of every background job.", "tests": ["DATA-REAL", "GPU-OVERFLOW"]}],
}


def main() -> int:
    tests = json.loads((HERE / "tests.json").read_text(encoding="utf-8"))
    ids = {t["id"] for t in NEW_TESTS}
    tests = [t for t in tests if t["id"] not in ids] + NEW_TESTS
    story = json.loads((HERE / "story.json").read_text(encoding="utf-8"))
    story["eyebrow"] = "Text Research Model · September to 17 October 2026"
    keep = [c for c in story["chapters"] if c["id"] not in {"memscale", "anylength", "repair", "now"}]
    story["chapters"] = keep + NEW_CHAPTERS + [NOW]
    for g in story["lessons"]:
        for item in NEW_LESSONS.get(g["title"], []):
            if item["text"] not in [i["text"] for i in g["items"]]:
                g["items"].append(item)
    known = {t["id"] for t in tests}
    missing = [i for c in story["chapters"] for i in c["tests"] if i not in known]
    missing += [i for g in story["lessons"] for it in g["items"] for i in it["tests"] if i not in known]
    (HERE / "tests.json").write_text(json.dumps(tests, ensure_ascii=False, indent=1), encoding="utf-8")
    (HERE / "story.json").write_text(json.dumps(story, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{len(tests)} tests, {len(story['chapters'])} chapters; unknown links: {sorted(set(missing)) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
