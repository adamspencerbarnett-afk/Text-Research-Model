# Chat notes: graded conversations with the Q&A models

How this works: `compiler/experiments/chat_probe.py` asks every model the same 32 questions in
seven sections, each with memory (the model as designed) and without (the network alone).
Transcripts are in `docs/chat_probes/`. I read every answer and grade it here:
**2** correct and readable, **1** on topic but partly wrong or garbled, **0** wrong, off topic,
or broken. The keyword check in the transcript is only a rough flag; these grades are the record.

## 10 October 2026: `models/qa_best_5M.pt` (R1b seed 2, 4.7M core adapted from books + Wikipedia, copy head)

| section | what it tests | grade (of max) | without memory |
| --- | --- | ---: | ---: |
| stored | Alpaca training questions, verbatim | **10 / 10** | 0 / 10 |
| reworded | the same five questions in my words | **1 / 10** | 0 / 10 |
| knowledge | general facts (planets, Shakespeare, boiling point, Tokyo, photosynthesis) | 9 / 10 | 0 / 10 |
| heldout | held-out Alpaca instructions (poem, Italian dishes, hairstyle, print) | 4 / 10 | 1 / 10 |
| taught | 3 facts added with /add, asked verbatim then reworded | **10 / 12** | 0 / 12 |
| arithmetic | 12 + 30, 5 − 2 apples, 7 × 8 | 0 / 6 | 0 / 6 |
| chat | greeting, a short email, what to cook | 1 / 6 | 0 / 6 |
| **total** | | **35 / 64 (55%)** | **1 / 64** |

What the transcript shows, in order of importance:

1. **The model answers from memory, and only from memory.** Without memory, 1 of 64 points;
   with it, 35. Every correct answer came from a stored or near-stored exchange. The network
   on its own writes grammatical-looking text that is never right ("The capital of France is the
   capital of Spain"; "The boiling point of water in Celsius is 10").
2. **Copying works when the retrieved question is close to the asked one, and breaks when it is
   not.** Taught facts asked in new words were answered word-perfect at similarity 0.74 and
   0.48 ("On what date was Version 1 frozen?" → "...frozen on 9 October 2026"). But
   "List ten verbs you would use when cooking" retrieved exactly the right stored exchange
   (0.47) and produced "When cooking, cooking.", and "What are some well-known computer
   viruses?" retrieved the right one (0.51) and produced "'Bird', 'C', and 'F'". The copy head
   decides *whether* to copy from the wording of the question, and it was only ever trained on
   two cases: the identical question (copy) and a different question (don't). Reworded
   questions fall in between. This is the training fix R0b.
3. **Retrieval misses synonyms.** "Which graphics card is in the research computer?" did not
   find the taught GPU fact (it matched "Give three types of computer graphics"); "What food do
   Americans usually eat on Thanksgiving?" matched a question about Spain; "most common
   hairstyle for men" matched sneakers. The key is literal content words, so "graphics card"
   and "GPU", "Americans" and "US" never meet. Needs a second, learned retrieval key.
4. **The knowledge section is memory too.** All five "general facts" scored because Alpaca
   happens to contain near-identical questions (similarity 0.59 to 0.79). It is not evidence of
   knowledge in the network.
5. **Arithmetic and conversation are outside the training data's shape.** "What is 12 plus 30?"
   → "12 plus 30 is 14". "Hello, how are you today?" → "I today / I today is today." Alpaca has
   almost no greetings or small talk, and its arithmetic answers are bare numbers with no steps.
6. **Data bug found by reading:** the email answer stopped at "Dear [1],". Exchanges were split
   at every blank line, so the 3,450 Alpaca answers (6.6%) with more than one paragraph (letters,
   stories, poems, code) were cut to their first paragraph and 10,826 paragraphs were dropped.
   The model has never seen a whole letter. Fixed in `parse_exchanges` (split only before a new
   `User:` turn); every Alpaca result before this date carries the truncation.
7. **Smaller defects:** blocking repeated 3-code phrases stops loops but sometimes forces broken
   word pieces ("Jupy", "ingrednts"); answers can run on into an invented next turn
   ("Assistant: The night was a beautiful day"), which the engine should cut.

What this changes in the research queue:

| priority | change | why (from the transcript) |
| --- | --- | --- |
| 1 | parse fix: whole multi-paragraph answers | finding 6 |
| 2 | R0b: train the copy head on reworded questions (the retrieved exchange's question perturbed: words dropped, reordered, swapped for neighbours in the dictionary), plus the retrieval similarity as a trust-head input | finding 2 |
| 3 | conversation data: add Dolly 15k and a dialogue set, so greetings and multi-turn exist in training | finding 5 |
| 4 | a learned retrieval key (the core's own state for the question) beside the literal one | finding 3 |
| 5 | cut generation at an invented "Assistant:" turn; repeat blocking on words, not codes | finding 7 |
