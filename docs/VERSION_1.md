# Version 1: the deterministic text compiler and the table-backed model

*Frozen 9 October 2026. Research concept and direction by Adam Barnett. This file and the
accompanying `releases/version1.zip` record what was built, why, and what was learned.*

## Why the research was done

The question was whether a model trained on text that has first been compiled into integer
codes by a small deterministic encoder can be lighter, faster and smarter than a conventional
model trained on the same text; and whether, by putting far more parameters into cheap
lookup tables around a small compute core, a model can hold much more knowledge at a fraction
of the cost. The same compiler was to serve at training time and at conversation time, so the
model and its users would always speak the same codes.

## What was built

- **Compiler v0** (`compiler/compiler_v0.py`, `compiler/native/`): a stateless single-pass
  encoder with one frozen dictionary. Words keep their leading space; rare words are spelled in
  pieces, then bytes, so every input round-trips exactly. The dictionary is a text file with a
  hash that every encoded file carries. The C++ encoder runs at 112 MB/s and is byte-identical
  to the Python reference. 19 contract tests.
- **An equation encoder** (`HashCodes` in `ladder_encodings.py`): no dictionary at all, every
  unit's code is a hash of its bytes; decoding uses a reverse map of units seen in training.
- **A training harness** (`train_lm.py`, `ladder1.py`): one small Transformer, trained for a
  fixed time budget on any encoding, scored in bits per original byte on held-out books, with a
  validation split, plateau early stopping, a talk-back sample through the compiler, and a
  split of the result by word frequency and type. `chat.py` talks to a saved model.
- **Tables**: hashed n-gram input tables (learned rows, one lookup per code) and exact n-gram
  follower tables mixed into the output through a learned trust head, with leave-one-out
  counting in training.
- **Corpora**: 45 MB, 200 MB and 233 MB of public-domain books with manifests; a Wikipedia
  split; a chat-formatted instruction set (`prepare_qa.py`).

## What was learned, in order

1. Compiled text beats raw bytes by 8% at equal compute and answers 2.5x faster; against a
   standard BPE tokenizer with the same core it is only 1% better. The compiler's value is the
   stability of its codes, which is what the tables need.
2. Phrase codes ("one code for a whole phrase") shorten sequences by 18% but make the model
   2.5-4% worse: phrases are rarer than their words and hide the words' identities.
3. The dictionary-free hash encoder is within 1.5% of the fitted dictionary, with 1.3-4.2% of
   held-out bytes undecodable because they were never seen in training.
4. Parameters in tables are nearly free: hashed tables of 34M to 537M parameters improved the
   model by 3.4-4.7% at a 5-13% step cost and no loss of reply speed; a quadrupled core bought
   nothing in the same time.
5. Exact count tables with a learned trust head are the strongest idea so far, once the counts
   exclude the current occurrence in training: 10.9% better than the BPE model at equal time,
   10.5% at equal text, with the same 1.9M learned parameters. The gain survived 4.4x more data
   (11.0% at 45 MB, 9.3% at 200 MB on the same books).
6. The gain lives on the middle and rare vocabulary (16.6% and 10.6% better), not on function
   words (0%), and the model loses on words so rare that they are spelled in bytes (8% worse).
7. 64 followers per context beats 16 by 2.1%; the hashed tables are then nearly redundant.
8. Learning by reading: one book by an author added to the count tables alone, with no
   retraining, improved prediction of another book by that author by 0.59%, against 0.02% for
   an unrelated book of the same size. The tables are a knowledge store the model reads from
   and that can be updated in seconds.
9. Protocol lessons: 4-minute screens reproduce the 10-minute ordering; seed noise is 0.2-0.6%;
   the n-gram proxy misleads about phrases; tables counted on the training text must use
   leave-one-out; a watcher must not match its own command line.

## What it is not

The models are 0.79M-parameter cores trained for minutes on novels. They continue text in the
style of the books; they cannot answer questions, and nothing in them is reasoning. The
numbers above are learning-efficiency results that say which direction to scale, not
intelligence results.

## Contents of the zip

`compiler/` (encoder, native code, tests, experiments), `docs/` (plan, this file, figures),
`results/` (every result file and the corpus manifests), `dictionaries/` (the fitted v0 8k
dictionary and BPE merges), `models/` (the saved hash-encoder model and its reverse map, and
the learn-by-reading core, usable with `chat.py`; the n-gram tables are rebuilt from the corpus
with `--data`), and `README.md` pointing here.
