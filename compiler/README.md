# Compiler v0 — the research text encoder

Compiler v0 turns text (any bytes) into integer IDs for model training, and back again exactly. It is one layer: a scanner splits the text into units, and each unit is looked up in one frozen dictionary. It keeps no state between units, so the same text always gets the same IDs, in any file and at any position.

| File | What it is |
| --- | --- |
| `compiler_v0.py` | Reference implementation: the specification, dictionary fitting, encode/decode, file formats, CLI |
| `native/cv0.hpp`, `native/cv0.cpp` | Fast C++ encoder/decoder; must produce byte-identical ID files |
| `test_compiler_v0.py` | Contract tests: scanner rules, round trips on adversarial and random input, statelessness, deterministic fitting, file formats, C++/Python parity |
| `experiments/prepare_books.py` | Downloads and prepares the public-domain book corpus |
| `experiments/initial_tests.py` | The initial measurements recorded in `results/compiler_v0_initial.json` |
| `experiments/ladder_encodings.py` | One interface over the encodings under test: bytes, v0 dictionaries, dictionary-free hash codes |
| `experiments/train_lm.py` | Trains one small Transformer on one encoding for a fixed time budget; reports bits per original byte, speed, a talk-back sample |
| `experiments/ladder1.py` | Ladder 1: the same model and text across six encodings; writes `results/ladder1.json` |
| `experiments/summarize_ladder.py` | Renders a ladder result as Markdown tables, including the equal-bytes view |
| `experiments/chat.py` | Talks to a saved ladder model through its own encoder |
| `experiments/test_ladder.py` | Contract tests for the encodings and the harness |

## How it encodes

```text
text ─► scan into units ─► [case flag] ─► phrase? ─► longest-match pieces ─► IDs
                                                                   └─ byte fallback (IDs 0–255)
IDs ─► look up each ID's bytes ─► apply case flags ─► exact original text
```

- **Units:** a word with its leading space (`" cat"`), a bare word, a run of spaces, newlines or tabs, or any other single byte. Digits are single bytes, so numbers are spelled digit by digit.
- **Case flags (optional):** `The` becomes `CAP` + `the`, and `NASA` becomes `UPPER` + `nasa`; mixed case such as `iPhone` is left as is.
- **Phrases (optional):** the longest run of 2–3 consecutive units that matches a phrase entry (`" of the"`, `".\n\n"`) becomes one ID.
- **Pieces:** otherwise the unit is split greedily, longest match first. The first piece comes from the start table (which includes whole words) and later pieces from the continuation table. A single byte is used when nothing matches, so every input is lossless.
- **IDs:** 0–255 are bytes, 256 and 257 are the CAP and UPPER flags, and 258 onwards are dictionary entries in file order.
- **Dictionary file (`.cv0d`):** a canonical text file. Its FNV-1a 64-bit hash identifies it, and every ID file records that hash, so IDs are never decoded with the wrong dictionary.
- **ID file (`.ids`):** a 24-byte header (`CV0I`, version, ID width, dictionary hash, count) followed by little-endian `uint16` IDs, or `uint32` when the vocabulary exceeds 65,536.

Dictionaries are fitted once, on the training split only. Start and continuation pieces come from byte-pair merges inside units, tracking whether a piece starts its unit. Phrases are the runs of single-ID units that save the most symbols.

## Use

```bash
# fit a dictionary (training text only)
python compiler/compiler_v0.py fit data/train/*.txt --out v0_32k.cv0d --vocab 32768 --no-case-flags \
    --phrases 4096 --max-phrase-units 3

# build the fast tool
g++ -O2 -std=c++17 -o cv0 compiler/native/cv0.cpp      # MSVC: cl /O2 /std:c++17 /EHsc compiler\native\cv0.cpp

# encode, decode, benchmark
./cv0 encode v0_32k.cv0d book.txt book.ids
./cv0 decode v0_32k.cv0d book.ids book.txt
./cv0 bench  v0_32k.cv0d corpus.txt 5

# tests
python -B -m unittest compiler/test_compiler_v0.py -v
python -B -m unittest compiler/experiments/test_ladder.py -v
```

## Run it yourself (menu, popups, GPU)

```bash
python run_model.py            # from the repository root
```

Option 1 asks for a folder in a popup and trains on every `.txt` file in it (the last file is
held out for scoring; or use `train/`, `heldout/`, `ood/` subfolders). The model, its result
JSON and the dictionary are saved in `models/` inside that folder. Option 2 asks for a `.pt`
file and opens a chat; the encoder and the count tables are rebuilt from the dataset the model
was trained on. Every knob (model size, encoding, tables, budget, device, sampling) is in the
`SETTINGS` block at the top of the script. `--train DIR` and `--chat MODEL.pt` skip the menu.

## Train and talk (needs PyTorch)

```bash
python compiler/experiments/prepare_books.py data
g++ -O2 -std=c++17 -o data/cv0 compiler/native/cv0.cpp
python compiler/experiments/ladder1.py data data/work --native data/cv0 --budget 600 --out results/ladder1.json
python compiler/experiments/summarize_ladder.py results/ladder1.json
python compiler/experiments/chat.py data/work/ladder1_v0_8k_plain.pt --prompt "We went to the park"
```

In Python, `compiler_v0.read_ids(path, dictionary)` returns the IDs for training. For large corpora, `numpy.fromfile(path, dtype="<u2", offset=24)` reads a 16-bit ID file directly.

## Reproduce the initial tests

```bash
python compiler/experiments/prepare_books.py DATA
g++ -O2 -std=c++17 -o cv0 compiler/native/cv0.cpp
python compiler/experiments/initial_tests.py DATA WORK --native ./cv0
```

## Plan

The review of this encoder and the phased plan for the next experiments are in [`docs/ENCODER_RESEARCH_PLAN.md`](../docs/ENCODER_RESEARCH_PLAN.md).

## Limits of v0

- Letters are ASCII only. Other scripts and accented letters fall back to UTF-8 bytes (2–3 IDs per character).
- Numbers are spelled digit by digit; the number channel is Track D.
- Phrases make an ID depend on up to two neighbouring units. Encoding stays deterministic and local, but a sentence's IDs can differ at its edges depending on what touches it.
- The dictionary is fixed. Changing it means a new dictionary hash and retraining the model's embeddings.
