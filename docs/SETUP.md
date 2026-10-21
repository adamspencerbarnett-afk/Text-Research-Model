# Setup, training data and the full pipeline

*Current as of 20 October 2026: the best model is C2b (recipe in `recipes/BEST.json`). The 100M has its own guide: `docs/TRAIN_100M.md`.*

Everything runs on one machine with an NVIDIA GPU (the research uses an RTX 3090, 24 GB, Windows 10). Commands use the
Windows paths (`.venv/Scripts/python.exe`) in bash (Git Bash); on Linux or macOS use `.venv/bin/python`, and for the chain
scripts set `PY=.venv/bin/python`.

## 1. Environment

```bash
git lfs install                       # once per machine, before cloning; then: git clone ..., or git lfs pull afterwards
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -B -m unittest compiler/test_compiler_v0.py compiler/experiments/test_ladder.py
```

Verified with Python 3.12, PyTorch 2.11 (CUDA 12.8 build), NumPy 2.5. PyTorch 2.11 or later is needed (torch.optim.Muon).
`bitsandbytes` is only for the int8 kernel experiments.

For training a core or encoding large corpora, build the native encoder:

```bash
g++ -O2 -std=c++17 -o data/cv0.exe compiler/native/cv0.cpp     # pass its ABSOLUTE path to --native
```

## 2. What the repository carries

| file | what | size |
| --- | --- | --- |
| `runs/chat22/combo2b_5m_s1.pt` + `.tables.npz` | the best chat model (C2b) and its n-gram tables (LFS) | 28 + 92 MB |
| `runs/core_mt5m/scale_custom_v0_8k.pt` | the 5M core every chat model starts from (LFS) | 27 MB |
| `runs/core_mt5m/v0_8k_plain.cv0d` | the 8,192-entry dictionary of the compiler | 66 KB |
| `data_real/shared/knowledge/knowledge_real.jsonl` | the permanent store: 311,010 real facts (LFS) | 87 MB |
| `data_chat22/eval_clean/`, `data_core/eval_clean/heldout/`, `data_mt/heldout/` | held-out text and dialogues the suite and training's held-out windows read | 16 MB |
| `data_skills/test_v2.json`, `test_v4_rewrite.json` | the skills test sets | 61 KB |

That is enough to chat with the best model (`run_model.py`, option 3) and to run the full suite on it. The training data
and the 100M core are in `datasets/` (Git LFS, fetched on request; `datasets/README.md`).

## 3. Training data for the best recipe

`chat22.train` with the C2b recipe (`--data-dir data_real/5m_mixed --parts longconv,longdocs,alpacaqa,multi2`) reads these
files (each training log prints the resolved list as "data sources"); 2.75 GB in all, in one bundle:

```bash
git lfs pull --include="datasets/c2b_5m/*"
tar -xJf datasets/c2b_5m/c2b_5m_training_data.tar.xz      # from the repo root
```

| path | size | what | built by |
| --- | ---: | --- | --- |
| `data_qa/train/qa_train.txt`, `data_qa/heldout/qa_heldout.txt` | 19 MB | Alpaca Q&A (instruction style; also the n-gram tables) | `compiler/experiments/prepare_qa.py alpaca_data.json data_qa` |
| `runs/chat5m_v2/dense_top4.npy` | 1.6 MB | MiniLM neighbours of the Alpaca questions | `compiler/experiments/train_chat.py` (chat model v2) |
| `data_real/shared/skills/train_v4c.jsonl` | 44 MB | generated personal facts told in conversation (skills v4c) | `compiler/experiments/gen_skills_v4.py` (copy of `data_skills/train_v4c.jsonl`) |
| `data_real/shared/convo/dialogues_train.jsonl` | 1.2 GB | real dialogues (UltraChat, OASST), de-duplicated | `compiler/experiments/prepare_dialogue.py` -> `chat22/data/build_data.py --only dialogues` |
| `data_chat22/rehearse_P100_whole_554MB_seed1.txt` | 555 MB | plain-text rehearsal (Wikipedia, web, books, conversations, evidence pairs) | `chat22/data/build_data.py --only rehearse` (from `data_core/mix_P100`) |
| `data_core/mix_P100/train/books.txt` | 224 MB | long documents for story continuation and reading | `compiler/experiments/prepare_core_mix.py mix ... --recipe P100` |
| `data_real/shared/look/facts_pool_real.jsonl` | 61 MB | real look-alike facts for the search in training | `python -m chat22.data.build_real` |
| `data_real/shared/look/facts_pool_mixed.jsonl` | 65 MB | the pool behind the sessions (to build more) | real pool + the skills data's 20,001 personal facts |
| `data_real/5m_mixed/longconv/longconv_s1_[0-3].jsonl` | 593 MB | any-length sessions on the mixed pool (real facts + generated personal facts) | `chat22.data.build_v23`, below |

The any-length sessions of the best recipe (4 shards in parallel on the CPU, about 1.5 hours):

```bash
for S in 0 1 2 3; do OMP_NUM_THREADS=4 .venv/Scripts/python.exe -B -m chat22.data.build_v23 --only longconv --sessions 1250 --shard $S \
    --facts data_real/shared/look/facts_pool_mixed.jsonl --out-dir data_real/shared/longconv_mixed \
    --dialogues data_real/shared/convo/dialogues_train.jsonl & done; wait
mkdir -p data_real/5m_mixed/longconv && cp data_real/shared/longconv_mixed/longconv_s1_[0-3].jsonl data_real/5m_mixed/longconv/
```

`facts_pool_mixed.jsonl` is `facts_pool_real.jsonl` (212,947 real facts) plus the 20,001 generated personal facts of the
skills data: a training signal only. The store used for answering stays real-only.

## 4. Train and test

```bash
bash compiler/experiments/chains/train_best.sh 1 base_s1                  # reproduces the best recipe on seed 1
bash compiler/experiments/chains/train_best.sh 1 idea_s1 --parts ...      # the base recipe plus one change
.venv/Scripts/python.exe research_side/compare_two_seeds.py idea_s1 idea_s2   # after a second seed: the verdict
```

`train_best.sh` reads `recipes/BEST.json` (core, data folder, mix, parts, shares, step floor and ceiling, patience,
memory settings, GPU share, answer-time engine options), trains (about 2 hours), runs the full suite and the new tests
(about 50 minutes) and compares with the best. The underlying commands:

```bash
.venv/Scripts/python.exe -B -m chat22.train --name NAME --core runs/core_mt5m/scale_custom_v0_8k.pt --data-dir data_real/5m_mixed \
    --mix v22 --parts longconv,longdocs,alpacaqa,multi2 --shares "$(python -c "import json;print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['shares']))")" \
    --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 \
    --memory-opts '{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}' --gpu-frac 0.6
EO='{"search": "v2", "hop2": true, "split": true}'
.venv/Scripts/python.exe -B -m chat22.tests.run_suite runs/chat22/NAME.pt --tier full --engine-opts "$EO" --tag _ev2full
.venv/Scripts/python.exe -B -m chat22.tests.run_suite runs/chat22/NAME.pt --tier full --engine-opts "$EO"     --sections afterno,phrasing,probe2,reason,twomem,open,multi --tag _ev2new
```

`--shares` must name every kind (the trainer logs the effective shares). The model, its tables and a resume state go to
`runs/chat22/`; the training curve and settings to `results/chat22_train_NAME.json`; the suite to
`results/suite_NAME_ev2full.json`.

Side tests used on candidates: `research_side/natural_rewording_test.py` (the same facts asked the way people talk),
`research_side/fresh_facts_test.py` (facts added after training).

### The 20 October additions: new tests, training switches, answer-time options, the loop

```bash
# the new tests (chat22/tests/suite_v2.py), merged by compare_two_seeds.py
.venv/Scripts/python.exe -B -m chat22.tests.run_suite runs/chat22/NAME.pt --tier full --engine-opts '{"search": "v2"}' \
    --sections afterno,phrasing,probe2,reason,twomem,open,multi --tag _ev2new      # add --merge to re-run one section
```

| test | what it checks |
| --- | --- |
| `afterno` | answers directly after a correct refusal (arithmetic, told fact, stored fact, after two refusals) vs the same with no refusal |
| `phrasing` | 200 told-and-asked facts in natural chat wording the training never uses |
| `probe2` | the 32-question chat probe with stored / reworded questions rewritten for the real store |
| `reason` | counting what was told, the first thing told, a summary |
| `twomem` | a told personal fact plus a stored real fact ("My favorite song is X." -> "Who wrote my favorite song?"); control: the store question asked directly |
| `open` | open-ended requests vs held-out Alpaca references (overlap, complete last sentence, repetition, length) |
| `multi` | several facts told at once, two asked at once, known + unknown, statement + question, two stored facts, a quoted note |

Training switches (`--parts`, each leaves older recipes byte-identical): `afterno` (30% of qa / skills / recall /
any-length windows get 1-2 earlier correct refusals), `twomem` (20% of recall windows: told fact + stored fact from the
store's train split), `reason` (30% of recall windows: counting, first told, recap), `multi` (35% of skills windows:
multi-part messages; 15% of qa windows: two stored facts asked at once). `--accum N` runs a batch in N parts (for the 100M).

Answer-time options (engine and suite `--engine-opts`, off by default): `"hop2": true` (two-hop search through what the
user said), `"split": true` (one search per question of a message; each sentence of a statement parsed).

The loop: `recipes/LOOP.json` lists the candidates (BASE + changes, optional `engine_opts`); `research_side/loop_5m.py`
trains seed 1, runs the full suite and the new tests, compares, trains seed 2 only for a win with no front lost, and
writes a two-seed winner into `recipes/BEST.json`. Start it with
`nohup .venv/Scripts/python.exe -u research_side/loop_5m.py >> runs/loop_5m.log 2>&1 &`; mark a round `"ready": true`
when its code is smoke-tested.

## 5. Rebuilding the data from the sources

Only needed without a copy of the data folders. Sources, licences and URLs are in `docs/DATASETS.md` and
`data_real/README.md`; each builder documents its options at the top of the file.

1. Source corpora into `data_core/src_p50/` and the P100 mix: `compiler/experiments/prepare_core_mix.py` (wiki, books,
   oasst, ultrachat, fineweb, qqp; evidence pairs with `evidence_parallel.py`), then
   `prepare_core_mix.py mix data_core/src_p50 data_core/mix_P100 --mb 3300 --recipe P100`.
2. The 5M core (resumable; `--resume` continues from the last checkpoint):
   ```bash
   N="$(pwd -W)/data/cv0.exe"
   .venv/Scripts/python.exe -B compiler/experiments/scale_run.py data_core/mix_P100 runs/core_mt5m --native "$N" --device cuda --amp \
       --sizes custom --shape 256,6,8 --block modern --configs v0_8k --ctx 1024 --batch 32 --optimizer muon --schedule wsd --fused \
       --max-steps 42538 --budgets 20000 --patience 0 --eval-every 600 --no-wiki --resume --out results/core_mt5m_5M.json
   ```
   (the research run reused the P100 core's dictionary, `runs/core_P100/v0_8k_plain.cv0d`).
3. Chat data: `prepare_qa.py` (Alpaca -> `data_qa`), `prepare_dialogue.py` (-> `data_mt`), `gen_skills_v4.py` (skills),
   `python chat22/data/build_data.py` (-> `data_chat22`: rehearsal sample, long pools, dialogues, held-out text).
4. Real data: `python -m chat22.data.build_real` (-> `data_real/`: real Q&A, look-alikes, held-out sets),
   `chat22/data/build_knowledge.py` (the permanent store), `python -m chat22.data.plan_real` (passes per size).

## 6. Queued research

Sessions are shell scripts in `compiler/experiments/chains/`; each skips steps whose result file exists, so a chain can be
rerun after an interruption. `watch.sh RESULT` returns when that result appears, a run log shows a traceback, no Python
process has run for 10 minutes, or the logs have not been written for 30 minutes (a frozen job).
