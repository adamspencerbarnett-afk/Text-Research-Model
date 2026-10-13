# Setup and the full pipeline

*Current as of 13 October 2026. Replaces the September handoff (now in `archive/experiments_oct2026/`).*

Everything here runs on one machine with an NVIDIA GPU (the research used an RTX 3090, 24 GB) and
Windows or Linux. Corpora, checkpoints and run folders are not in git: `data*/`, `runs/` and `*.pt`
are ignored. Every run writes its result to `results/*.json`, and `docs/RUN_LEDGER.md` collects them.

## 1. Environment

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python.exe -m pip install -r requirements.txt
g++ -O2 -std=c++17 -o data/cv0.exe compiler/native/cv0.cpp      # the fast encoder; pass its ABSOLUTE path to --native
.venv/Scripts/python.exe -B -m unittest compiler/test_compiler_v0.py compiler/experiments/test_ladder.py
```

On Windows, paths with spaces must be quoted wherever `--native` is passed (the chains do this).

## 2. Source corpora

Each source becomes one file of documents separated by a form-feed line, in `data_core/src_p50/`:

```bash
P=.venv/Scripts/python.exe -I compiler/experiments/prepare_core_mix.py
$P wiki      ENWIK9.txt                    data_core/src_p50/wiki.txt
$P books     data_v2/train                 data_core/src_p50/books.txt
$P oasst     oasst_ready.trees.jsonl.gz    data_core/src_p50/oasst.txt
$P ultrachat ULTRACHAT_DIR                 data_core/src_p50/ultrachat.txt
$P fineweb   FINEWEB.parquet               data_core/src_p50/fineweb.txt
$P qqp       QQP_DIR                       data_core/src_p50/qqp.txt
# evidence pairs (paragraph + a related paragraph from another document), on 10 workers:
E=.venv/Scripts/python.exe -I compiler/experiments/evidence_parallel.py
$E prep  data_core/src_p50/wiki.txt,data_core/src_p50/fineweb.txt runs/evp --min-len 200 --max-len 900
$E work  runs/evp --worker 0 --of 10      # run workers 0..9 in parallel
$E merge runs/evp data_core/src_p50/evidence.txt data_core/eval/evidence_val_p50.jsonl --max-pairs 450000
```

Licences and sources are in `docs/DATASETS.md`. The worked-maths source (MetaMath) is not used from
12 October, when maths was paused.

## 3. Training mixes

```bash
$P mix data_core/src_p50 data_core/mix_D3s_5M --mb 300  --recipe D3     # the 5M research core
$P mix data_core/src_p50 data_core/mix_P100  --mb 3300 --recipe P100   # the 100M core (no maths)
```

A mix has `train/`, `heldout/` (1% of documents, by hash) and a `manifest.json` with sizes and
duplicate counts.

## 4. Stage 1: the core

The adopted recipe (two seeds each): bf16, modern block for 50M and up, Muon for the block matrices,
warmup-stable-decay schedule, fused AdamW for the rest, no count tables above 5M, resumable.

```bash
N="$(pwd -W)/data/cv0.exe"
.venv/Scripts/python.exe -B compiler/experiments/scale_run.py data_core/mix_P100 runs/core_P100 --native "$N" \
    --device cuda --amp --sizes custom --shape 1024,8,16 --block modern --configs v0_8k --ctx 1024 --batch 32 \
    --optimizer muon --schedule wsd --fused --max-steps 61035 --budgets 80000 --patience 0 --eval-every 600 \
    --no-wiki --resume --checkpoint-every 1800 --out results/core_P100_100M.json
.venv/Scripts/python.exe -B compiler/experiments/core_eval.py runs/core_P100/scale_custom_v0_8k.pt \
    --data data_core/mix_D3_5M --device cuda --precision bf16                   # the report card
```

`--resume` checkpoints every 30 minutes; running the same command again continues from the last
checkpoint with the same learning-rate curve.

## 5. Q&A data

```bash
curl -o alpaca_data.json https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json
.venv/Scripts/python.exe -I compiler/experiments/prepare_qa.py alpaca_data.json data_qa
```

## 6. Stage 2: the Q&A stage (the best chat recipe)

Dense retrieval in the training loop, and 30% of self windows showing the right answer under a reworded
question (session 23 B: probe 24 and 23/32 over two seeds at 5M):

```bash
.venv/Scripts/python.exe -B compiler/experiments/m2_qa.py data_qa runs/qa --native "$N" --device cuda --amp \
    --batch 32 --max-steps 15565 --budget 7200 --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy \
    --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse data_core/mix_D3s_5M --rehearse-p 0.3 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt \
    --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --dense 2 --dense-train --perturb-self 0.3 --out results/qa.json --save runs/qa/qa.pt
```

Options measured and kept available: `--no-retrieval-p 0.2 --distractor-p 0.25` (misleading evidence:
network alone +20%, reworded questions lower), `--perturb-synonyms` (paraphrase-like rewording: ties
random edits), `--rerank-core N` (rejected: lowers every answer score).

## 7. Evaluate and chat

```bash
.venv/Scripts/python.exe -B compiler/experiments/chat_probe.py runs/qa/qa.pt --device cuda --graph --precision bf16
.venv/Scripts/python.exe -B compiler/experiments/chat_probe.py runs/qa/qa.pt --device cuda --graph --precision int8 --tag _int8
.venv/Scripts/python.exe -I compiler/experiments/run_ledger.py        # regenerates docs/RUN_LEDGER.md
.venv/Scripts/python.exe run_model.py                                 # menu: train, chat, chat with the best model
```

In the chat, `/add question || answer` teaches a fact at once and `/load file.txt` teaches a file of them.

## 8. Queued research

Sessions are shell scripts in `compiler/experiments/chains/`; each skips steps whose result file
exists, so a chain can be rerun after an interruption. `watch.sh RESULT` exits when that result
appears, any run log shows a traceback, or no Python process has run for 10 minutes.
