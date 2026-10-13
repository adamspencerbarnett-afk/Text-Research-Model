#!/bin/bash
# Session 27: the 100M core (HYBRID_DESIGN 3.5, research item 20), then the adopted Q&A stage on it.
# Why: three separate text gains at 5M (modern block, Muon + WSD, depth) left answers unchanged, and
# reading rose 1% -> 28% exact match from 5M to 50M; the generator's size is the next lever.
#  1  core: 1,024 wide, 8 layers, 16 heads (109M), modern block, Muon + warmup-stable-decay, fused AdamW,
#     bf16, window 1,024, batch 32, 61,035 steps = 2 billion codes (~20 per parameter), P100 mix (3.3 GB,
#     no worked maths), no count tables (they hurt from 20M). Measured 51,000 codes/s, 18.3 GB: ~11 h.
#     Resumable: a checkpoint every 30 minutes; rerunning this script continues the run.
#  2  report card (bf16 weights).
#  3  Q&A stage, exactly the adopted 5M recipe so the comparison is size alone: dense retrieval in the
#     loop, 30% reworded self windows, bf16; then the chat probe at bf16 and at int8.
# Compared with the 5M models (probe 24 and 23/32) and the 50M S2b model (21/32; 24 with the gate).
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f data_core/mix_P100/manifest.json ]; do sleep 60; done
C=runs/core_P100; mkdir -p $C
[ -f results/core_P100_100M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_P100 $C --native "$N" --device cuda --amp \
    --sizes custom --shape 1024,8,16 --block modern --configs v0_8k --ctx 1024 --batch 32 \
    --optimizer muon --schedule wsd --fused --max-steps 61035 --budgets 80000 --patience 0 --eval-every 600 \
    --no-wiki --resume --checkpoint-every 1800 --out results/core_P100_100M.json > $C.log 2>&1
[ -f results/core_eval_core_P100_scale_custom_v0_8k.json ] || $PY -B compiler/experiments/core_eval.py $C/scale_custom_v0_8k.pt \
    --data data_core/mix_D3_5M --device cuda --precision bf16 > runs/core_eval_P100.log 2>&1
Q=runs/s2b_100m; mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
[ -f results/s2b_100m.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 14400 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict $C/v0_8k_plain.cv0d --init $C/scale_custom_v0_8k.pt --block modern --width 1024 --layers 8 --heads 16 --ctx 1024 --neighbor-codes 400 \
    --dense 2 --dense-train --perturb-self 0.3 --amp --rehearse data_core/mix_P100 --out results/s2b_100m.json --save $Q/qa_s2b_100m.pt > $Q.log 2>&1
[ -f results/probe_qa_s2b_100m.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_100m.pt --device cuda --graph --precision bf16 > runs/probe_s2b_100m.log 2>&1
[ -f results/probe_qa_s2b_100m_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_100m.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_100m_int8.log 2>&1
ls results/core_P100_100M.json results/core_eval_core_P100_*.json results/s2b_100m.json results/probe_qa_s2b_100m*.json
