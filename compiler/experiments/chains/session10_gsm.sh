#!/bin/bash
# Session 10c (GPU free while E1 runs on the CPU): math under the new recipe. GSM8K on the 512 core
# (MetaMath was in its stage-1 data), S2b Q&A recipe, copy from the question, calculator + answer check.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
mkdir -p runs/s2b_gsm
[ -f results/s2b_gsm_512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm runs/s2b_gsm --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --copy-question --calculator --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine \
    --loss-answer-only --eos --rehearse-p 0.3 --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt \
    --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 --rehearse data_core/mix_D3s_5M --out results/s2b_gsm_512.json --save runs/s2b_gsm/qa_s2b_gsm_512.pt > runs/s2b_gsm.log 2>&1
ls results/s2b_gsm_512.json
