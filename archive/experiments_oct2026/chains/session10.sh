#!/bin/bash
# Session 10: confirm S2b-512 with a second seed (the protocol rule before building on a winner).
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
mkdir -p runs/s2b_512_s2
[ -f results/s2b_512_s2.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/s2b_512_s2 --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --rehearse data_core/mix_D3s_5M --seed 2 --out results/s2b_512_s2.json --save runs/s2b_512_s2/qa_s2b_512_s2.pt > runs/s2b_512_s2.log 2>&1
[ -f results/probe_qa_s2b_512_s2.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_512_s2/qa_s2b_512_s2.pt --device cuda > runs/probe_s2b_512_s2.log 2>&1
ls results/s2b_512_s2.json results/probe_qa_s2b_512_s2.json
