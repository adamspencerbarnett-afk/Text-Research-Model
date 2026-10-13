#!/bin/bash
# Session 10b: K1 in use. Re-score the best model (S2b-512) with retrieval by literal + learned key; no retraining.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
mkdir -p runs/s2b_512_k1
[ -f results/s2b_512_k1.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/s2b_512_k1 --native "$N" --device cuda --budget 0 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --eos --loss-answer-only \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/s2b_512/qa_s2b_512.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --key-head runs/core_D3s512/key_head.pt --out results/s2b_512_k1.json --save runs/s2b_512_k1/qa_s2b_512_k1.pt > runs/s2b_512_k1.log 2>&1
[ -f results/probe_qa_s2b_512_k1.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_512_k1/qa_s2b_512_k1.pt --device cuda > runs/probe_s2b_512_k1.log 2>&1
ls results/s2b_512_k1.json results/probe_qa_s2b_512_k1.json
