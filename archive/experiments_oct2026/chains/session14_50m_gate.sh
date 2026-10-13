#!/bin/bash
# Session 14: the 50M network is now competent alone, and weak retrievals hurt it.
# 14a: chat probe of the current 50M model using memory only for exact or strong (>= 0.5) matches (no retraining).
# 14b: the 50M Q&A stage again, full 15,565 steps, with the retrieval similarity as a trust-head input and
#      20% of non-self windows trained with nothing retrieved; then its probe, plain and with the 0.5 gate.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
[ -f results/probe_qa_s2b_50m_gate05.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_50m/qa_s2b_50m.pt --device cuda --min-sim 0.5 --tag _gate05 > runs/probe_s2b_50m_gate05.log 2>&1
mkdir -p runs/s2c_50m
[ -f results/s2c_50m.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/s2c_50m --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 16000 \
    --self-context-p 0.5 --no-retrieval-p 0.2 --sim-feature --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine \
    --loss-answer-only --eos --rehearse-p 0.3 --dict runs/core_P50/v0_8k_plain.cv0d --init runs/core_P50/scale_50M_v0_8k.pt \
    --width 640 --layers 10 --heads 10 --ctx 1024 --neighbor-codes 400 --rehearse data_core/mix_P50 --out results/s2c_50m.json --save runs/s2c_50m/qa_s2c_50m.pt > runs/s2c_50m.log 2>&1
[ -f results/probe_qa_s2c_50m.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2c_50m/qa_s2c_50m.pt --device cuda > runs/probe_s2c_50m.log 2>&1
[ -f results/probe_qa_s2c_50m_gate05.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2c_50m/qa_s2c_50m.pt --device cuda --min-sim 0.5 --tag _gate05 > runs/probe_s2c_50m_gate05.log 2>&1
ls results/probe_qa_s2b_50m_gate05.json results/s2c_50m.json results/probe_qa_s2c_50m*.json
