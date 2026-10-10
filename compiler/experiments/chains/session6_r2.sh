#!/bin/bash
# Session 6: after R1d, R2 (SQuAD span answers, the cleanest copy-head test). Run from the repository root.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/r1d_alpaca_19M_init.json ] || grep -q Traceback runs/r1d_19M.log 2>/dev/null; do sleep 30; done
mkdir -p runs/r2
[ -f results/r2_squad_5M_init.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_squad runs/r2 --native "$N" --device cuda \
    --dict runs/scale/v0_8k_plain.cv0d --init runs/scale/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 256 \
    --neighbor-codes 0 --batch 32 --budget 900 --patience 3 --eval-n 200 --copy-head --copy-question --greedy \
    --out results/r2_squad_5M_init.json --save runs/r2/squad_5M_init.pt > runs/r2_squad.log 2>&1
grep -l Traceback runs/r2_squad.log; ls results/r2_squad_5M_init.json
