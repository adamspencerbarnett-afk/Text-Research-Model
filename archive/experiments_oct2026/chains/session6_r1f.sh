#!/bin/bash
# Session 6, after the R0b chain: R1f, the 19M core with R1e's settings and equal training
# (15,565 steps, the 4.7M runs' count; up to an hour), then its chat probe.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_r0b_5M.json ] || grep -lq Traceback runs/r1e.log runs/r0b.log runs/r2_squad.log 2>/dev/null; do sleep 60; done
mkdir -p runs/r1f
[ -f results/r1f_alpaca_19M_equalsteps.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/r1f --native "$N" --device cuda \
    --dict runs/scale/v0_8k_plain.cv0d --init runs/scale/scale_20M_v0_8k_ng64.pt --width 448 --layers 8 --heads 8 --ctx 256 \
    --neighbor-codes 110 --batch 32 --budget 3600 --max-steps 15565 --self-context-p 0.5 --eval-n 200 --copy-head --greedy --no-repeat 3 \
    --out results/r1f_alpaca_19M_equalsteps.json --save runs/r1f/qa_r1f_19M.pt > runs/r1f.log 2>&1
[ -f runs/r1f/qa_r1f_19M.pt ] && $PY -B compiler/experiments/chat_probe.py runs/r1f/qa_r1f_19M.pt --device cuda > runs/probe_qa_r1f_19M.log 2>&1
grep -l Traceback runs/r1f.log runs/probe_qa_r1f_19M.log
ls results/r1f_alpaca_19M_equalsteps.json results/probe_qa_r1f_19M.json
