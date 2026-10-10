#!/bin/bash
# Session 6, after R1f: R2b, SQuAD re-run with loss on the answer only and replies cut at the
# end of the span (R2 failed: rambling after the right span; passages memorised).
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_r1f_19M.json ] || grep -lq Traceback runs/r1f.log 2>/dev/null; do sleep 60; done
mkdir -p runs/r2b
[ -f results/r2b_squad_5M_answeronly.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_squad runs/r2b --native "$N" --device cuda \
    --dict runs/scale/v0_8k_plain.cv0d --init runs/scale/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 256 \
    --neighbor-codes 0 --batch 32 --budget 1800 --patience 4 --eval-n 200 --copy-head --copy-question --greedy \
    --loss-answer-only --stop-at-blank --out results/r2b_squad_5M_answeronly.json --save runs/r2b/squad_r2b_5M.pt > runs/r2b.log 2>&1
grep -l Traceback runs/r2b.log; ls results/r2b_squad_5M_answeronly.json
