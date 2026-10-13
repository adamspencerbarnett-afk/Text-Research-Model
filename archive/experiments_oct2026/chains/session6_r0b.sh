#!/bin/bash
# Session 6, after R2: R1e (baseline on the parse-fixed Alpaca data) and R0b (+ reworded-question
# training and the similarity input), same settings otherwise; then the chat probe of both.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/r2_squad_5M_init.json ] || grep -q Traceback runs/r2_squad.log 2>/dev/null; do sleep 30; done
COMMON=(--native "$N" --device cuda --dict runs/scale/v0_8k_plain.cv0d --init runs/scale/scale_5M_v0_8k_ng64.pt
        --width 256 --layers 6 --heads 8 --ctx 256 --neighbor-codes 110 --batch 32 --budget 600 --patience 3
        --self-context-p 0.5 --eval-n 200 --copy-head --greedy --no-repeat 3)
mkdir -p runs/r1e runs/r0b
[ -f results/r1e_alpaca_5M_parsefix.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/r1e "${COMMON[@]}" \
    --out results/r1e_alpaca_5M_parsefix.json --save runs/r1e/qa_r1e_5M.pt > runs/r1e.log 2>&1
[ -f results/r0b_alpaca_5M_reworded.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/r0b "${COMMON[@]}" --perturb-self 0.5 --sim-feature \
    --out results/r0b_alpaca_5M_reworded.json --save runs/r0b/qa_r0b_5M.pt > runs/r0b.log 2>&1
for m in runs/r1e/qa_r1e_5M.pt runs/r0b/qa_r0b_5M.pt; do [ -f "$m" ] && $PY -B compiler/experiments/chat_probe.py "$m" --device cuda > "runs/probe_$(basename "$m" .pt).log" 2>&1; done
grep -l Traceback runs/r1e.log runs/r0b.log runs/probe_*.log
ls results/r1e_alpaca_5M_parsefix.json results/r0b_alpaca_5M_reworded.json results/probe_qa_r1e_5M.json results/probe_qa_r0b_5M.json
