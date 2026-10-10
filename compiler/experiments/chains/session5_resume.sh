#!/bin/bash
# Resume chain after the power cut: R3c, R1b seed 2, R1d. Run from the repository root.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
COMMON=(--native "$N" --device cuda --dict runs/scale/v0_8k_plain.cv0d --ctx 256 --neighbor-codes 110 --batch 32
        --budget 600 --patience 3 --self-context-p 0.5 --copy-bias 0 --eval-n 200 --copy-head --greedy)
C5=(--init runs/scale/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8)
C19=(--init runs/scale/scale_20M_v0_8k_ng64.pt --width 448 --layers 8 --heads 8)
mkdir -p runs/r3c runs/r1b runs/r1d
[ -f results/r3c_gsm_5M_init_calc.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm runs/r3c "${COMMON[@]}" "${C5[@]}" --copy-question --calculator \
    --out results/r3c_gsm_5M_init_calc.json --save runs/r3c/gsm_5M_init.pt > runs/r3c_gsm.log 2>&1
[ -f results/r1b_alpaca_5M_init_seed2.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/r1b "${COMMON[@]}" "${C5[@]}" --seed 2 \
    --out results/r1b_alpaca_5M_init_seed2.json --save runs/r1b/alpaca_5M_init_s2.pt > runs/r1b_seed2.log 2>&1
[ -f results/r1d_alpaca_19M_init.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/r1d "${COMMON[@]}" "${C19[@]}" \
    --out results/r1d_alpaca_19M_init.json --save runs/r1d/alpaca_19M_init.pt > runs/r1d_19M.log 2>&1
grep -l Traceback runs/r3c_gsm.log runs/r1b_seed2.log runs/r1d_19M.log
ls results/r3c_gsm_5M_init_calc.json results/r1b_alpaca_5M_init_seed2.json results/r1d_alpaca_19M_init.json
