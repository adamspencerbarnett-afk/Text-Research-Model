#!/bin/bash
# Session 18 (after the 160x16 point of the depth sweep): the depth sweep stops at 16 layers (11 layers already
# lost to 6 at equal steps, so 128x24 is dropped), and the best core by every language measure, the modern
# 256x6 block, gets what every other core had: the Alpaca Q&A stage (S2b recipe, 15,565 steps), the chat probe
# and the GSM8K stage (8,000 steps, calculator). Decides whether the modern block's lower error reaches answers.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/gsm_d160x16.json ] || grep -lq Traceback runs/core_d160x16.log runs/gsm_d160x16.log 2>/dev/null; do sleep 120; done
# stop session 17 before (or just after) it starts 128x24
powershell -NoProfile -Command "Get-CimInstance Win32_Process | Where-Object { \$_.CommandLine -like '*session17_depth*' -or \$_.CommandLine -like '*core_d128x24*' } | ForEach-Object { Stop-Process -Id \$_.ProcessId -Force }" 2>/dev/null
sleep 5
C=runs/core_modern512; I=$C/scale_5M_v0_8k_ng64.pt; K=$C/v0_8k_plain.cv0d
Q=runs/s2b_modern512; mkdir -p $Q
[ -f results/s2b_modern512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict $K --init $I --block modern --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --rehearse data_core/mix_D3s_5M --out results/s2b_modern512.json --save $Q/qa_s2b_modern512.pt > $Q.log 2>&1
[ -f results/probe_qa_s2b_modern512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_modern512.pt --device cuda > runs/probe_s2b_modern512.log 2>&1
G=runs/gsm_modern512; mkdir -p $G
[ -f results/gsm_modern512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm $G --native "$N" --device cuda --batch 32 --max-steps 8000 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --copy-question --calculator --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine \
    --loss-answer-only --eos --rehearse-p 0.3 --dict $K --init $I --block modern --width 256 --layers 6 --heads 8 \
    --ctx 512 --neighbor-codes 250 --rehearse data_core/mix_D3s_5M --out results/gsm_modern512.json --save $G/qa.pt > $G.log 2>&1
ls results/s2b_modern512.json results/probe_qa_s2b_modern512.json results/gsm_modern512.json
