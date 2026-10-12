#!/bin/bash
# Session 16 (after session 15 and the idle speed measurement): the modern block on real data.
# Same mix, dictionary, size, window and steps as the best 5M core (D3s512); then its report card.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f runs/kv_speed_5m_idle.log ] && grep -q speedup runs/kv_speed_5m_idle.log 2>/dev/null; do sleep 120; done
mkdir -p runs/core_modern512 && cp -n runs/core_D3s512/v0_8k_plain.cv0d runs/core_modern512/v0_8k_plain.cv0d
[ -f results/core_modern512_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M runs/core_modern512 --native "$N" --device cuda --amp \
    --sizes 5M --configs v0_8k_ng64 --budgets 5400 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 --block modern \
    --out results/core_modern512_5M.json > runs/core_modern512.log 2>&1
[ -f results/core_eval_core_modern512_scale_5M_v0_8k_ng64.json ] || $PY -B compiler/experiments/core_eval.py runs/core_modern512/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_modern512.log 2>&1
ls results/core_modern512_5M.json results/core_eval_core_modern512_*.json
