#!/bin/bash
# Session 17 (after 16): depth at equal parameters. Same mix, dictionary, window (512) and steps as the best
# 5M core (256w x 6L = 4.7M): 384x3 (5.3M), 192x11 (4.9M), 160x16 (4.9M), 128x24 (4.7M). Each core gets the
# report card and the GSM8K stage (5M recipe, 8,000 steps, calculator) so depth is judged on math as well as text.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/core_eval_core_modern512_scale_5M_v0_8k_ng64.json ] || grep -lq Traceback runs/core_modern512.log 2>/dev/null; do sleep 120; done
for S in 384,3,8 192,11,6 160,16,5 128,24,4; do
  W=${S%%,*}; rest=${S#*,}; L=${rest%%,*}; H=${rest#*,}; D=runs/core_d${W}x${L}
  mkdir -p $D && cp -n runs/core_D3s512/v0_8k_plain.cv0d $D/v0_8k_plain.cv0d
  [ -f results/core_d${W}x${L}_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M $D --native "$N" --device cuda --amp \
      --sizes custom --shape $S --configs v0_8k_ng64 --budgets 7200 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 \
      --out results/core_d${W}x${L}_5M.json > $D.log 2>&1
  [ -f results/core_eval_core_d${W}x${L}_scale_custom_v0_8k_ng64.json ] || $PY -B compiler/experiments/core_eval.py $D/scale_custom_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_d${W}x${L}.log 2>&1
  G=runs/gsm_d${W}x${L}; mkdir -p $G
  [ -f results/gsm_d${W}x${L}.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm $G --native "$N" --device cuda --batch 32 --max-steps 8000 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --copy-question --calculator --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine \
      --loss-answer-only --eos --rehearse-p 0.3 --dict $D/v0_8k_plain.cv0d --init $D/scale_custom_v0_8k_ng64.pt --width $W --layers $L --heads $H \
      --ctx 512 --neighbor-codes 250 --rehearse data_core/mix_D3s_5M --out results/gsm_d${W}x${L}.json --save $G/qa.pt > $G.log 2>&1
done
ls results/core_d*_5M.json results/core_eval_core_d*.json results/gsm_d*.json
