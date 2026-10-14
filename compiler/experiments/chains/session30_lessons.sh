#!/bin/bash
# Session 30 (after session 28): lessons from NVIDIA DLI "Fundamentals of Deep Learning", applied to the best
# 5M chat recipe (dense retrieval in training + 30% reworded self windows, bf16; probe 24 and 23/32 on two seeds).
# Every arm records the overfitting gap (--track-gap: 120 training exchanges scored as if held out), and every
# model gets the chat probe at bf16, at int8, and with realistic typos (--typos 0.15).
#  base    the recipe unchanged, with the gap tracked (control, and a third seed of the recipe's behaviour)
#  freeze  two-phase fine-tuning: the new copy and trust heads train alone for 1,000 steps, then the core joins
#  drop    residual dropout 0.1 (the Q&A stage reads its data ~10 times; the course pairs dropout with augmentation)
#  typo    realistic typing mistakes in 20% of training questions (the course's rule: augment only with variation
#          that occurs in reality)
# Plus the typo probe of the current best model (models/qa_best_5M_dense.pt) as the robustness baseline.
# No new core and no new data: each arm is one Q&A stage from the same D3s512 core. bf16, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_confall512_int8.json ] || grep -lq Traceback runs/s2b_conf512.log runs/s2b_confall512.log 2>/dev/null; do sleep 120; done
[ -f results/probe_qa_best_5M_dense_typo.json ] || $PY -B compiler/experiments/chat_probe.py models/qa_best_5M_dense.pt --device cuda --graph --precision bf16 --typos 0.15 --tag _typo > runs/probe_best_dense_typo.log 2>&1
for R in base: freeze:--freeze-core-steps,1000 drop:--dropout,0.1 typo:--typo-p,0.2; do
  T=${R%%:*}; X=${R#*:}; X=${X//,/ }; Q=runs/s2b_l${T}512
  mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
  [ -f results/s2b_l${T}512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
      --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
      --dense 2 --dense-train --perturb-self 0.3 --amp --track-gap $X --rehearse data_core/mix_D3s_5M \
      --out results/s2b_l${T}512.json --save $Q/qa_s2b_l${T}512.pt > $Q.log 2>&1
  for P in "bf16:" "int8:_int8" "bf16:_typo:--typos,0.15"; do
    PR=${P%%:*}; REST=${P#*:}; TAG=${REST%%:*}; EXTRA=""; [ "$REST" != "$TAG" ] && EXTRA=${REST#*:} && EXTRA=${EXTRA//,/ }
    [ -f results/probe_qa_s2b_l${T}512${TAG}.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_l${T}512.pt --device cuda --graph \
        --precision $PR $EXTRA --tag "$TAG" > runs/probe_s2b_l${T}512${TAG}.log 2>&1
  done
done
ls results/s2b_l*512.json results/probe_qa_s2b_l*512*.json results/probe_qa_best_5M_dense_typo.json
