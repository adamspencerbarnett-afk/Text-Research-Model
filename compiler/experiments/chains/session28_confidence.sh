#!/bin/bash
# Session 28 (after session 27): does the trust head need to see retrieval confidence? (HYBRID_DESIGN 3.6, item 3)
# Every stage-2 run so far trained with the similarity input off, so a wrong exchange and a reworded right
# one differed only in content; combining misleading evidence with copy-under-rewording then cancelled
# (probe 22/32). Two 5M runs with --sim-feature on:
#  A  the adopted recipe (dense retrieval in training + 30% reworded self windows), against its two seeds (24, 23/32)
#  B  the combined recipe (A + 10% nothing + 10% wrong exchange), against session 24 (22/32)
# bf16 training, bf16 answers, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_100m_int8.json ] || grep -lq Traceback runs/core_P100.log runs/core_eval_P100.log runs/s2b_100m.log 2>/dev/null; do sleep 120; done
# first: the 100M core's clean report card (data_core/eval_clean: no paragraph in any training mix), so it compares
# fairly with the 5M and 50M clean cards (chains/clean_rescore.sh)
[ -f results/core_eval_core_P100_scale_custom_v0_8k_clean.json ] || $PY -B compiler/experiments/core_eval.py runs/core_P100/scale_custom_v0_8k.pt     --data data_core/eval_clean --evidence-file data_core/eval_clean/evidence_clean.jsonl --device cuda --precision bf16 --tag _clean > runs/core_eval_P100_clean.log 2>&1
for R in conf: confall:--no-retrieval-p,0.2,--distractor-p,0.25; do
  T=${R%%:*}; X=${R#*:}; X=${X//,/ }; Q=runs/s2b_${T}512
  mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
  [ -f results/s2b_${T}512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
      --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
      --dense 2 --dense-train --perturb-self 0.3 --sim-feature $X --amp --rehearse data_core/mix_D3s_5M \
      --out results/s2b_${T}512.json --save $Q/qa_s2b_${T}512.pt > $Q.log 2>&1
  [ -f results/probe_qa_s2b_${T}512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_${T}512.pt --device cuda --graph --precision bf16 > runs/probe_s2b_${T}512.log 2>&1
  [ -f results/probe_qa_s2b_${T}512_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_${T}512.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_${T}512_int8.log 2>&1
done
ls results/s2b_conf512.json results/s2b_confall512.json results/probe_qa_s2b_conf*512*.json
