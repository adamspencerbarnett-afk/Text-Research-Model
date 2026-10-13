#!/bin/bash
# Session 26: copy under rewording with meaning-equivalent substitutions (--perturb-synonyms; build_synonyms:
# 1,352 question words with substitutes from the dense key, e.g. food -> meal, famous -> renowned,
# child -> kid; names never substituted). Otherwise the best chat recipe (session 23 B: dense retrieval in
# training + --perturb-self 0.3), two seeds, against its two seeds (probe 24 and 23/32, reworded 5/5, 4/5).
# Decides whether paraphrase-like rewording beats random edits for reworded and taught-reworded questions.
# bf16 training, bf16 answers, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
for SEED in 1 2; do
  T=densesyn512_s$SEED; Q=runs/s2b_$T; mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
  [ -f results/s2b_$T.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
      --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
      --dense 2 --dense-train --perturb-self 0.3 --perturb-synonyms --amp --seed $SEED --rehearse data_core/mix_D3s_5M \
      --out results/s2b_$T.json --save $Q/qa_s2b_$T.pt > $Q.log 2>&1
  [ -f results/probe_qa_s2b_$T.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_$T.pt --device cuda --graph --precision bf16 > runs/probe_s2b_$T.log 2>&1
  [ -f results/probe_qa_s2b_${T}_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_$T.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_${T}_int8.log 2>&1
done
ls results/s2b_densesyn512_s*.json results/probe_qa_s2b_densesyn512_s*.json
