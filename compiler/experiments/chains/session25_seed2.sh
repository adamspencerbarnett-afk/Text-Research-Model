#!/bin/bash
# Session 25: second seed of the best 5M chat model (session 23 B: dense retrieval in training +
# copy under rewording, --perturb-self 0.3), the protocol before building on a win (probe 24/32).
# bf16 training, bf16 answers, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
Q=runs/s2b_densep512_s2; mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
[ -f results/s2b_densep512_s2.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --dense 2 --dense-train --perturb-self 0.3 --amp --seed 2 --rehearse data_core/mix_D3s_5M --out results/s2b_densep512_s2.json --save $Q/qa_s2b_densep512_s2.pt > $Q.log 2>&1
[ -f results/probe_qa_s2b_densep512_s2.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_densep512_s2.pt --device cuda --graph --precision bf16 > runs/probe_s2b_densep512_s2.log 2>&1
[ -f results/probe_qa_s2b_densep512_s2_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_densep512_s2.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_densep512_s2_int8.log 2>&1
ls results/s2b_densep512_s2.json results/probe_qa_s2b_densep512_s2*.json
