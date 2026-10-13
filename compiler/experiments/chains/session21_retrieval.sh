#!/bin/bash
# Session 21 (after session 20): is retrieval the limit on reworded and new questions?
# Dense question keys (dense_keys.py: all-MiniLM-L6-v2, a pretrained sentence encoder, ranked together
# with the literal key at weight 2, which was best on the 2,000 QQP rewordings: recall@1 75.1% -> 80.9%).
#  R1  answer-time only: the best 5M Q&A model unchanged, retrieval swapped (Q&A groups + chat probe).
#  R2  in the training loop: the S2b-512 Q&A stage again from the same D3s512 core, with training
#      neighbours retrieved densely too, so the network learns to copy from matches that share few words.
# Same data, split, seed and recipe as S2b-512, so the only change is retrieval; R2 trains in bf16 (--amp).
# Answers with bf16 weights plus an int8 check. No math, no fp32 or int4 runs.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_muon_wsd512_int8.json ] || grep -lq Traceback runs/core_muon_wsd512.log runs/s2b_muon_wsd512.log 2>/dev/null; do sleep 120; done
# R1
D=runs/retr_r1; mkdir -p $D
[ -f results/retr_r1_dense_eval.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $D --native "$N" --device cuda --budget 0 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --eos --loss-answer-only --dense 2 --precision bf16 \
    --dict models/qa_best_5M_512.cv0d --init models/qa_best_5M_512.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --out results/retr_r1_dense_eval.json > $D.log 2>&1
[ -f results/probe_qa_best_5M_512_dense.json ] || $PY -B compiler/experiments/chat_probe.py models/qa_best_5M_512.pt --device cuda --graph --dense 2 --precision bf16 --tag _dense > runs/probe_retr_r1.log 2>&1
# R2
Q=runs/s2b_dense512; mkdir -p $Q
[ -f results/s2b_dense512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --dense 2 --dense-train --amp --rehearse data_core/mix_D3s_5M --out results/s2b_dense512.json --save $Q/qa_s2b_dense512.pt > $Q.log 2>&1
[ -f results/probe_qa_s2b_dense512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_dense512.pt --device cuda --graph --precision bf16 > runs/probe_s2b_dense512.log 2>&1
[ -f results/probe_qa_s2b_dense512_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_dense512.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_dense512_int8.log 2>&1
ls results/retr_r1_dense_eval.json results/probe_qa_best_5M_512_dense.json results/s2b_dense512.json results/probe_qa_s2b_dense512*.json
