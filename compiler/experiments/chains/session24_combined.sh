#!/bin/bash
# Session 24 (after session 23): the two training lessons together, on dense retrieval.
#  Bad evidence (session 22 B) taught the model when NOT to copy: network alone +20%, stored 0.901 / 62%,
#  but reworded questions fell. Copy under rewording (session 23 B, --perturb-self) targets the opposite
#  case: the right answer under a reworded question. This run teaches both at once, with dense retrieval in
#  the loop (session 21 R2), so the window always holds the best retrieved exchange, sometimes a wrong one,
#  sometimes nothing, and sometimes the right one reworded. Compared with R2 (s2b_dense512), 22 B and 23 B.
# bf16 training, bf16 answers, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_densep512_int8.json ] || grep -lq Traceback runs/rerank_a.log runs/s2b_densep512.log 2>/dev/null; do sleep 120; done
Q=runs/s2b_all512; mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
[ -f results/s2b_all512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --dense 2 --dense-train --perturb-self 0.3 --no-retrieval-p 0.2 --distractor-p 0.25 --amp --rehearse data_core/mix_D3s_5M \
    --out results/s2b_all512.json --save $Q/qa_s2b_all512.pt > $Q.log 2>&1
[ -f results/probe_qa_s2b_all512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_all512.pt --device cuda --graph --precision bf16 > runs/probe_s2b_all512.log 2>&1
[ -f results/probe_qa_s2b_all512_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_all512.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_all512_int8.log 2>&1
[ -f results/probe_qa_s2b_all512_rerank.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_all512.pt --device cuda --graph --precision bf16 --rerank-core 10 --tag _rerank > runs/probe_s2b_all512_rerank.log 2>&1
ls results/s2b_all512.json results/probe_qa_s2b_all512*.json
