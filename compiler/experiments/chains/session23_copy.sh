#!/bin/bash
# Session 23 (after session 22): copying under rewording, and reranking by the core.
# Finding behind it (13 October): even with dense retrieval in training, the model does not copy the
# right answer when the stored question is worded differently ("graphics card" vs "GPU"). In training it
# only ever copies from windows holding its own word-identical question; retrieved neighbours are other
# exchanges whose answers are wrong to copy. So it learned "copy only on an exact match".
#  A  reranking by the core (HYBRID_DESIGN 3.3, item 15), answer time only: the dense-trained model (R2) with
#     its top 10 candidates reordered by its own surprise at the question; Q&A groups and chat probe.
#  B  copy under rewording: the R2 recipe plus --perturb-self 0.3 (30% of self windows show the right
#     exchange under a reworded question: words dropped, swapped or replaced). Decides whether reworded and
#     taught-reworded questions improve without losing stored and added copying.
# bf16 training, bf16 answers, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/core_eval_core_muon_wsd512_s2_scale_5M_v0_8k_ng64.json ] || grep -lq Traceback runs/s2b_amp512.log runs/s2b_bad512.log runs/core_muon_wsd512_s2.log 2>/dev/null; do sleep 120; done
# A
D=runs/rerank_a; mkdir -p $D && cp -n runs/prec_5m_fp32/neighbors_3c515ad55740504b.json $D/ 2>/dev/null
[ -f results/rerank_a_eval.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $D --native "$N" --device cuda --budget 0 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --eos --loss-answer-only --dense 2 --rerank-core 10 --precision bf16 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/s2b_dense512/qa_s2b_dense512.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --out results/rerank_a_eval.json > $D.log 2>&1
[ -f results/probe_qa_s2b_dense512_rerank.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_dense512/qa_s2b_dense512.pt --device cuda --graph --rerank-core 10 --precision bf16 --tag _rerank > runs/probe_rerank_a.log 2>&1
# B
Q=runs/s2b_densep512; mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
[ -f results/s2b_densep512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --dense 2 --dense-train --perturb-self 0.3 --amp --rehearse data_core/mix_D3s_5M --out results/s2b_densep512.json --save $Q/qa_s2b_densep512.pt > $Q.log 2>&1
[ -f results/probe_qa_s2b_densep512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_densep512.pt --device cuda --graph --precision bf16 > runs/probe_s2b_densep512.log 2>&1
[ -f results/probe_qa_s2b_densep512_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_densep512.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_densep512_int8.log 2>&1
ls results/rerank_a_eval.json results/probe_qa_s2b_dense512_rerank.json results/s2b_densep512.json results/probe_qa_s2b_densep512*.json
