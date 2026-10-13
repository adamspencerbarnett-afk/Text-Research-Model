#!/bin/bash
# Session 22 (after session 21): settle session 20's reading, then teach the model when not to copy.
#  A  control: the AdamW + cosine core (D3s512) through the unchanged S2b Q&A stage in bf16 (--amp). Every
#     session-20 recipe arm sat ~1.5 points below the fp32 baseline seeds on stored answers, and all of them
#     trained the Q&A stage in bf16 for the first time; this run says whether bf16 or the recipes caused it.
#  B  bad evidence (HYBRID_DESIGN 3.6): the same stage with 10% of windows carrying nothing and 10% carrying
#     a random wrong exchange (--no-retrieval-p 0.2 --distractor-p 0.25 of the non-self windows), compared
#     with A. Decides whether the model learns to ignore wrong evidence (new and reworded questions, network
#     alone) without losing copying (stored, added).
#  C  second seed of the Muon + warmup-stable-decay core with its report card, the protocol before
#     adopting it for the 100-300M cores.
# bf16 training, bf16 answers, int8 check. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_dense512_int8.json ] || grep -lq Traceback runs/retr_r1.log runs/s2b_dense512.log 2>/dev/null; do sleep 120; done
QA="--device cuda --batch 32 --max-steps 15565 --budget 3600 --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy \
    --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 --amp \
    --dict runs/core_D3s512/v0_8k_plain.cv0d --init runs/core_D3s512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 \
    --neighbor-codes 250 --rehearse data_core/mix_D3s_5M"
for R in amp: bad:--no-retrieval-p,0.2,--distractor-p,0.25; do
  T=${R%%:*}; X=${R#*:}; X=${X//,/ }; Q=runs/s2b_${T}512
  mkdir -p $Q && cp -n runs/prec_5m_fp32/neighbors_3c515ad55740504b.json $Q/ 2>/dev/null
  [ -f results/s2b_${T}512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" $QA $X --out results/s2b_${T}512.json --save $Q/qa_s2b_${T}512.pt > $Q.log 2>&1
  [ -f results/probe_qa_s2b_${T}512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_${T}512.pt --device cuda --graph --precision bf16 > runs/probe_s2b_${T}512.log 2>&1
  [ -f results/probe_qa_s2b_${T}512_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_${T}512.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_${T}512_int8.log 2>&1
done
C=runs/core_muon_wsd512_s2; mkdir -p $C && cp -n runs/core_D3s512/v0_8k_plain.cv0d $C/v0_8k_plain.cv0d
[ -f results/core_muon_wsd512_s2_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M $C --native "$N" --device cuda --amp \
    --sizes 5M --configs v0_8k_ng64 --budgets 7200 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 \
    --optimizer muon --schedule wsd --fused --seed 2 --out results/core_muon_wsd512_s2_5M.json > $C.log 2>&1
[ -f results/core_eval_core_muon_wsd512_s2_scale_5M_v0_8k_ng64.json ] || $PY -B compiler/experiments/core_eval.py $C/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda --precision bf16 > runs/core_eval_muon_wsd512_s2.log 2>&1
ls results/s2b_amp512.json results/s2b_bad512.json results/probe_qa_s2b_*512*.json results/core_muon_wsd512_s2_5M.json results/core_eval_core_muon_wsd512_s2_*.json
