#!/bin/bash
# Session 20 (after session 19): speed and the training recipe. No math stages (math paused 12 October).
# A. Clean speed and energy of the CUDA-graph answer step (fast_decode.py) on an idle GPU: the chat
#    probe with --graph for the best 5M and the 50M Q&A models (replies identical to the eager loop).
# B. The modern training recipe on the 5M core, at the same mix, dictionary, window and 42,538 steps
#    as the AdamW + cosine core (D3s512): Muon, warmup-stable-decay, and both. Fused AdamW on all
#    three (same maths, fewer launches). Each core gets the report card, the Alpaca Q&A stage (S2b
#    recipe, unchanged) and the chat probe, so a recipe win is judged on answers, not only on text.
# Precision (12 October): training in bf16 (cores, and the Q&A stage with --amp); answers and report cards
# with bf16 weights, plus an int8 check of every new Q&A model. No fp32 or int4 runs from here on.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/core_eval_core_fp32_512_scale_5M_v0_8k_ng64.json ] || grep -lq Traceback runs/core_fp32_512.log runs/core_eval_fp32_512.log 2>/dev/null; do sleep 120; done
# A
[ -f results/probe_qa_best_5M_512_graph.json ] || $PY -B compiler/experiments/chat_probe.py models/qa_best_5M_512.pt --device cuda --graph --precision bf16 --tag _graph > runs/probe_graph_5m.log 2>&1
[ -f results/probe_qa_s2b_50m_graph.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_50m/qa_s2b_50m.pt --device cuda --graph --precision bf16 --tag _graph > runs/probe_graph_50m.log 2>&1
# B
for R in muon:cosine adamw:wsd muon:wsd; do
  O=${R%%:*}; S=${R#*:}; T=${O}_${S}; C=runs/core_${T}512; Q=runs/s2b_${T}512
  mkdir -p $C $Q && cp -n runs/core_D3s512/v0_8k_plain.cv0d $C/v0_8k_plain.cv0d
  [ -f results/core_${T}512_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M $C --native "$N" --device cuda --amp \
      --sizes 5M --configs v0_8k_ng64 --budgets 7200 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 \
      --optimizer $O --schedule $S --fused --out results/core_${T}512_5M.json > $C.log 2>&1
  [ -f results/core_eval_core_${T}512_scale_5M_v0_8k_ng64.json ] || $PY -B compiler/experiments/core_eval.py $C/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda --precision bf16 > runs/core_eval_${T}512.log 2>&1
  [ -f results/s2b_${T}512.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
      --dict $C/v0_8k_plain.cv0d --init $C/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
      --amp --rehearse data_core/mix_D3s_5M --out results/s2b_${T}512.json --save $Q/qa_s2b_${T}512.pt > $Q.log 2>&1
  [ -f results/probe_qa_s2b_${T}512.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_${T}512.pt --device cuda --graph --precision bf16 > runs/probe_s2b_${T}512.log 2>&1
  [ -f results/probe_qa_s2b_${T}512_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_${T}512.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_${T}512_int8.log 2>&1
done
ls results/probe_*_graph.json results/core_*_cosine512_5M.json results/core_*_wsd512_5M.json results/s2b_*_wsd512.json results/s2b_muon_cosine512.json
