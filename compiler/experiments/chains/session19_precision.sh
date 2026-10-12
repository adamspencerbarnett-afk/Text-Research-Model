#!/bin/bash
# Session 19 (after session 18): precision. (1) Training precision: a 5M core trained in fp32 at the same mix,
# window and 42,538 steps as the bf16 core (D3s512), with its report card: does bf16 training cost anything?
# (2) Weight precision at answer time: the best 5M Q&A model and the 50M Q&A model with weights rounded to
# bf16, int8 and int4 (precision.py), scored on the Q&A groups (eval only) and the chat probe; the 5M core's
# report card at each precision. The GPU must be otherwise idle for the cost readings to be valid.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/gsm_modern512.json ] || grep -lq Traceback runs/s2b_modern512.log runs/gsm_modern512.log 2>/dev/null; do sleep 120; done
# (2) first: cheap, and it answers the deployment question
for P in fp32 bf16 int8 int4; do
  [ -f results/probe_qa_best_5M_512_$P.json ] || $PY -B compiler/experiments/chat_probe.py models/qa_best_5M_512.pt --device cuda --precision $P --tag _$P > runs/probe_prec_5m_$P.log 2>&1
  D=runs/prec_5m_$P; mkdir -p $D
  [ -f results/prec_5m_$P.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $D --native "$N" --device cuda --budget 0 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --eos --loss-answer-only --precision $P \
      --dict models/qa_best_5M_512.cv0d --init models/qa_best_5M_512.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
      --out results/prec_5m_$P.json > $D.log 2>&1
  [ -f results/core_eval_core_D3s512_scale_5M_v0_8k_ng64_$P.json ] || $PY -B compiler/experiments/core_eval.py runs/core_D3s512/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda --precision $P --tag _$P > runs/core_eval_prec_$P.log 2>&1
done
for P in bf16 int8 int4; do
  [ -f results/probe_qa_s2b_50m_$P.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_50m/qa_s2b_50m.pt --device cuda --precision $P --tag _$P > runs/probe_prec_50m_$P.log 2>&1
  D=runs/prec_50m_$P; mkdir -p $D
  [ -f results/prec_50m_$P.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $D --native "$N" --device cuda --budget 0 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --eos --loss-answer-only --precision $P \
      --dict runs/core_P50/v0_8k_plain.cv0d --init runs/s2b_50m/qa_s2b_50m.pt --width 640 --layers 10 --heads 10 --ctx 1024 --neighbor-codes 400 \
      --out results/prec_50m_$P.json > $D.log 2>&1
done
# (1) fp32 training at equal steps
C=runs/core_fp32_512; mkdir -p $C && cp -n runs/core_D3s512/v0_8k_plain.cv0d $C/v0_8k_plain.cv0d
[ -f results/core_fp32_512_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M $C --native "$N" --device cuda \
    --sizes 5M --configs v0_8k_ng64 --budgets 9000 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 \
    --out results/core_fp32_512_5M.json > $C.log 2>&1
[ -f results/core_eval_core_fp32_512_scale_5M_v0_8k_ng64.json ] || $PY -B compiler/experiments/core_eval.py $C/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_fp32_512.log 2>&1
ls results/probe_qa_best_5M_512_*.json results/prec_*.json results/core_eval_core_D3s512_*_*.json results/core_fp32_512_5M.json
