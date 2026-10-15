#!/bin/bash
# Session 33 (after the 100k skills arm): settle the multi-turn study (docs/MULTITURN_PLAN.md, decision rule 7).
# Session 31 gave no winner: history gain on 1,000 real conversations favoured the mixed (M) and sequential (S) arms,
# the 27-conversation probe favoured the control (Q); LoRA (L1/L2) lost 40%+ of the network-alone answers (dropped).
#  1. the 120-conversation probe v2 (mt_probe2.py: recall, correction, choosing between two people, returning after
#     three unrelated turns; reserved names and phrasings) on every session 31 arm and the old best 5M
#  2. seed 2 of Q, S (from Q seed 2) and M with every test (probe, probe v2, history gain, chat probe, typos, card)
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f skills_lab/results/gpu_skills_done ]; do sleep 300; done
# wait until no Python job uses the GPU (Windows always lists desktop programs such as Explorer as GPU users)
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 120; done
p2() { [ -f results/mtprobe2_$(basename $1 .pt).json ] || $PY -B compiler/experiments/mt_probe2.py $1 --device cuda --graph --precision bf16 > runs/mtprobe2_$(basename $1 .pt).log 2>&1; }
for m in Q S M L1 L2; do p2 runs/mt_$m/qa_mt_$m.pt; done
p2 models/qa_best_5M_dense.pt
C=runs/core_mt5m; CK=$C/scale_custom_v0_8k.pt
E=data_core/eval_clean/evidence_clean.jsonl
QA=(--device cuda --batch 32 --budget 20000 --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3
    --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 --dict $C/v0_8k_plain.cv0d --block modern --width 256 --layers 6 --heads 8 --ctx 1024
    --neighbor-codes 250 --history-codes 400 --dense 2 --dense-train --perturb-self 0.3 --amp --track-gap --rehearse data_core/mix_P100 --seed 2)
stage() {    # $1 name, $2 init, $3 steps, then extra flags
  local T=$1 I=$2 STEPS=$3; shift 3; local Q=runs/mt_$T; mkdir -p $Q && cp -n runs/mt_Q/neighbors_*.json $Q/ 2>/dev/null
  [ -f results/mt_$T.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" "${QA[@]}" --init $I --max-steps $STEPS "$@" \
      --out results/mt_$T.json --save $Q/qa_mt_$T.pt > $Q.log 2>&1
}
probes() {   # $1 model, $2 name
  [ -f results/mtprobe_$2.json ] || $PY -B compiler/experiments/mt_probe.py $1 --device cuda --graph --precision bf16 > runs/mtprobe_$2.log 2>&1
  p2 $1
  [ -f results/probe_$2.json ] || $PY -B compiler/experiments/chat_probe.py $1 --device cuda --graph --precision bf16 > runs/probe_$2.log 2>&1
  [ -f results/probe_$2_typo.json ] || $PY -B compiler/experiments/chat_probe.py $1 --device cuda --graph --precision bf16 --typos 0.15 --tag _typo > runs/probe_$2_typo.log 2>&1
  [ -f results/core_eval_mt_${2#qa_mt_}_${2}_clean.json ] || $PY -B compiler/experiments/core_eval.py $1 --data data_core/eval_clean \
      --evidence-file $E --device cuda --precision bf16 --tag _clean > runs/card_$2.log 2>&1
}
stage Q2 $CK 15565;                                               probes runs/mt_Q2/qa_mt_Q2.pt qa_mt_Q2
stage S2 runs/mt_Q2/qa_mt_Q2.pt 8000 --mt-data data_mt --mt-p 0.6; probes runs/mt_S2/qa_mt_S2.pt qa_mt_S2
stage M2 $CK 23565 --mt-data data_mt --mt-p 0.2;                  probes runs/mt_M2/qa_mt_M2.pt qa_mt_M2
touch results/session33_done
ls results/mtprobe2_*.json results/mt_*2.json
