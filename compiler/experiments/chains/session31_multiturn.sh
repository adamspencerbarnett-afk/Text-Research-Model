#!/bin/bash
# Session 31 (after session 30): which training structure holds a conversation best? docs/MULTITURN_PLAN.md
#  core  new 5M core: 256 x 6, modern block, Muon + WSD, bf16, P100 mix and dictionary, window 1,024, batch 32
#  Q     Q&A stage only (control)                                   15,565 steps
#  S     sequential: Q, then a multi-turn stage (60% conversation windows)        +8,000 steps
#  M     one mixed stage from the core (20% conversation windows: the same Q&A and conversation
#        exposure as Q + S)                                          23,565 steps
#  L     LoRA rank 16 on the frozen core: Q&A adapter (15,565), merged, then a multi-turn adapter (8,000)
# Every arm: dense retrieval in the loop, 30% reworded self windows, 30% rehearsal of core text, gap tracked,
# plus the options adopted from sessions 28 and 30 (chains/session31.env). Tests per model: multi-turn probe
# (with and without history, and history gain on 1,000 clean held-out conversations), chat probe, typo probe,
# Q&A groups (in the run), clean report card. bf16 throughout. No math.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_ltypo512_typo.json ] || grep -lq Traceback runs/s2b_l*512.log 2>/dev/null; do sleep 120; done
source compiler/experiments/chains/session31.env
# baseline: the current best 5M chat model, never trained on conversations (CPU smoke test: 5/22, history gain 4.3%)
[ -f results/mtprobe_qa_best_5M_dense.json ] || $PY -B compiler/experiments/mt_probe.py models/qa_best_5M_dense.pt --device cuda --graph --precision bf16 > runs/mtprobe_best_dense.log 2>&1
C=runs/core_mt5m; mkdir -p $C && cp -n runs/core_P100/v0_8k_plain.cv0d $C/v0_8k_plain.cv0d
[ -f results/core_mt5m_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_P100 $C --native "$N" --device cuda --amp \
    --sizes custom --shape 256,6,8 --block modern --configs v0_8k --ctx 1024 --batch 32 --optimizer muon --schedule wsd --fused \
    --max-steps 42538 --budgets 20000 --patience 0 --eval-every 600 --no-wiki --resume --out results/core_mt5m_5M.json >> $C.log 2>&1
CK=$C/scale_custom_v0_8k.pt
E=data_core/eval_clean/evidence_clean.jsonl
card() { [ -f results/core_eval_$(basename $(dirname $1))_$(basename $1 .pt)_clean.json ] || $PY -B compiler/experiments/core_eval.py $1 --data data_core/eval_clean \
           --evidence-file $E --device cuda --precision bf16 --tag _clean > runs/card_$(basename $1 .pt).log 2>&1; }
probes() {   # $1 model, $2 name
  [ -f results/mtprobe_$2.json ] || $PY -B compiler/experiments/mt_probe.py $1 --device cuda --graph --precision bf16 > runs/mtprobe_$2.log 2>&1
  [ -f results/probe_$2.json ] || $PY -B compiler/experiments/chat_probe.py $1 --device cuda --graph --precision bf16 > runs/probe_$2.log 2>&1
  [ -f results/probe_$2_typo.json ] || $PY -B compiler/experiments/chat_probe.py $1 --device cuda --graph --precision bf16 --typos 0.15 --tag _typo > runs/probe_$2_typo.log 2>&1
  card $1
}
card $CK
QA=(--device cuda --batch 32 --budget 20000 --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3
    --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 --dict $C/v0_8k_plain.cv0d --block modern --width 256 --layers 6 --heads 8 --ctx 1024
    --neighbor-codes 250 --history-codes 400 --dense 2 --dense-train --perturb-self 0.3 --amp --track-gap --rehearse data_core/mix_P100)
stage() {    # $1 name, $2 init, $3 steps, then extra flags
  local T=$1 I=$2 STEPS=$3; shift 3; local Q=runs/mt_$T; mkdir -p $Q && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
  [ -f results/mt_$T.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" "${QA[@]}" --init $I --max-steps $STEPS $EXTRA "$@" \
      --out results/mt_$T.json --save $Q/qa_mt_$T.pt > $Q.log 2>&1
}
stage Q $CK 15565;                                              probes runs/mt_Q/qa_mt_Q.pt qa_mt_Q
stage S runs/mt_Q/qa_mt_Q.pt 8000 --mt-data data_mt --mt-p 0.6; probes runs/mt_S/qa_mt_S.pt qa_mt_S
stage M $CK 23565 --mt-data data_mt --mt-p 0.2;                 probes runs/mt_M/qa_mt_M.pt qa_mt_M
stage L1 $CK 15565 --lora-rank 16;                              probes runs/mt_L1/qa_mt_L1.pt qa_mt_L1
stage L2 runs/mt_L1/qa_mt_L1.pt 8000 --lora-rank 16 --mt-data data_mt --mt-p 0.6; probes runs/mt_L2/qa_mt_L2.pt qa_mt_L2
ls results/core_mt5m_5M.json results/mt_*.json results/mtprobe_qa_mt_*.json
