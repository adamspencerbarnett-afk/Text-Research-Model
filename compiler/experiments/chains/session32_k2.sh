#!/bin/bash
# Session 32 (after session 31): K2, our own learned question key, to replace the outside sentence encoder (MiniLM).
# Adam, 14 October: keep MiniLM for now, but retrieval should be our own. K2 = K1's design (head on a frozen core of ours)
# on stronger cores with in-domain pairs and near-miss negatives (compiler/experiments/k2_question_key.py).
#  1. K2 on the 100M core (quality) and on the 5M chat model's own core (speed): Quora recall@1, skills-test
#     retrieval@1 and ms per query, each against the literal key alone and literal + MiniLM
#  2. answers with each retrieval (skills test + 32-question chat probe, skills_lab/eval_skills.py, GPU):
#     best 5M chat model: literal only / MiniLM / K2-5M / K2-100M; 100M model: literal only / K2-100M;
#     the skills-lab model (skills_lab/models/ft_skills.pt, if trained): MiniLM / K2-5M
PY=.venv/Scripts/python.exe
until [ -f results/core_eval_mt_L2_qa_mt_L2_clean.json ] || grep -lq Traceback runs/mt_L2.log 2>/dev/null; do sleep 120; done
for C in "100m runs/core_P100/scale_custom_v0_8k.pt" "5m runs/core_D3s512/scale_5M_v0_8k_ng64.pt"; do
  set -- $C
  [ -f results/k2_question_key_$1.json ] || $PY -B compiler/experiments/k2_question_key.py $2 --tag $1 --device cuda > runs/k2_$1.log 2>&1
done
ev() {  # $1 name, $2 model, then flags
  local N=$1 M=$2; shift 2
  [ -f skills_lab/results/$N.json ] || $PY -B skills_lab/eval_skills.py $M --name $N --device cuda --modes orig,k3close "$@" > skills_lab/results/$N.log 2>&1
}
ev k2_best5M_literal models/qa_best_5M_dense.pt --dense 0
ev k2_best5M_minilm models/qa_best_5M_dense.pt --dense 2
ev k2_best5M_k2_5m models/qa_best_5M_dense.pt --key-head runs/k2_5m/key_head.pt
ev k2_best5M_k2_100m models/qa_best_5M_dense.pt --key-head runs/k2_100m/key_head.pt
ev k2_100M_literal runs/s2b_100m/qa_s2b_100m.pt --dense 0
ev k2_100M_k2_100m runs/s2b_100m/qa_s2b_100m.pt --key-head runs/k2_100m/key_head.pt
if [ -f skills_lab/models/ft_skills.pt ]; then
  ev k2_ftskills_minilm skills_lab/models/ft_skills.pt --dense 2
  ev k2_ftskills_k2_5m skills_lab/models/ft_skills.pt --key-head runs/k2_5m/key_head.pt
fi
touch results/session32_done
ls results/k2_question_key_*.json skills_lab/results/k2_*.json
