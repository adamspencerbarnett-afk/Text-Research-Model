#!/bin/bash
# Skills lab, round 2 (CPU, beside the GPU queue; Adam, 14 October: side quest, nothing adopted without a clear win):
#  K2 on the 5M chat core (lab copy, skills_lab/models/k2_5m_lab), decision heads, and the skills test with each.
PY=.venv/Scripts/python.exe
until [ -f skills_lab/results/ft_control.json ] && [ -f skills_lab/results/ft_skills.json ] && [ -f skills_lab/results/decision_base5M_k1.json ] \
      && ! ps -W | grep -q "train_skills\|eval_skills"; do sleep 60; done
sleep 120
K=skills_lab/models/k2_5m_lab/key_head.pt
[ -f $K ] || $PY -B compiler/experiments/k2_question_key.py runs/core_D3s512/scale_5M_v0_8k_ng64.pt --tag 5m_lab --device cpu \
    --out-dir skills_lab/models > skills_lab/results/k2_5m_lab.log 2>&1
cp -n skills_lab/models/k2_question_key_5m_lab.json skills_lab/results/ 2>/dev/null
dh() { [ -f skills_lab/results/decision_$1.json ] || $PY -B skills_lab/decision.py "${@:2}" --name $1 > skills_lab/results/decision_$1.log 2>&1; }
dh ftskills_k3 skills_lab/models/ft_skills.pt --mode k3close
dh ftskills_k3_k2 skills_lab/models/ft_skills.pt --mode k3close --key-head $K
dh base5M_k1_k2 models/qa_best_5M_dense.pt --mode k1 --key-head $K
ev() { [ -f skills_lab/results/$1.json ] || $PY -B skills_lab/eval_skills.py "${@:2}" --name $1 > skills_lab/results/$1.log 2>&1; }
ev r2_base5M_head models/qa_best_5M_dense.pt --modes k1 --decision skills_lab/models/decision_base5M_k1.pt
ev r2_base5M_k2_head models/qa_best_5M_dense.pt --modes k1 --key-head $K --decision skills_lab/models/decision_base5M_k1_k2.pt
ev r2_ftskills_head skills_lab/models/ft_skills.pt --modes k3close --decision skills_lab/models/decision_ftskills_k3.pt
ev r2_ftskills_k2_head skills_lab/models/ft_skills.pt --modes orig,k3close --key-head $K --decision skills_lab/models/decision_ftskills_k3_k2.pt
ev r2_base5M_literal models/qa_best_5M_dense.pt --modes orig --dense 0
touch skills_lab/results/round2_done
# round 3: training version 2 (balanced: general questions answered beside unrelated evidence, more answerable and
# clarifying windows), then its decision head and scores
[ -f skills_lab/models/ft_skills2.pt ] || $PY -B skills_lab/train_skills.py --init models/qa_best_5M_dense.pt --save skills_lab/models/ft_skills2.pt \
    --steps 3000 --threads 16 --skill-p 0.35 --k-p 0.5 --gen-version 2 --unrelated-p 0.2 > skills_lab/results/train_ft_skills2.log 2>&1
ev ft_skills2 skills_lab/models/ft_skills2.pt --modes orig,k1,k3close
dh ftskills2_k3 skills_lab/models/ft_skills2.pt --mode k3close
ev r3_ftskills2_head skills_lab/models/ft_skills2.pt --modes k3close --decision skills_lab/models/decision_ftskills2_k3.pt
touch skills_lab/results/round3_done
