#!/bin/bash
# Session 34 (after session 33; Adam, 15 October): three lines queued together.
#  1. Skills arm v4: rewriting ("answer in its own words, in the form asked"): data_skills/train_v4.jsonl, 130,000
#     examples (v3 kinds + short answers, yes/no with true or false claims, lists, and 'your' instead of 'my');
#     ~11,600 steps of 32 from the best 5M chat model (~one pass over the skill data); scored on test v1, held-out v2 and
#     the held-out rewriting test (data_skills/test_v4_rewrite.json)
#  2. K2 version 2 (compiler/experiments/k2v2_question_key.py): the core's top 2 blocks trained with the key head, on
#     outcome pairs (which stored exchange lowers the true answer's bits, found by our own word key) + invented facts
#     + Quora; 5M chat core and 100M core; then answers with each key
#  3. Grounded scorecard (skills_lab/grounded.py): where each answer came from and whether it made things up, for the
#     best 5M, 100M, the skills arms (v3, v4) and the multi-turn arms S and M
PY=.venv/Scripts/python.exe
until [ -f results/session33_done ]; do sleep 300; done
# wait until no Python job uses the GPU (Windows always lists desktop programs such as Explorer as GPU users)
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 120; done
ev() { [ -f skills_lab/results/$1.json ] || $PY -B skills_lab/eval_skills.py "${@:2}" --name $1 --device cuda > skills_lab/results/$1.log 2>&1; }
gr() { [ -f skills_lab/results/grounded_$1.json ] || $PY -B skills_lab/grounded.py "${@:2}" --name $1 --device cuda > skills_lab/results/grounded_$1.log 2>&1; }
# 1. skills arm v4
M4=skills_lab/models/ft_skills4.pt
[ -f $M4 ] || $PY -B skills_lab/train_skills.py --init models/qa_best_5M_dense.pt --save $M4 --device cuda --amp --batch 32 --steps 11600 \
    --threads 8 --skill-p 0.35 --k-p 0.5 --unrelated-p 0.2 --dataset data_skills/train_v4.jsonl > skills_lab/results/train_ft_skills4.log 2>&1
ev ft_skills4 $M4 --modes orig,k1,k3close
ev ft_skills4_v2 $M4 --modes k1,k3close --testset data_skills/test_v2.json
ev ft_skills4_rw $M4 --modes k1,k3close --testset data_skills/test_v4_rewrite.json
[ -f skills_lab/models/ft_skills3.pt ] && ev ft_skills3_rw skills_lab/models/ft_skills3.pt --modes k1,k3close --testset data_skills/test_v4_rewrite.json
ev base5M_rw models/qa_best_5M_dense.pt --modes k1,k3close --testset data_skills/test_v4_rewrite.json
# 2. K2 version 2
for C in "5m runs/core_D3s512/scale_5M_v0_8k_ng64.pt" "100m runs/core_P100/scale_custom_v0_8k.pt"; do
  set -- $C
  [ -f results/k2v2_question_key_$1.json ] || $PY -B compiler/experiments/k2v2_question_key.py $2 --tag $1 --device cuda > runs/k2v2_$1.log 2>&1
done
ev k2v2_best5M_5m models/qa_best_5M_dense.pt --modes orig,k3close --key-head runs/k2v2_5m/key_head.pt
ev k2v2_best5M_100m models/qa_best_5M_dense.pt --modes orig,k3close --key-head runs/k2v2_100m/key_head.pt
ev k2v2_100M_100m runs/s2b_100m/qa_s2b_100m.pt --modes orig,k3close --key-head runs/k2v2_100m/key_head.pt
# 3. grounded scorecards
gr base5M models/qa_best_5M_dense.pt --mode k1
gr base5M_literal models/qa_best_5M_dense.pt --mode k1 --dense 0
gr 100M runs/s2b_100m/qa_s2b_100m.pt --mode k1
gr 100M_literal runs/s2b_100m/qa_s2b_100m.pt --mode k1 --dense 0
[ -f skills_lab/models/ft_skills3.pt ] && gr ftskills3 skills_lab/models/ft_skills3.pt --mode k3close
gr ftskills4 $M4 --mode k3close
gr mt_S runs/mt_S/qa_mt_S.pt --mode k1
gr mt_M runs/mt_M/qa_mt_M.pt --mode k1
gr k2v2_best5M models/qa_best_5M_dense.pt --mode k1 --key-head runs/k2v2_5m/key_head.pt
touch results/session34_done
ls skills_lab/results/grounded_*.json results/k2v2_question_key_*.json
