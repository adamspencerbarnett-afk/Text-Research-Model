#!/bin/bash
# Skills arm v3 on the GPU (Adam, 14 October: 100,000 skill examples), queued last: after session 32 and the GPU
# speed tests, on an idle GPU. From the best 5M chat model, 9,000 steps of 32 windows (35% skill windows = ~100,800,
# about one pass over data_skills/train_v3.jsonl), balanced as round 3 (20% of research windows answered beside
# unrelated evidence). Scored on test v1 and the held-out test v2, with and without its decision head.
PY=.venv/Scripts/python.exe
until [ -f results/session32_done ] && [ -f inference_lab/results/gpu.json ]; do sleep 300; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
M=skills_lab/models/ft_skills3.pt
[ -f $M ] || $PY -B skills_lab/train_skills.py --init models/qa_best_5M_dense.pt --save $M --device cuda --amp --batch 32 --steps 9000 \
    --threads 8 --skill-p 0.35 --k-p 0.5 --unrelated-p 0.2 --dataset data_skills/train_v3.jsonl > skills_lab/results/train_ft_skills3.log 2>&1
ev() { [ -f skills_lab/results/$1.json ] || $PY -B skills_lab/eval_skills.py "${@:2}" --name $1 --device cuda > skills_lab/results/$1.log 2>&1; }
ev ft_skills3 $M --modes orig,k1,k3close
ev ft_skills3_v2 $M --modes orig,k1,k3close --testset data_skills/test_v2.json
ev base5M_v2 models/qa_best_5M_dense.pt --modes orig,k1,k3close --testset data_skills/test_v2.json
[ -f skills_lab/models/ft_skills.pt ] && ev ft_skills_v2 skills_lab/models/ft_skills.pt --modes k1,k3close --testset data_skills/test_v2.json
[ -f skills_lab/results/decision_ftskills3_k3.json ] || $PY -B skills_lab/decision.py $M --name ftskills3_k3 --mode k3close --device cuda > skills_lab/results/decision_ftskills3_k3.log 2>&1
ev ft_skills3_head $M --modes k3close --decision skills_lab/models/decision_ftskills3_k3.pt
ev ft_skills3_head_v2 $M --modes k3close --decision skills_lab/models/decision_ftskills3_k3.pt --testset data_skills/test_v2.json
[ -f runs/k2_100m/key_head.pt ] && ev ft_skills3_k2_100m_v2 $M --modes k3close --key-head runs/k2_100m/key_head.pt --testset data_skills/test_v2.json
touch skills_lab/results/gpu_skills_done
