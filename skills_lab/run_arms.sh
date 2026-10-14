#!/bin/bash
# Skills lab, round 1 (CPU, beside the GPU queue): control and skills arms from the best 5M chat model, 3,000 steps each.
PY=.venv/Scripts/python.exe
arm() {  # $1 name, then trainer flags
  local N=$1; shift
  [ -f skills_lab/models/$N.pt ] || $PY -B skills_lab/train_skills.py --init models/qa_best_5M_dense.pt --save skills_lab/models/$N.pt --steps 3000 --threads 12 "$@" > skills_lab/results/train_$N.log 2>&1
  [ -f skills_lab/results/$N.json ] || $PY -B skills_lab/eval_skills.py skills_lab/models/$N.pt --name $N --threads 6 > skills_lab/results/$N.log 2>&1
}
arm ft_control --skill-p 0 --k-p 0 &
arm ft_skills --skill-p 0.35 --k-p 0.5 &
wait
