#!/bin/bash
# Session 54 (19 Oct): variant A's second seed (v24a_s2), the control for round 4's two-seed comparison (A had one seed;
# the noisy fronts - story contradictions, documents, "unknown" - need A's own seed spread). Same recipe as v24a_s1.
PY=.venv/Scripts/python.exe
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
SH='{"qa": 0.176, "skills": 0.132, "convo": 0.114, "recall": 0.106, "buried": 0.088, "docqa": 0.062, "story": 0.088, "rehearse": 0.114, "longconv": 0.12}'
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=v24a_s2
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --mix v22 --parts longconv,longdocs --shares "$SH" --seed 2 \
    --min-steps 20000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
touch results/session54_done
