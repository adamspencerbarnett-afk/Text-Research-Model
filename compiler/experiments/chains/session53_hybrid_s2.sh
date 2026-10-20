#!/bin/bash
# Session 53 (19 Oct): seed 2 of round 4 ("A + real memory"), the first candidate to beat A on seed 1 (right with memory
# 1,174 vs 1,116; correction / distractor from memory 189 vs 176 / 169; documents 96/61 vs 81/53; story questions 58 vs 51;
# open: story contradictions 1.68 vs 0.68). Full suite with engine v2.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=hyb_5m_s2
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m_hybrid --mix v22 --parts longconv,longdocs,alpacaqa \
    --shares "$SH" --seed 2 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
touch results/session53_done
