#!/bin/bash
# Session 49 (18 Oct), improvement round 2: real_5m_s1's recipe (real data, no realev / memqa) with ONE change: the
# buried-evidence Q&A drawn from the Alpaca exchanges A trained on (alpacabury). Question: do documents recover toward A
# while the real data's memory and skills gains hold? Full suite with engine v2 (fixed suite).
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until ! pgrep -f "chat22.train" > /dev/null 2>&1 && [ -f results/session48_done ]; do sleep 60; done
N=real_5m_ab_s1
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m --mix v22 --parts longconv,longdocs,alpacabury \
    --shares "$SH" --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
[ -f results/suite_real_5m_s1_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/real_5m_s1.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_real_5m_s1_ev2full.log 2>&1
touch results/session49_done
