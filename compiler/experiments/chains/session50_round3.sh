#!/bin/bash
# Session 50 (18 Oct), improvement round 3: the decisive ablation. real_5m_s1's recipe (real data) with ONE change: the
# Q&A kind (and its n-gram tables) from Alpaca, as variant A trained (alpacaqa); rehearsal, long documents, any-length
# sessions, look-alikes and the permanent store stay real. If documents, "unknown" and the invented share recover toward
# A while the real memory gains hold, the real Q&A format is the cause and gets reformatted. Full suite with engine v2.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until [ -f results/session49_done ] && ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=real_5m_aq_s1
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m --mix v22 --parts longconv,longdocs,alpacaqa \
    --shares "$SH" --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
touch results/session50_done
