#!/bin/bash
# Session 52 (19 Oct), improvement round 4: "A + real memory". Variant A's data (Alpaca Q&A, P100 plain text with the
# novels, novels for story continuation) plus only the real components that carry the memory gains: the any-length
# sessions built on the real pool, the real look-alikes and the real permanent store. data_real/5m_hybrid holds nothing;
# shared/ supplies the real parts and everything else falls back to A's defaults (printed as "data sources" in the log).
# Seed 1, full suite with engine v2; seed 2 only if seed 1 beats A.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=hyb_5m_s1
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m_hybrid --mix v22 --parts longconv,longdocs,alpacaqa \
    --shares "$SH" --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
touch results/session52_done
