#!/bin/bash
# Session 51 (18 Oct): seed 2 of round 3 (real data, Q&A kind from Alpaca). Round 3 seed 1 is level with A on documents,
# ahead on memory and told skills, behind on stories (36 vs 51/75: noise or recipe?). Seed 2 decides; then the probe
# section on A's store for both seeds (the probe asks about facts in A's Alpaca store, so it is store-bound).
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until [ -f results/session50_done ] && ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=real_5m_aq_s2
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m --mix v22 --parts longconv,longdocs,alpacaqa \
    --shares "$SH" --seed 2 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
for M in real_5m_aq_s1 real_5m_aq_s2; do
  [ -f results/suite_${M}_ev2probeA.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$M.pt --tier full --sections probe \
      --engine-opts '{"search": "v2", "facts": "data_qa/train/qa_train.txt"}' --tag _ev2probeA > runs/suite_${M}_ev2probeA.log 2>&1
done
touch results/session51_done
