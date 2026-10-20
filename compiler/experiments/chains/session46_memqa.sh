#!/bin/bash
# Session 46 (Adam, 18 Oct): facts come from permanent memory. The real 5M of session 45 with one change, the memqa switch:
# fact questions are answered only from a memory item that holds the fact (70%); when the block lacks it the target is
# "I don't have that" (30%); reading questions (passage / context / options) unchanged. Permanent store = knowledge_real
# (311k: Wikipedia, science facts, the memory-fact Q&A). Same recipe and steps as session 45; no grounding stage (rejected).
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
REFS=chat5m_v2_A,chat5m_v21_s1,v22_s1,v23_s1,v24a_s1,real_5m_s1,real_5m_s2
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 60; done
for S in 1 2; do
  N=real_5m_mq_s$S
  [ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m --mix v22 --parts longconv,longdocs,memqa \
      --shares "$SH" --seed $S --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
  [ -f results/suite_$N.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full > runs/suite_$N.log 2>&1
  $PY -B -m chat22.tests.gate22 $N --refs $REFS > runs/gate22_$N.log 2>&1
  [ -f results/suite_${N}_ev2.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --sections skills_stored,long,memory \
      --engine-opts '{"search": "v2"}' --tag _ev2 > runs/suite_${N}_ev2.log 2>&1
done
touch results/session46_done
