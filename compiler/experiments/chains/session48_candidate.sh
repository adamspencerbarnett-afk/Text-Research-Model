#!/bin/bash
# Session 48 (18 Oct): the improvement loop's first candidate. Real data (data_real/5m) + realev (buried evidence = the real
# passage sentence holding the answer; stored facts real facts only) on the corrected recipe; memqa dropped (clean tests:
# documents 50/29 vs 69/38, story questions 29 vs 54). Tests: the full suite with engine v2 on the fixed suite (clean store
# after stored skills), for the candidate and for A (like for like). Seed 2 only if seed 1 beats A.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
N=real_5m_rev_s1
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m --mix v22 --parts longconv,longdocs,realev \
    --shares "$SH" --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1 &
[ -f results/suite_v24a_s1_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/v24a_s1.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_v24a_s1_ev2full.log 2>&1
wait
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
touch results/session48_done
