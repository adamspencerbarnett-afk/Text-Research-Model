#!/bin/bash
# Session 43 (Adam, 17 Oct): repair the v2.3 losses while keeping its memory gains. Diagnosis in LOG.md (17 Oct 08:00).
#   A  v2.2 recipe + any-length conversations 12% + full-length buried documents
#   B  A + look-alike stored facts + varied memory budget + 1,024-code memory
#   C  all v2.3 changes with v2.2's recall / buried / document shares restored
# One run takes ~11 GB of GPU cache uncapped; three at once overflowed (17 Oct 08:10, stopped in time). Now one at a time:
# A (already running), then B, then C. Each model's full suite and guard follow its training.
PY=.venv/Scripts/python.exe
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
SH_AB='{"qa": 0.176, "skills": 0.132, "convo": 0.114, "recall": 0.106, "buried": 0.088, "docqa": 0.062, "story": 0.088, "rehearse": 0.114, "longconv": 0.12}'
SH_C='{"longconv": 0.14, "recall": 0.12, "buried": 0.15, "qa": 0.12, "skills": 0.12, "convo": 0.10, "story": 0.08, "docqa": 0.05, "rehearse": 0.12}'
tr() { N=$1; shift; [ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N "$@" --seed 1 --min-steps 20000 --patience 10 --memory-opts "$MO" > runs/chat22_train_$N.log 2>&1; }
suite() { N=$1; [ -f results/suite_$N.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full > runs/suite_$N.log 2>&1
  $PY -B -m chat22.tests.gate22 $N --refs chat5m_v2_A,chat5m_v21_s1,chat5m_v21_s2,v22_s1,v22_s2,v23_s1,v23_s2 > runs/gate22_$N.log 2>&1; }
# B OOMed at a 1/3 cap beside A (real need ~9-11 GB each): one training at a time, capped at 0.6 so a suite can run beside it
until [ -f results/chat22_train_v24a_s1.json ]; do sleep 60; done
suite v24a_s1 &
tr v24b_s1 --mix v22 --parts longconv,longdocs,crowd,budget,mem1024 --shares "$SH_AB" --gpu-frac 0.6
suite v24b_s1 &
tr v24c_s1 --mix v23 --parts longconv,longdocs,crowd,budget,mem1024,stored --shares "$SH_C" --gpu-frac 0.6
suite v24c_s1 &
wait
touch results/session43_done
