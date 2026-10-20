#!/bin/bash
# Session 44 (Adam, 17 Oct): target the gaps variant A (v24a) left, after session 43 ends. All on A's recipe and switches
# (longconv, longdocs), seed 1, same floors and test settings; one training at a time (GPU cap 0.6), each suite + guard after.
#   D  documents: v2.1's mix (Q&A 25, skills 20, conversations 15, recall 10, buried 15, plain text 15; no document-
#      reading or story kinds) scaled to make room for any-length conversations 12%. Diagnostic, not a candidate (no stories).
#   E  documents: A's mix with buried evidence 15% (from Q&A and plain text); document-reading and stories kept.
#   F  long windows: A's mix with short recall 15% (the long-history kind; from Q&A and plain text).
# Reading: D fixes documents and E does not -> the document-reading / story kinds interfere; both fix -> buried share;
# neither -> the trainer itself (v2.1's buried filler came from a small repeated sample) -> next study.
PY=.venv/Scripts/python.exe
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until [ -f results/session43_done ]; do sleep 60; done
tr() { N=$1; shift; [ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N "$@" --mix v22 --parts longconv,longdocs --seed 1 --min-steps 20000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1; }
suite() { N=$1; [ -f results/suite_$N.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full > runs/suite_$N.log 2>&1
  $PY -B -m chat22.tests.gate22 $N --refs chat5m_v2_A,chat5m_v21_s1,chat5m_v21_s2,v22_s1,v22_s2,v23_s1,v23_s2,v24a_s1 > runs/gate22_$N.log 2>&1; }
tr v24d_s1 --shares '{"qa": 0.22, "skills": 0.176, "convo": 0.132, "recall": 0.088, "buried": 0.132, "rehearse": 0.132, "longconv": 0.12}'
suite v24d_s1 &
tr v24e_s1 --shares '{"qa": 0.14, "skills": 0.132, "convo": 0.10, "recall": 0.106, "buried": 0.15, "docqa": 0.062, "story": 0.088, "rehearse": 0.102, "longconv": 0.12}'
suite v24e_s1 &
tr v24f_s1 --shares '{"qa": 0.15, "skills": 0.132, "convo": 0.114, "recall": 0.15, "buried": 0.088, "docqa": 0.062, "story": 0.088, "rehearse": 0.094, "longconv": 0.12}'
suite v24f_s1 &
wait
touch results/session44_done
