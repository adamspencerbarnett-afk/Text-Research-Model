#!/bin/bash
# Session 47 (18 Oct): re-test the sections the stored-skills leak contaminated (probe, convo, long, memory, story) with a
# clean store (run_suite now rebuilds the engine after skills_stored; these runs skip skills_stored anyway).
PY=.venv/Scripts/python.exe
S=probe,convo,long,memory,story
for M in v24a_s1 real_5m_s1 real_5m_mq_s1 real_5m_s2; do
  [ -f results/suite_${M}_ev2clean.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$M.pt --tier full --sections $S \
      --engine-opts '{"search": "v2"}' --tag _ev2clean > runs/suite_${M}_ev2clean.log 2>&1
done
for M in v24a_s1 real_5m_s1; do
  [ -f results/suite_${M}_clean.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$M.pt --tier full --sections $S --tag _clean > runs/suite_${M}_clean.log 2>&1
done
touch results/session47_done
