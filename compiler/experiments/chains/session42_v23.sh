#!/bin/bash
# Session 42 (Adam, 16 October): chat model v2.3, the any-length "infinite loop" study, from the clean chat22 package.
# Design: docs/CHAT5M_V2_DESIGN.html#v23; mix: docs/MIX_STUDY_V23.md (any-length conversations with the real engine
# search 18, buried evidence 15, plain text 17, Q&A 12, skills 12 told and stored, conversations 12, stories 8, document
# reading 4, short recall 2); look-alike stored facts in half the windows; varied memory budget; memory settings from
# step 1 (MEMORY_OPTS below); a fixed held-out set; plateau stop.
#   1. v2.3 seeds 1 and 2 (lr 1e-4, batch 16)
#   2. the full suite (with the grounding score) on both seeds, then the guard against arm A, v2.1 and v2.2
#   2b. search engine v2: judge training, then the memory sections with engine v2 (rules only, and with the judge)
#   3. the learning-rate / batch check on seed 1: lr 2e-4 (batch 16) and batch 32 (lr 1e-4); judged by time to the
#      plateau and the held-out score at the best step (and the quick suite)
PY=.venv/Scripts/python.exe
MEMORY_OPTS=${MEMORY_OPTS:-'{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'}   # step 1 verdict: stricter (both seeds)
until [ -f results/step1_done ] && [ -f data_chat22/longconv_ready ]; do sleep 60; done
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 60; done
tr() { N=$1; shift; [ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --mix v23 --memory-opts "$MEMORY_OPTS" "$@" > runs/chat22_train_$N.log 2>&1; }
tr v23_s1 --seed 1
tr v23_s2 --seed 2
for S in 1 2; do
  [ -f results/suite_v23_s$S.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/v23_s$S.pt --tier full > runs/suite_v23_s$S.log 2>&1
  $PY -B -m chat22.tests.gate22 v23_s$S --refs chat5m_v2_A,chat5m_v21_s1,chat5m_v21_s2,v22_s1,v22_s2 > runs/gate22_v23_s$S.log 2>&1
done
touch results/session42_main_done
# permanent store of real facts only (Adam, 16 Oct: data_chat22/knowledge_wiki.jsonl, Wikipedia; training data unchanged) and
# search engine v2 (docs/SEARCH_ENGINE_V2.md): train the judge (2 blocks from the 5M core), then the memory sections on both
# v2.3 seeds with the real store: today's search (_real), engine v2 rules only (_ev2real), engine v2 + judge (_ev2jreal)
[ -f runs/chat22/judge.pt ] || $PY -B -m chat22.judge --layers 2 > runs/chat22_judge.log 2>&1
K='"facts": "data_chat22/knowledge_wiki.jsonl"'
for S in 1 2; do
  for V in "real|{$K}" "ev2real|{\"search\": \"v2\", $K}" "ev2jreal|{\"search\": \"v2\", \"judge\": \"runs/chat22/judge.pt\", $K}"; do
    T=${V%%|*}; O=${V#*|}
    [ -f results/suite_v23_s${S}_$T.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/v23_s$S.pt --tier full --sections skills_stored,long,memory       --engine-opts "$O" --tag _$T > runs/suite_v23_s${S}_$T.log 2>&1
  done
done
touch results/session42_ev2_done
tr v23_lr2 --seed 1 --lr 2e-4
tr v23_b32 --seed 1 --batch 32
for N in v23_lr2 v23_b32; do
  [ -f results/suite_${N}_quick.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier quick > runs/suite_${N}_quick.log 2>&1
done
touch results/session42_done
