#!/bin/bash
# Step 1 (Adam, 16 Oct): stricter memory selection on v2.2, no training. The engine let six look-alike stored facts
# (from 50,937 general Q&A pairs) into every memory block; v2.2 copied from them ("Pablo is a large oil painting").
# Three settings, both seeds, the memory-sensitive sections only (long, memory, skills_stored); compare with the
# default-setting results already in results/suite_v22_s{1,2}.json (same cases).
#   strict    stored facts need word 0.35 or meaning 0.70; at most 2 facts and 4 conversation items
#   stricter  conversation items need 0.30 / 0.60, stored facts 0.50 / 0.80; 2 and 4
#   nofacts   no stored facts at all; at most 4 conversation items
PY=.venv/Scripts/python.exe
until [ -f results/session40_done ]; do sleep 60; done
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 60; done
declare -A OPTS=(
  [strict]='{"gate_facts": [0.35, 0.7], "k_facts": 2, "k_session": 4}'
  [stricter]='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
  [nofacts]='{"use_facts": false, "k_session": 4}'
)
for S in 1 2; do
  for C in strict stricter nofacts; do
    [ -f results/suite_v22_s${S}_$C.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/v22_s$S.pt --sections long,memory,skills_stored \
        --engine-opts "${OPTS[$C]}" --tag _$C > runs/step1_v22_s${S}_$C.log 2>&1
  done
done
touch results/step1_done
