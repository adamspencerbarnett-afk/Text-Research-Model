#!/bin/bash
# Session 45 (Adam, 17 Oct): a new 5M model on the real datasets (data_real/5m + shared/), after sessions 43-44.
#   stage 2: one merged chat stage, every kind at the shares in data_real/5m/recipe.json (set from variants C-F and marked
#            "final"; if not final 90 min after session 44 ends, the draft is used and logged), two seeds, 30k-48k steps
#   stage 3: grounding (chat22/ground.py) on each seed: auto-graded answer pairs + the mix replayed
#   tests:   full suite + guard after each stage; engine v2 memory sections on the final models
PY=.venv/Scripts/python.exe
until [ -f results/session44_done ]; do sleep 60; done
for i in $(seq 1 90); do $PY -c "import json,sys; sys.exit(0 if json.load(open('data_real/5m/recipe.json'))['final'] else 1)" && break; sleep 60; done
$PY -c "import json; r=json.load(open('data_real/5m/recipe.json')); print('recipe', 'final' if r['final'] else 'DRAFT (not finalized in time)', r['shares'], r['parts'])" >> runs/session45_chain.log
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
PARTS=$($PY -c "import json; print(json.load(open('data_real/5m/recipe.json'))['parts'])")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
REFS=chat5m_v2_A,chat5m_v21_s1,chat5m_v21_s2,v22_s1,v22_s2,v23_s1,v23_s2,v24a_s1
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 60; done
suite() { N=$1; [ -f results/suite_$N.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full > runs/suite_$N.log 2>&1
  $PY -B -m chat22.tests.gate22 $N --refs $REFS${2:+,$2} > runs/gate22_$N.log 2>&1; }
for S in 1 2; do
  N=real_5m_s$S
  [ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m --mix v22 --parts "$PARTS" --shares "$SH" \
      --seed $S --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
  suite $N &
  [ -f results/chat22_ground_${N}_g.json ] || $PY -B -m chat22.ground --source runs/chat22/$N.pt --name ${N}_g --steps 3000 --gpu-frac 0.6 > runs/chat22_ground_${N}_g.log 2>&1
  wait
  suite ${N}_g $N                                   # stage 3 is also judged against its own stage 2
  for M in $N ${N}_g; do
    [ -f results/suite_${M}_ev2.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$M.pt --tier full --sections skills_stored,long,memory \
        --engine-opts '{"search": "v2"}' --tag _ev2 > runs/suite_${M}_ev2.log 2>&1
  done
done
touch results/session45_done
