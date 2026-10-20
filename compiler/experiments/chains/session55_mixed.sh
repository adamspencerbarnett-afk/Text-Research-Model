#!/bin/bash
# Session 55 (19 Oct), round 5: round 4 ("A + real memory") with the any-length sessions rebuilt on the MIXED pool (real
# facts + the generated personal facts): stored facts about named people are back in the sessions as training signal; the
# deployed store stays real-only. data_real/5m_mixed holds longconv/ -> the mixed sessions; everything else as round 4.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('data_real/5m/recipe.json'))['shares']))")
MO='{"gate": [0.3, 0.6], "gate_facts": [0.5, 0.8], "k_facts": 2, "k_session": 4}'
until [ -f data_real/shared/longconv_mixed_ready ]; do sleep 60; done
mkdir -p data_real/5m_mixed/longconv && cp data_real/shared/longconv_mixed/longconv_s1_[0-3].jsonl data_real/5m_mixed/longconv/
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
for S in 1 2; do
  N=mix_5m_s$S
  [ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m_mixed --mix v22 --parts longconv,longdocs,alpacaqa \
      --shares "$SH" --seed $S --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
  [ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
done
touch results/session55_done
