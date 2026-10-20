#!/bin/bash
# Session 56 (19 Oct), round 6: round 5 with the sessions rebuilt on mixed2 (real facts + generated personal facts +
# Alpaca Q&A), so the sessions practise prose answers from stored Q&A as A's did. Seed 1; seed 2 only if it beats A.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['shares']))")
MO=$($PY -c "import json; print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['memory_opts']))")
until [ -f data_real/shared/longconv_mixed2_ready ] && [ -f results/session55_done ]; do sleep 60; done
mkdir -p data_real/5m_mixed2/longconv && cp data_real/shared/longconv_mixed2/longconv_s1_[0-3].jsonl data_real/5m_mixed2/longconv/
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=mix2_5m_s1
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m_mixed2 --mix v22 --parts longconv,longdocs,alpacaqa \
    --shares "$SH" --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
$PY research_side/compare_two_seeds.py $N > runs/compare_$N.log 2>&1 || true
touch results/session56_done
