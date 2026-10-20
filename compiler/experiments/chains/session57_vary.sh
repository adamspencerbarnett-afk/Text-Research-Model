#!/bin/bash
# Session 57 (19 Oct), round 7: round 5 (BEST v2) with ONE change: the any-length sessions regenerated with natural
# rewordings of 40% of the questions ("I forgot, ...", "Can you tell me ...", "..., do you remember?"), after Adam's
# hands-on case (a stored fact refused when asked in natural words). Waits for round 6 and the varied sessions.
PY=.venv/Scripts/python.exe
SH=$($PY -c "import json; print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['shares']))")
MO=$($PY -c "import json; print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['memory_opts']))")
until [ -f data_real/shared/longconv_mixedv_ready ] && [ -f results/session56_done ]; do sleep 60; done
mkdir -p data_real/5m_mixedv/longconv && cp data_real/shared/longconv_mixedv/longconv_s1_[0-3].jsonl data_real/5m_mixedv/longconv/
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
N=mixv_5m_s1
[ -f results/chat22_train_$N.json ] || $PY -B -m chat22.train --name $N --data-dir data_real/5m_mixedv --mix v22 --parts longconv,longdocs,alpacaqa \
    --shares "$SH" --seed 1 --min-steps 30000 --max-steps 48000 --patience 10 --memory-opts "$MO" --gpu-frac 0.6 > runs/chat22_train_$N.log 2>&1
[ -f results/suite_${N}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$N.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${N}_ev2full.log 2>&1
$PY research_side/compare_two_seeds.py $N > runs/compare_$N.log 2>&1 || true
$PY research_side/natural_rewording_test.py --models $N > runs/rewording_$N.log 2>&1 || true
touch results/session57_done
