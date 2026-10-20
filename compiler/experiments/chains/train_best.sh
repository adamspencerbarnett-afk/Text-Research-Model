#!/bin/bash
# Train and test the BASE recipe of recipes/BEST.json on one seed (plus any extra args = one change), then compare
# it with the current BEST (two seeds).
#   bash compiler/experiments/chains/train_best.sh SEED [NAME] [EXTRA TRAIN ARGS...]
# With no extra args it reproduces the best recipe (same data, switches, shares, floors). Extra args (e.g. --parts ...,
# --data-dir ..., --shares '{...}') try one change on top of it: the candidate for the next best. Results:
# results/chat22_train_NAME.json, results/suite_NAME_ev2full.json, runs/compare_NAME.log (vs the best's confirmed seeds).
# A candidate becomes the new best only after a second seed (train_best.sh 2 NAME_s2 ...) and the two-seed comparison
# (python research_side/compare_two_seeds.py NAME_s1 NAME_s2) wins; then recipes/BEST.json is edited by hand.
set -e
cd "$(dirname "$0")/../../.."
PY=${PY:-.venv/Scripts/python.exe}   # Linux/macOS: PY=.venv/bin/python bash compiler/experiments/chains/train_best.sh ...
SEED=${1:?seed}; NAME=${2:-base_s$SEED}; shift; shift 2>/dev/null || true
read -r CORE DATA MIX PARTS MINS MAXS PAT FRAC <<< "$($PY -c "
import json; t=json.load(open('recipes/BEST.json'))['base']['train']
print(t['core'], t['data_dir'], t['mix'], t['parts'], t['min_steps'], t['max_steps'], t['patience'], t['gpu_frac'])")"
SH=$($PY -c "import json; print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['shares']))")
MO=$($PY -c "import json; print(json.dumps(json.load(open('recipes/BEST.json'))['base']['train']['memory_opts']))")
until ! pgrep -f "chat22.train" > /dev/null 2>&1; do sleep 60; done
echo "$(date +%H:%M:%S) train_best: $NAME seed $SEED from recipes/BEST.json (base recipe) extra: $*"
DD=""; [ "$DATA" != "None" ] && DD="--data-dir $DATA"
[ -f results/chat22_train_$NAME.json ] || $PY -B -m chat22.train --name $NAME --core $CORE $DD --mix $MIX --parts $PARTS \
    --shares "$SH" --seed $SEED --min-steps $MINS --max-steps $MAXS --patience $PAT --memory-opts "$MO" --gpu-frac $FRAC "$@" > runs/chat22_train_$NAME.log 2>&1
[ -f results/suite_${NAME}_ev2full.json ] || $PY -B -m chat22.tests.run_suite runs/chat22/$NAME.pt --tier full --engine-opts '{"search": "v2"}' --tag _ev2full > runs/suite_${NAME}_ev2full.log 2>&1
$PY research_side/compare_two_seeds.py $NAME > runs/compare_$NAME.log 2>&1 || true
echo "$(date +%H:%M:%S) train_best: $NAME done; see runs/compare_$NAME.log"
