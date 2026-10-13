#!/bin/bash
# Evidence pairs for the P50 mix, in parallel: index once, 10 workers, merge (log line ends the wait in session 13).
PY=".venv/Scripts/python.exe -I compiler/experiments/evidence_parallel.py"
D=runs/evp50
[ -f $D/index.npz ] || $PY prep data_core/src/wiki.txt,data_core/src/fineweb.txt $D --min-len 200 --max-len 900 > data_core/logs/evidence_p50_prep.log 2>&1
for w in $(seq 0 9); do [ -f $D/w$w.done ] || ( $PY work $D --worker $w --of 10 > $D/w$w.log 2>&1 && touch $D/w$w.done ) & done
wait
$PY merge $D data_core/src_p50/evidence.txt data_core/eval/evidence_val_p50.jsonl --max-pairs 450000 > data_core/logs/evidence_p50.log 2>&1
