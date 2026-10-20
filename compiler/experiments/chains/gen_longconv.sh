#!/bin/bash
# v2.3 any-length sessions: 4 parallel parts x 1,250 sessions (CPU), with the retrieval study's search fixes; marks
# data_chat22/longconv_ready when all parts are written (session 42 waits for it).
PY=.venv/Scripts/python.exe
for S in 0 1 2 3; do
  [ -f data_chat22/longconv_s1_$S.jsonl ] && grep -q "\"$S\"" data_chat22/manifest_v23.json 2>/dev/null || \
    OMP_NUM_THREADS=4 $PY -B -m chat22.data.build_v23 --only longconv --sessions 1250 --shard $S > runs/gen_longconv_$S.log 2>&1 &
done
wait
ls data_chat22/longconv_s1_[0-3].jsonl && touch data_chat22/longconv_ready
