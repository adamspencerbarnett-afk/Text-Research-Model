#!/bin/bash
# Wake the assistant when the next run of the queue finishes or anything fails.
#   bash compiler/experiments/chains/watch.sh RESULT_FILE [RESULT_FILE ...]
# Exits (printing what happened) as soon as any listed result file exists, a run log written after
# the watch began contains a Traceback, or no Python process has run for 10 minutes (a stalled queue).
start=$(date +%s); idle=0
while true; do
  for f in "$@"; do [ -f "$f" ] && { echo "DONE $f"; exit 0; }; done
  for l in runs/*.log data_core/logs/*.log; do
    [ -f "$l" ] && [ $(stat -c %Y "$l") -ge $start ] && grep -q Traceback "$l" && { echo "FAILED $l"; tail -n 5 "$l"; exit 1; }
  done
  if tasklist | grep -qi python.exe; then idle=0; else idle=$((idle+1)); fi
  [ $idle -ge 20 ] && { echo "STALLED: no python process for 10 minutes"; exit 2; }
  sleep 30
done
