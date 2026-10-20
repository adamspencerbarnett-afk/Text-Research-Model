#!/bin/bash
# Wake the assistant when the next run of the queue finishes or anything fails.
#   bash compiler/experiments/chains/watch.sh RESULT_FILE [RESULT_FILE ...]
# Exits (printing what happened) as soon as any listed result file exists, a run log written after
# the watch began contains a Traceback, no Python process has run for 10 minutes (a stalled queue), or
# (16 October, after session 36 hung for 9.5 h unnoticed) a Python job is running but no log under runs/ or
# data_core/logs/ has been written for STALL_MIN minutes (default 30): a frozen job looks alive to a process check.
STALL_MIN=${STALL_MIN:-30}
start=$(date +%s); idle=0
while true; do
  for f in "$@"; do [ -f "$f" ] && { echo "DONE $f"; exit 0; }; done
  for l in runs/*.log runs/*/*.log data_core/logs/*.log; do
    [ -f "$l" ] && [ $(stat -c %Y "$l") -ge $start ] && grep -q Traceback "$l" && { echo "FAILED $l"; tail -n 5 "$l"; exit 1; }
  done
  if tasklist | grep -qi python.exe; then
    idle=0
    newest=$(stat -c %Y runs/*.log runs/*/*.log data_core/logs/*.log skills_lab/manual/*.txt 2>/dev/null | sort -n | tail -1)
    now=$(date +%s)
    if [ -n "$newest" ] && [ $((now - newest)) -ge $((STALL_MIN * 60)) ] && [ $((now - start)) -ge $((STALL_MIN * 60)) ]; then
      echo "STALLED: Python running but no log written for $(( (now - newest) / 60 )) minutes"
      ls -t runs/*.log runs/*/*.log 2>/dev/null | head -3
      nvidia-smi --query-gpu=memory.used,utilization.gpu,power.draw --format=csv,noheader
      exit 3
    fi
  else
    idle=$((idle+1))
  fi
  [ $idle -ge 20 ] && { echo "STALLED: no python process for 10 minutes"; exit 2; }
  sleep 30
done
