#!/bin/bash
# Session 36 (after session 35; Adam, 15 October): how far can the context be pushed on this machine, against the
# "1 million token" windows of large models? compiler/experiments/context_stress.py, on the GPU (+ CPU for the 5M):
#   raw window: 2k -> 1M codes (5M chat model) and 2k -> 512k (100M): read time (= first-word wait), ms per code,
#     peak GPU memory, quality by position band, needle at 10% depth found by attention alone
#   session memory: a fact in turn 1, then 10 / 100 / 1,000 / 5,000 real turns (up to ~3M codes), recall + timing
PY=.venv/Scripts/python.exe
until [ -f results/session35_done ]; do sleep 300; done
# wait until no Python job uses the GPU (Windows always lists desktop programs such as Explorer as GPU users)
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 120; done
[ -f results/context_stress.json ] || $PY -B compiler/experiments/context_stress.py --device cuda --max-5m 1048576 --max-100m 524288 --budget-s 900 > runs/context_stress.log 2>&1
touch results/session36_done
