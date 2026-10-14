#!/bin/bash
# Inference lab phase B: after session 31's last report card, and only once no other program uses the GPU.
PY=.venv/Scripts/python.exe
until [ -f results/core_eval_mt_L2_qa_mt_L2_clean.json ] || grep -lq Traceback runs/mt_L2.log 2>/dev/null; do sleep 300; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
$PY -B inference_lab/bench_gpu.py > inference_lab/results/gpu.log 2>&1
