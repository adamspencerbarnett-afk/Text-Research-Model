#!/bin/bash
# Inference lab phase B: after session 32 (K2, which follows session 31), and only once no other program uses the GPU.
PY=.venv/Scripts/python.exe
until [ -f results/session32_done ]; do sleep 300; done
until [ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; do sleep 120; done
$PY -B inference_lab/bench_gpu.py > inference_lab/results/gpu.log 2>&1
