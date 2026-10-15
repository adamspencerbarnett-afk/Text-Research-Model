#!/bin/bash
# Inference lab phase B: after session 32 (K2, which follows session 31), and only once no other program uses the GPU.
PY=.venv/Scripts/python.exe
until [ -f results/session32_done ]; do sleep 300; done
# wait until no Python job uses the GPU (Windows always lists desktop programs such as Explorer as GPU users)
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 120; done
$PY -B inference_lab/bench_gpu.py > inference_lab/results/gpu.log 2>&1
