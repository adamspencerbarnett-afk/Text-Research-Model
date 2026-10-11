#!/bin/bash
# Session 8: does a gentler stage 2 keep what better stage-1 data built? From the D3r core.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
BASE=(--native "$N" --device cuda --dict runs/core_D3r/v0_8k_plain.cv0d --init runs/core_D3r/scale_5M_v0_8k_ng64.pt
      --width 256 --layers 6 --heads 8 --ctx 256 --neighbor-codes 110 --batch 32 --budget 1800 --max-steps 15565
      --self-context-p 0.5 --eval-n 200 --copy-head --greedy)
run() {   # name, extra args...
  local name=$1; shift
  mkdir -p runs/$name
  [ -f results/$name.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/$name "${BASE[@]}" "$@" --out results/$name.json --save runs/$name/qa_$name.pt > runs/$name.log 2>&1
  [ -f results/probe_qa_$name.json ] || $PY -B compiler/experiments/chat_probe.py runs/$name/qa_$name.pt --device cuda > runs/probe_$name.log 2>&1
}
run s2a1_lr3e-4 --lr 3e-4
run s2a2_lr3e-4_reh --lr 3e-4 --rehearse data_core/mix_D3_5M --rehearse-p 0.3
run s2a3_lr1e-4_reh --lr 1e-4 --rehearse data_core/mix_D3_5M --rehearse-p 0.3
grep -l Traceback runs/s2a*.log runs/probe_s2a*.log; ls results/s2a*.json results/probe_qa_s2a*.json
