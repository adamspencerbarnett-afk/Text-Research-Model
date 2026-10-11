#!/bin/bash
# Session 11: the evidence-window test at full share. D3e = D3 recipe with the E1 evidence pairs
# (18% of the mix instead of 3%), document separators, 512 window, same steps as D3s512; then the S2b
# Q&A recipe, report card and chat probe. Compared with D3s512 / S2b-512, this isolates evidence windows.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until grep -q "training pairs" data_core/logs/evidence_e1.log 2>/dev/null && [ -f results/probe_qa_s2b_512_s2.json ]; do sleep 60; done
# keep the report cards comparable: the held-out evidence test stays the v1 set; the E1 one is kept beside it
if [ ! -f data_core/eval/evidence_val_e1.jsonl ]; then
  mv data_core/eval/evidence_val.jsonl data_core/eval/evidence_val_e1.jsonl; cp data_core/eval/evidence_val_v1.jsonl data_core/eval/evidence_val.jsonl
  mv data_core/src/evidence.txt data_core/src/evidence_v1.txt; cp data_core/src_e1/evidence.txt data_core/src/evidence.txt
fi
[ -f data_core/mix_D3e_5M/manifest.json ] || { $PY -I compiler/experiments/prepare_core_mix.py mix data_core/src data_core/mix_D3e_5M --mb 400 --recipe D3 > data_core/logs/mix_D3e.log 2>&1; mkdir -p data_core/mix_D3e_5M/ood; cp data_v2/heldout/*.txt data_core/mix_D3e_5M/ood/; }
[ -f results/core_D3e512_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3e_5M runs/core_D3e512 --native "$N" --device cuda --amp \
    --sizes 5M --configs v0_8k_ng64 --budgets 5400 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 --out results/core_D3e512_5M.json > runs/core_D3e512.log 2>&1
[ -f results/core_eval_core_D3e512_scale_5M_v0_8k_ng64.json ] || $PY -B compiler/experiments/core_eval.py runs/core_D3e512/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_D3e512.log 2>&1
mkdir -p runs/s2b_512e
[ -f results/s2b_512e.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/s2b_512e --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_D3e512/v0_8k_plain.cv0d --init runs/core_D3e512/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
    --rehearse data_core/mix_D3e_5M --out results/s2b_512e.json --save runs/s2b_512e/qa_s2b_512e.pt > runs/s2b_512e.log 2>&1
[ -f results/probe_qa_s2b_512e.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_512e/qa_s2b_512e.pt --device cuda > runs/probe_s2b_512e.log 2>&1
ls results/core_D3e512_5M.json results/core_eval_core_D3e512_*.json results/s2b_512e.json results/probe_qa_s2b_512e.json
