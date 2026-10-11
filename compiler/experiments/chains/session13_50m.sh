#!/bin/bash
# Session 13: the 50M large-scale test (docs/PLAN_50M.md). Resumable.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
MIX="$PY -I compiler/experiments/prepare_core_mix.py"
until grep -q "training pairs" data_core/logs/evidence_p50.log 2>/dev/null; do sleep 120; done
# the builder writes its held-out pairs over the report cards' fixed test set: keep both
if [ ! -f data_core/eval/evidence_val_p50.jsonl ]; then
  mv data_core/eval/evidence_val.jsonl data_core/eval/evidence_val_p50.jsonl; cp data_core/eval/evidence_val_v1.jsonl data_core/eval/evidence_val.jsonl
fi
# P50 sources: the shared ones plus the P50 evidence pairs
mkdir -p data_core/src_p50
for f in wiki fineweb books oasst ultrachat math qqp; do [ -e data_core/src_p50/$f.txt ] || cp data_core/src/$f.txt data_core/src_p50/$f.txt; done
[ -f data_core/mix_P50/manifest.json ] || { $MIX mix data_core/src_p50 data_core/mix_P50 --mb 2400 --recipe P50 > data_core/logs/mix_P50.log 2>&1; mkdir -p data_core/mix_P50/ood; cp data_v2/heldout/*.txt data_core/mix_P50/ood/; }
# wait for the GPU queue (verified self-training) to finish
until [ -f results/star_r2.json ] || grep -lq Traceback runs/star_r2/*.log 2>/dev/null; do sleep 120; done
[ -f results/core_P50_50M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_P50 runs/core_P50 --native "$N" --device cuda --amp \
    --sizes 50M --configs v0_8k --budgets 18000 --max-steps 30500 --batch 32 --ctx 1024 --patience 0 --eval-every 600 --no-wiki \
    --out results/core_P50_50M.json > runs/core_P50.log 2>&1
[ -f results/core_eval_core_P50_scale_50M_v0_8k.json ] || $PY -B compiler/experiments/core_eval.py runs/core_P50/scale_50M_v0_8k.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_P50.log 2>&1
mkdir -p runs/s2b_50m
[ -f results/s2b_50m.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/s2b_50m --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 7200 \
    --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
    --dict runs/core_P50/v0_8k_plain.cv0d --init runs/core_P50/scale_50M_v0_8k.pt --width 640 --layers 10 --heads 10 --ctx 1024 --neighbor-codes 400 \
    --rehearse data_core/mix_P50 --out results/s2b_50m.json --save runs/s2b_50m/qa_s2b_50m.pt > runs/s2b_50m.log 2>&1
[ -f results/probe_qa_s2b_50m.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2b_50m/qa_s2b_50m.pt --device cuda > runs/probe_s2b_50m.log 2>&1
ls results/core_P50_50M.json results/core_eval_core_P50_*.json results/s2b_50m.json results/probe_qa_s2b_50m.json
