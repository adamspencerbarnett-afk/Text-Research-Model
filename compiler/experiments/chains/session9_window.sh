#!/bin/bash
# Session 9: stop cutting answers. Industry-style Q&A stage (lr 1e-4 warm-up + cosine, answer-only loss,
# rehearsal, 200 questions per group), first on the 256-window core, then on a new 512-window core
# whose Q&A windows hold whole exchanges and whole retrieved exchanges. Resumable.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2a3_lr1e-4_reh.json ] || grep -lq Traceback runs/s2a*.log 2>/dev/null; do sleep 60; done
S2=(--native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 --self-context-p 0.5 --eval-n 200 --net-n 200
    --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --rehearse-p 0.3)
run2() {  # name, core dir, ctx, neighbor codes, rehearse mix
  mkdir -p runs/$1
  [ -f results/$1.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/$1 "${S2[@]}" --dict runs/$2/v0_8k_plain.cv0d \
      --init runs/$2/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx $3 --neighbor-codes $4 --rehearse $5 \
      --out results/$1.json --save runs/$1/qa_$1.pt > runs/$1.log 2>&1
  [ -f results/probe_qa_$1.json ] || $PY -B compiler/experiments/chat_probe.py runs/$1/qa_$1.pt --device cuda > runs/probe_$1.log 2>&1
}
run2 s2b_256 core_D3r 256 110 data_core/mix_D3_5M
# the 512-window core: D3 data with document separators, random windows, equal steps
[ -f data_core/mix_D3s_5M/manifest.json ] || { $PY -I compiler/experiments/prepare_core_mix.py mix data_core/src data_core/mix_D3s_5M --mb 400 --recipe D3 > data_core/logs/mix_D3s.log 2>&1; mkdir -p data_core/mix_D3s_5M/ood; cp data_v2/heldout/*.txt data_core/mix_D3s_5M/ood/; }
[ -f results/core_D3s512_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M runs/core_D3s512 --native "$N" --device cuda --amp \
    --sizes 5M --configs v0_8k_ng64 --budgets 5400 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki --ctx 512 --out results/core_D3s512_5M.json > runs/core_D3s512.log 2>&1
[ -f results/core_eval_scale_5M_v0_8k_ng64_mix_D3s512_5M.json ] || { $PY -B compiler/experiments/core_eval.py runs/core_D3s512/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_D3s512.log 2>&1 \
    && mv results/core_eval_scale_5M_v0_8k_ng64_mix_D3_5M.json results/core_eval_scale_5M_v0_8k_ng64_mix_D3s512_5M.json; }
run2 s2b_512 core_D3s512 512 250 data_core/mix_D3s_5M
grep -l Traceback runs/s2b_*.log runs/probe_s2b_*.log runs/core_D3s512.log runs/core_eval_D3s512.log
ls results/s2b_*.json results/probe_qa_s2b_*.json results/core_D3s512_5M.json results/core_eval_*D3s512*.json
