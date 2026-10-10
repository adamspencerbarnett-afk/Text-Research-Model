#!/bin/bash
# Session 7: which stage-1 data makes the 5M core best at the three jobs (fluency incl. conversation,
# using evidence, question representations)? D0 = today's core (books + raw-markup Wikipedia).
#   D1 = clean modern text (Wikipedia, FineWeb-Edu, unwrapped books)
#   D2 = D1 + conversation (OpenAssistant, UltraChat)
#   D3 = D2 + evidence windows + worked math + question rewordings
# Equal training: 42,538 steps each (D0's count), 400 MB mixes, same model and settings; windows start
# at paragraph starts (--align), except D3r, which repeats D3 with random windows to measure alignment.
# Resumable: every step is skipped when its output exists.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
MIX="$PY -I compiler/experiments/prepare_core_mix.py"
until grep -q "articles kept" data_core/logs/wiki.log 2>/dev/null && grep -q "documents" data_core/logs/fineweb.log 2>/dev/null; do sleep 60; done
mkmix() { [ -f data_core/mix_$1_5M/manifest.json ] || { $MIX mix data_core/src data_core/mix_$1_5M --mb 400 --recipe $1 > data_core/logs/mix_$1.log 2>&1; mkdir -p data_core/mix_$1_5M/ood; cp data_v2/heldout/*.txt data_core/mix_$1_5M/ood/; }; }
mkmix D1; mkmix D2                        # these need no evidence pairs
[ -f data_core/eval/evidence_val.jsonl ] || $MIX evidence data_core/src/wiki.txt,data_core/src/fineweb.txt data_core/src/evidence.txt --max-pairs 150000 > data_core/logs/evidence.log 2>&1
mkmix D3
# report card for D0 on the CPU while the GPU finishes its queue
[ -f results/core_eval_core_5M_stage1_books_wiki_mix_D3_5M.json ] || $PY -B compiler/experiments/core_eval.py models/core_5M_stage1_books_wiki.pt --data data_core/mix_D3_5M --threads 12 > runs/core_eval_D0.log 2>&1
until [ -f results/r2b_squad_5M_answeronly.json ] || grep -lq Traceback runs/r2b.log runs/r1f.log 2>/dev/null; do sleep 60; done
for D in D1 D2 D3 D3r; do
  MIXD=${D%r}; ALIGN=--align; [ "$D" = "D3r" ] && ALIGN=""      # D3r: D3's data with random (unaligned) windows
  [ -f results/core_${D}_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_${MIXD}_5M runs/core_${D} --native "$N" --device cuda --amp \
      --sizes 5M --configs v0_8k_ng64 --budgets 3600 --max-steps 42538 --patience 0 --eval-every 300 --no-wiki $ALIGN --out results/core_${D}_5M.json > runs/core_${D}.log 2>&1
  [ -f results/core_eval_scale_5M_v0_8k_ng64_mix_${D}_5M.json ] || { $PY -B compiler/experiments/core_eval.py runs/core_${D}/scale_5M_v0_8k_ng64.pt --data data_core/mix_D3_5M --device cuda > runs/core_eval_${D}.log 2>&1;
      cp results/core_eval_scale_5M_v0_8k_ng64_mix_D3_5M.json results/core_eval_scale_5M_v0_8k_ng64_mix_${D}_5M.json; }
  [ -f results/s2_${D}_alpaca_5M.json ] || $PY -B compiler/experiments/m2_qa.py data_qa runs/s2_${D} --native "$N" --device cuda \
      --dict runs/core_${D}/v0_8k_plain.cv0d --init runs/core_${D}/scale_5M_v0_8k_ng64.pt --width 256 --layers 6 --heads 8 --ctx 256 \
      --neighbor-codes 110 --batch 32 --budget 900 --max-steps 15565 --self-context-p 0.5 --eval-n 200 --copy-head --greedy \
      --out results/s2_${D}_alpaca_5M.json --save runs/s2_${D}/qa_s2_${D}_5M.pt > runs/s2_${D}.log 2>&1
  [ -f results/probe_qa_s2_${D}_5M.json ] || $PY -B compiler/experiments/chat_probe.py runs/s2_${D}/qa_s2_${D}_5M.pt --device cuda > runs/probe_s2_${D}.log 2>&1
done
grep -l Traceback runs/core_D*.log runs/core_eval_*.log runs/s2_D*.log runs/probe_s2_*.log data_core/logs/*.log
ls results/core_D*_5M.json results/core_eval_*.json results/s2_D*_alpaca_5M.json results/probe_qa_s2_*.json
