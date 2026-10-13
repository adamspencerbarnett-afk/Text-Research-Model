#!/bin/bash
# Session 15 (after session 14): reasoning and reading at 50M, plus the cache speed and the gate threshold.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2c_50m_gate05.json ] || grep -lq Traceback runs/s2c_50m.log runs/probe_s2c_50m*.log 2>/dev/null; do sleep 120; done
BEST50=runs/s2c_50m/qa_s2c_50m.pt; [ -f $BEST50 ] || BEST50=runs/s2b_50m/qa_s2b_50m.pt
# 1. gate threshold
for g in 04 045; do
  [ -f results/probe_qa_s2c_50m_gate$g.json ] || $PY -B compiler/experiments/chat_probe.py $BEST50 --device cuda --min-sim 0.${g#0} --tag _gate$g > runs/probe_s2c_50m_gate$g.log 2>&1
done
# 5. cache speed at 50M
[ -f results/kv_speed_$(basename $BEST50 .pt).json ] || $PY -B compiler/experiments/kv_speed.py $BEST50 --device cuda > runs/kv_speed_50m.log 2>&1
S2=(--native "$N" --device cuda --batch 32 --budget 16000 --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3
    --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 --dict runs/core_P50/v0_8k_plain.cv0d --width 640 --layers 10 --heads 10 --ctx 1024
    --neighbor-codes 400 --rehearse data_core/mix_P50)
# 2. math at 50M
mkdir -p runs/s2b_gsm_50m
[ -f results/s2b_gsm_50m.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm runs/s2b_gsm_50m "${S2[@]}" --max-steps 8000 --copy-question --calculator \
    --init runs/core_P50/scale_50M_v0_8k.pt --out results/s2b_gsm_50m.json --save runs/s2b_gsm_50m/qa_s2b_gsm_50m.pt > runs/s2b_gsm_50m.log 2>&1
# 3. one round of verified self-training on the 50M math model
D=runs/star50_r1; mkdir -p $D
if [ ! -f data_qa_gsm_star50_r1/star_manifest.json ]; then
  for s in $(seq 0 9); do [ -f $D/gen_$s.done ] || ( $PY -B compiler/experiments/star.py gen runs/s2b_gsm_50m/qa_s2b_gsm_50m.pt --shard $s --of 10 --k 4 --out $D/gen_$s.jsonl > $D/gen_$s.log 2>&1 && touch $D/gen_$s.done ) & done
  wait
  $PY -B compiler/experiments/star.py merge $D data_qa_gsm data_qa_gsm_star50_r1 > $D/merge.log 2>&1
fi
[ -f results/star50_r1.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm_star50_r1 $D/train "${S2[@]}" --max-steps 8000 --copy-question --calculator \
    --init runs/s2b_gsm_50m/qa_s2b_gsm_50m.pt --out results/star50_r1.json --save $D/qa_star50_r1.pt > $D/train.log 2>&1
# 4. reading comprehension at 50M
mkdir -p runs/s2_squad_50m
[ -f results/s2_squad_50m.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_squad runs/s2_squad_50m --native "$N" --device cuda --batch 32 --budget 16000 \
    --max-steps 10000 --eval-n 200 --net-n 200 --copy-head --copy-question --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos \
    --stop-at-blank --rehearse data_core/mix_P50 --rehearse-p 0.3 --dict runs/core_P50/v0_8k_plain.cv0d --init runs/core_P50/scale_50M_v0_8k.pt \
    --width 640 --layers 10 --heads 10 --ctx 1024 --neighbor-codes 0 --out results/s2_squad_50m.json --save runs/s2_squad_50m/qa_squad_50m.pt > runs/s2_squad_50m.log 2>&1
ls results/probe_qa_s2c_50m_gate04*.json results/kv_speed_*.json results/s2b_gsm_50m.json results/star50_r1.json results/s2_squad_50m.json
