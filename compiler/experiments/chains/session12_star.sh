#!/bin/bash
# Session 12: verified self-training (STaR) on GSM8K, two rounds. Each round: 10 parallel generator
# shards (4 sampled solutions per training problem, own solution hidden, calculator on, kept only if the
# final answer checks) -> merge into a training folder -> fine-tune (S2b recipe, 8,000 steps) -> score.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
SHARDS=10
round() {  # round number, model to generate from
  local r=$1 model=$2 dir=runs/star_r$1
  mkdir -p $dir
  if [ ! -f data_qa_gsm_star_r$r/star_manifest.json ]; then
    for s in $(seq 0 $((SHARDS-1))); do
      [ -f $dir/gen_$s.done ] || ( $PY -B compiler/experiments/star.py gen $model --shard $s --of $SHARDS --k 4 --out $dir/gen_$s.jsonl > $dir/gen_$s.log 2>&1 && touch $dir/gen_$s.done ) &
    done
    wait
    $PY -B compiler/experiments/star.py merge $dir data_qa_gsm data_qa_gsm_star_r$r > $dir/merge.log 2>&1
  fi
  [ -f results/star_r$r.json ] || $PY -B compiler/experiments/m2_qa.py data_qa_gsm_star_r$r $dir/train --native "$N" --device cuda --batch 32 --max-steps 8000 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --copy-question --calculator --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine \
      --loss-answer-only --eos --rehearse-p 0.3 --dict runs/core_D3s512/v0_8k_plain.cv0d --init $model \
      --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 --rehearse data_core/mix_D3s_5M --out results/star_r$r.json --save $dir/qa_star_r$r.pt > $dir/train.log 2>&1
}
round 1 runs/s2b_gsm/qa_s2b_gsm_512.pt
round 2 runs/star_r1/qa_star_r1.pt
ls results/star_r1.json results/star_r2.json
