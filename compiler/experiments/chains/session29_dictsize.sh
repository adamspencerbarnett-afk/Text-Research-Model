#!/bin/bash
# Session 29 (after session 28): dictionary size, stage 2 (HYBRID_DESIGN item 16). Does a 16k, 32k or 64k
# dictionary give faster answers without losing intelligence?
# Stage 1 (dict_size.py, CPU) counted codes per KB of held-out text for each size. Here each size trains the
# 5M core exactly as the 8k baseline (D3s512: 256 x 6, legacy block, count tables, window 512, batch 64,
# AdamW + cosine, bf16) on the SAME TEXT: 4.88 GB seen, so its step count is 42,538 x (its codes per KB /
# the 8k codes per KB). The core has the same 4.7M parameters at every size; the embedding grows with the
# vocabulary (8k: 2.2M, 64k: 16.8M) and is reported separately. Then the adopted Q&A stage (dense retrieval
# in training + reworded self windows, bf16) and the chat probe at bf16 and int8 with seconds and joules
# per answer. Compared with the 8k baseline's two seeds (s2b_densep512: probe 24 and 23/32; 0.083 s,
# 11.8 J per answer). Window and memory span stay in codes (512, 250), so larger codes also carry more text.
PY=.venv/Scripts/python.exe
N="$(pwd -W)/data/cv0.exe"
until [ -f results/probe_qa_s2b_confall512_int8.json ] || grep -lq Traceback runs/s2b_conf512.log runs/s2b_confall512.log 2>/dev/null; do sleep 120; done
for K in 16k 32k 64k; do
  STEPS=$($PY -I -c "import json; d=json.load(open('results/dict_size_codes.json'))['sizes']; print(round(42538 * d['v0_${K}_plain']['codes_per_kb'] / d['v0_8k_plain']['codes_per_kb']))")
  C=runs/core_dict$K; Q=runs/s2b_dict$K; mkdir -p $C $Q
  cp -n runs/dict_size/v0_${K}_plain.cv0d $C/ && cp -n runs/s2b_dense512/neighbors_5bd4da54591f44cb.json $Q/ 2>/dev/null
  [ -f results/core_dict${K}_5M.json ] || $PY -B compiler/experiments/scale_run.py data_core/mix_D3s_5M $C --native "$N" --device cuda --amp \
      --sizes 5M --configs v0_${K}_ng64 --budgets 7200 --max-steps $STEPS --patience 0 --eval-every 300 --no-wiki --ctx 512 \
      --out results/core_dict${K}_5M.json > $C.log 2>&1
  [ -f results/core_eval_core_dict${K}_scale_5M_v0_${K}_ng64.json ] || $PY -B compiler/experiments/core_eval.py $C/scale_5M_v0_${K}_ng64.pt \
      --data data_core/mix_D3_5M --device cuda --precision bf16 > runs/core_eval_dict$K.log 2>&1
  [ -f results/s2b_dict$K.json ] || $PY -B compiler/experiments/m2_qa.py data_qa $Q --native "$N" --device cuda --batch 32 --max-steps 15565 --budget 3600 \
      --self-context-p 0.5 --eval-n 200 --net-n 200 --copy-head --greedy --lr 1e-4 --head-lr 1e-3 --schedule cosine --loss-answer-only --eos --rehearse-p 0.3 \
      --dict $C/v0_${K}_plain.cv0d --init $C/scale_5M_v0_${K}_ng64.pt --width 256 --layers 6 --heads 8 --ctx 512 --neighbor-codes 250 \
      --dense 2 --dense-train --perturb-self 0.3 --amp --rehearse data_core/mix_D3s_5M --out results/s2b_dict$K.json --save $Q/qa_s2b_dict$K.pt > $Q.log 2>&1
  [ -f results/probe_qa_s2b_dict$K.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_dict$K.pt --device cuda --graph --precision bf16 > runs/probe_s2b_dict$K.log 2>&1
  [ -f results/probe_qa_s2b_dict${K}_int8.json ] || $PY -B compiler/experiments/chat_probe.py $Q/qa_s2b_dict$K.pt --device cuda --graph --precision int8 --tag _int8 > runs/probe_s2b_dict${K}_int8.log 2>&1
done
ls results/core_dict*_5M.json results/s2b_dict*.json results/probe_qa_s2b_dict*.json
