#!/bin/bash
# Clean report cards (13 October): the anchor cores rescored on data_core/eval_clean (clean_eval.py), where no
# held-out paragraph and no evidence-gain pair occurs in any training mix. CPU only, so the 100M training keeps
# the GPU. The 100M core is rescored the same way when its training ends (session 28, first step).
PY=.venv/Scripts/python.exe
until [ -f data_core/eval_clean/manifest.json ]; do sleep 60; done
E=data_core/eval_clean/evidence_clean.jsonl
for CK in runs/core_D3s512/scale_5M_v0_8k_ng64.pt runs/core_muon_wsd512/scale_5M_v0_8k_ng64.pt runs/core_modern512/scale_5M_v0_8k_ng64.pt runs/core_P50/scale_50M_v0_8k.pt; do
  D=$(basename $(dirname $CK)); S=$(basename $CK .pt)
  [ -f results/core_eval_${D}_${S}_clean.json ] || $PY -B compiler/experiments/core_eval.py $CK --data data_core/eval_clean --evidence-file $E \
      --device cpu --threads 16 --precision bf16 --tag _clean > runs/core_eval_${D}_clean.log 2>&1
done
ls results/core_eval_*_clean.json
