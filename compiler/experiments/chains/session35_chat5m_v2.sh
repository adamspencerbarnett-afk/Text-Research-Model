#!/bin/bash
# Session 35 (after session 34; Adam, 15 October): the 5M chat model v2, "the best of everything we have learned",
# adapted after the reasoning-research review into the A/B/C(/D) experiment. Design: docs/CHAT5M_V2_DESIGN.html.
#   A  one mixed stage (Q&A 30%, skills v4 25%, conversations 15%, long recall 10%, rehearsal 20%), window 2,048,
#      30,000 steps, copy head over memory AND earlier turns; audited data (train_v4c, deduplicated conversations)
#   B  A + intermediate targets (--aux 0.5: the copy head is told which stored fact / earlier turn, which words)
#   C  B + the verifier at answer time (5 candidate replies, the best supported by the evidence shown); no training
#   D  B + one self-training round (2,500 verifiable questions x 4 answers; correct and supported answers become
#      targets; 4,000 steps from B at a lower rate)
# Every model: chat_tests.py (A/B/C comparison with the verifier off and on, speed vs window on GPU and CPU, quality by
# position to 4,096, recall after 2-40 turns with/without session memory, 8 stories, conversation probes as chats),
# the clean report card; the 50 questions for hand rating (B with the verifier, D with the verifier).
PY=.venv/Scripts/python.exe
until [ -f results/session34_done ]; do sleep 300; done
# wait until no Python job uses the GPU (Windows always lists desktop programs such as Explorer as GPU users)
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 120; done
O=runs/chat5m_v2; mkdir -p $O
tr() { [ -f $O/$1.pt ] || $PY -B compiler/experiments/train_chat.py --out $O --name "$@" --amp > $O/train_$1.log 2>&1; }
tr chat5m_v2_A --steps 30000
tr chat5m_v2_B --steps 30000 --aux 0.5
[ -f $O/selftrain.jsonl ] || $PY -B compiler/experiments/selftrain_chat.py $O/chat5m_v2_B.pt --n 2500 --k 4 --out $O/selftrain.jsonl > $O/selftrain.log 2>&1
tr chat5m_v2_D --steps 4000 --init $O/chat5m_v2_B.pt --lr 3e-5 --head-lr 3e-4 --aux 0.5 --extra-data $O/selftrain.jsonl --extra-p 0.3
E=data_core/eval_clean/evidence_clean.jsonl
for M in A B D; do
  P=$O/chat5m_v2_$M.pt
  [ -f results/chat_tests_chat5m_v2_$M.json ] || $PY -B compiler/experiments/chat_tests.py $P --device cuda --cpu-threads 16 --verify 5 > runs/chat_tests_chat5m_v2_$M.log 2>&1
  [ -f results/core_eval_chat5m_v2_chat5m_v2_${M}_clean.json ] || $PY -B compiler/experiments/core_eval.py $P --data data_core/eval_clean \
      --evidence-file $E --device cuda --precision bf16 --tag _clean > runs/card_chat5m_v2_$M.log 2>&1
done
cp -n $O/chat5m_v2_B.pt models/chat5m_v2.pt
PYTHONIOENCODING=utf-8 $PY -B skills_lab/manual/run_50.py $O/chat5m_v2_B.pt answers_chat5m_v2_B_verify5 5 > skills_lab/manual/answers_chat5m_v2_B_verify5.txt 2>&1
PYTHONIOENCODING=utf-8 $PY -B skills_lab/manual/run_50.py $O/chat5m_v2_D.pt answers_chat5m_v2_D_verify5 5 > skills_lab/manual/answers_chat5m_v2_D_verify5.txt 2>&1
touch results/session35_done
ls results/chat_tests_chat5m_v2_*.json
