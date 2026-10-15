#!/bin/bash
# Session 35 (after session 34; Adam, 15 October): the 5M chat model v2, "the best of everything we have learned".
# docs/CHAT5M_V2_DESIGN.html has the design. From the new 5M core (runs/core_mt5m), window 2,048 codes, one stage of
# Q&A + skills v4 + real conversations + long-conversation recall + rehearsal (or two stages if session 33 picks the
# sequential structure: plan_structure.py), then every test: chat_tests.py (speed and limits of the long window,
# remembering a whole conversation, short stories, conversation probes as real chats), the skills tests v1 / v2 /
# rewriting, the grounded scorecard, the chat and typo probes, the clean report card, mt_probe, and the 50 questions
# for hand rating.
PY=.venv/Scripts/python.exe
until [ -f results/session34_done ]; do sleep 300; done
# wait until no Python job uses the GPU (Windows always lists desktop programs such as Explorer as GPU users)
until ! nvidia-smi --query-compute-apps=process_name --format=csv,noheader | grep -qi python; do sleep 120; done
O=runs/chat5m_v2; M=$O/chat5m_v2.pt; mkdir -p $O
STRUCT=$($PY -B compiler/experiments/plan_structure.py | tail -1)
echo "structure: $STRUCT"
if [ ! -f $M ]; then
  if [ "$STRUCT" = "sequential" ]; then
    [ -f $O/chat5m_v2a.pt ] || $PY -B compiler/experiments/train_chat.py --out $O --name chat5m_v2a --stage a --steps 18000 --amp > $O/train_a.log 2>&1
    $PY -B compiler/experiments/train_chat.py --out $O --name chat5m_v2 --stage b --steps 12000 --init $O/chat5m_v2a.pt --amp > $O/train_b.log 2>&1
  else
    $PY -B compiler/experiments/train_chat.py --out $O --name chat5m_v2 --stage mixed --steps 30000 --amp > $O/train.log 2>&1
  fi
fi
cp -n $M models/chat5m_v2.pt
E=data_core/eval_clean/evidence_clean.jsonl
[ -f results/chat_tests_chat5m_v2.json ] || $PY -B compiler/experiments/chat_tests.py $M --device cuda --cpu-threads 16 > runs/chat_tests_chat5m_v2.log 2>&1
ev() { [ -f skills_lab/results/$1.json ] || $PY -B skills_lab/eval_skills.py "${@:2}" --name $1 --device cuda > skills_lab/results/$1.log 2>&1; }
ev chat5m_v2 $M --modes orig,k1,k3close
ev chat5m_v2_v2 $M --modes k1,k3close --testset data_skills/test_v2.json
ev chat5m_v2_rw $M --modes k1,k3close --testset data_skills/test_v4_rewrite.json
[ -f skills_lab/results/grounded_chat5m_v2.json ] || $PY -B skills_lab/grounded.py $M --name chat5m_v2 --mode k3close --device cuda > skills_lab/results/grounded_chat5m_v2.log 2>&1
[ -f results/probe_chat5m_v2.json ] || $PY -B compiler/experiments/chat_probe.py $M --device cuda --graph --precision bf16 > runs/probe_chat5m_v2.log 2>&1
[ -f results/probe_chat5m_v2_typo.json ] || $PY -B compiler/experiments/chat_probe.py $M --device cuda --graph --precision bf16 --typos 0.15 --tag _typo > runs/probe_chat5m_v2_typo.log 2>&1
[ -f results/mtprobe_chat5m_v2.json ] || $PY -B compiler/experiments/mt_probe.py $M --device cuda --graph --precision bf16 > runs/mtprobe_chat5m_v2.log 2>&1
[ -f results/core_eval_chat5m_v2_chat5m_v2_clean.json ] || $PY -B compiler/experiments/core_eval.py $M --data data_core/eval_clean --evidence-file $E \
    --device cuda --precision bf16 --tag _clean > runs/card_chat5m_v2.log 2>&1
PYTHONIOENCODING=utf-8 $PY -B skills_lab/manual/run_50.py $M answers_chat5m_v2 > skills_lab/manual/answers_chat5m_v2.txt 2>&1
touch results/session35_done
ls results/chat_tests_chat5m_v2.json skills_lab/results/chat5m_v2*.json
