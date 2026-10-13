#!/bin/bash
# Re-measure answer speed at 50M on an idle GPU (session 15's training shares the card until it ends).
until [ -f results/s2_squad_50m.json ] || grep -lq Traceback runs/s2_squad_50m.log 2>/dev/null; do sleep 120; done
sleep 60
.venv/Scripts/python.exe -B -X utf8 compiler/experiments/kv_speed.py runs/s2c_50m/qa_s2c_50m.pt --device cuda > runs/kv_speed_50m_idle.log 2>&1
.venv/Scripts/python.exe -B -X utf8 compiler/experiments/kv_speed.py models/qa_best_5M_512.pt --device cuda > runs/kv_speed_5m_idle.log 2>&1
tail -n1 runs/kv_speed_50m_idle.log runs/kv_speed_5m_idle.log
