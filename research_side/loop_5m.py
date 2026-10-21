"""The 5M improvement loop (20 October 2026, Adam: "loop the 5M to get the best results we can before the 100M").

Reads recipes/LOOP.json and runs its rounds in order, one at a time, each = the BASE recipe of recipes/BEST.json plus ONE
change (a --parts switch, another data folder, other shares):
  seed 1   train, full suite (engine v2), the new tests (when LOOP.json lists them), compare with the best's two seeds
  seed 2   only when seed 1 wins with no front lost; then the two-seed comparison
  adopt    a two-seed NEW BEST becomes BEST and BASE in recipes/BEST.json (history kept), so every later round builds on
           it; otherwise the round is parked with its verdict
A round waits until it is marked ready (its code exists and passed a smoke test). A failed step stops the loop (status
"failed") for a person to look at. Progress: runs/loop_5m.log and the status fields in recipes/LOOP.json.

    nohup .venv/Scripts/python.exe -u research_side/loop_5m.py > runs/loop_5m.log 2>&1 &
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "Scripts" / "python.exe") if (ROOT / ".venv" / "Scripts").exists() else str(ROOT / ".venv" / "bin" / "python")
LOOP, BEST = ROOT / "recipes" / "LOOP.json", ROOT / "recipes" / "BEST.json"


def say(*a):
    print(time.strftime("%m-%d %H:%M:%S"), *a, flush=True)


def jload(p):
    return json.loads(p.read_text(encoding="utf-8"))


def jsave(p, d):
    p.write_text(json.dumps(d, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def set_round(rid, **kv):
    d = jload(LOOP)
    for r in d["rounds"]:
        if r["id"] == rid:
            r.update(kv)
    jsave(LOOP, d)


def training_running() -> bool:
    """Any chat22 training on the machine. On Windows the process list comes from PowerShell: bash's pgrep does not see
    trainings started by another (e.g. an earlier) driver (20 Oct: a restarted driver started a duplicate run)."""
    if sys.platform == "win32":
        cmd = ["powershell", "-NoProfile", "-Command",
               "(Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { $_.CommandLine -match 'chat22.train' }).Count"]
        out = subprocess.run(cmd, capture_output=True, text=True).stdout.strip()
        return out.isdigit() and int(out) > 0
    out = subprocess.run(["bash", "-c", "pgrep -f 'chat22.train' || true"], capture_output=True, text=True).stdout.strip()
    return bool(out)


def run(cmd: list, log: Path) -> int:
    say("run", " ".join(cmd[2:])[:200], "->", log.name)
    with open(log, "a", encoding="utf-8") as f:
        return subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode


def train_args(base: dict, change: dict) -> list:
    t = dict(base)
    parts = [p for p in t["parts"].split(",") if p]
    parts += [p for p in change.get("parts_add", []) if p not in parts]
    t["parts"] = ",".join(parts)
    if "data_dir" in change:
        t["data_dir"] = change["data_dir"]
    if "shares" in change:
        t["shares"] = change["shares"]
    args = ["--core", t["core"], "--mix", t["mix"], "--parts", t["parts"], "--shares", json.dumps(t["shares"]),
            "--min-steps", str(t["min_steps"]), "--max-steps", str(t["max_steps"]), "--patience", str(t["patience"]),
            "--memory-opts", json.dumps(t["memory_opts"]), "--gpu-frac", str(t["gpu_frac"])]
    if t.get("data_dir"):
        args += ["--data-dir", t["data_dir"]]
    return args + list(change.get("extra_args", [])), t


def test_model(name: str, new_tests: list, engine_opts: dict | None = None) -> bool:
    """Full suite (+ the new tests) with the best's answer-time settings, or the round's own (a round that also changes
    the answer path, e.g. the two-hop search, is tested with it and, if adopted, its settings become the best's)."""
    eo = json.dumps(engine_opts or jload(BEST)["answer_time"].get("engine_opts", {"search": "v2"}))
    if not (ROOT / "results" / f"suite_{name}_ev2full.json").exists():
        run([PY, "-B", "-m", "chat22.tests.run_suite", f"runs/chat22/{name}.pt", "--tier", "full", "--engine-opts", eo,
             "--tag", "_ev2full"], ROOT / "runs" / f"suite_{name}_ev2full.log")
    if new_tests and not (ROOT / "results" / f"suite_{name}_ev2new.json").exists():
        run([PY, "-B", "-m", "chat22.tests.run_suite", f"runs/chat22/{name}.pt", "--tier", "full", "--engine-opts", eo,
             "--sections", ",".join(new_tests), "--tag", "_ev2new"], ROOT / "runs" / f"suite_{name}_ev2new.log")
    return (ROOT / "results" / f"suite_{name}_ev2full.json").exists()


def compare(names: list) -> tuple[int, int, str]:
    log = ROOT / "runs" / f"compare_{names[-1]}.log"
    out = subprocess.run([PY, "research_side/compare_two_seeds.py", *names], cwd=ROOT, capture_output=True, text=True).stdout
    log.write_text(out, encoding="utf-8")
    m = re.search(r"significant wins (\d+), losses (\d+)", out)
    return (int(m.group(1)), int(m.group(2)), out.strip().splitlines()[-1]) if m else (0, 99, "compare failed")


def train_seed(r: dict, seed: int, base: dict) -> str | None:
    name = f"{r['name']}_s{seed}"
    done = lambda: (ROOT / "results" / f"chat22_train_{name}.json").exists()
    if not done():
        while training_running():
            time.sleep(60)
    if not done():                                       # (checked again: the run may have been the one waited for)
        args, _ = train_args(base, r["change"])
        rc = run([PY, "-B", "-m", "chat22.train", "--name", name, "--seed", str(seed), *args], ROOT / "runs" / f"chat22_train_{name}.log")
        if rc != 0 or not (ROOT / "results" / f"chat22_train_{name}.json").exists():
            return None
    return name


def adopt(r: dict, names: list, verdict: str):
    b = jload(BEST)
    args, t = train_args(b["base"]["train"], r["change"])
    old = {k: b["best"][k] for k in ("version", "name", "model", "seeds_confirmed") if k in b["best"]}
    b["base"]["train"] = t
    if r.get("engine_opts"):
        b["answer_time"]["engine_opts"] = r["engine_opts"]
    b["best"] = {"version": int(b["best"].get("version", 1)) + 1, "name": f"{r['id']}: {r['what']}", "date": time.strftime("%Y-%m-%d"),
                 "model": f"runs/chat22/{names[0]}.pt", "seeds_confirmed": names, "train": t, "vs_previous_best": verdict,
                 "model_note": "set by the loop (seed 1's model); a person may pick the seed to ship after reading both"}
    b.setdefault("history", []).append({"date": time.strftime("%Y-%m-%d"), "name": f"{r['id']}: {r['what']}", "status": "adopted by the loop",
                                        "verdict": verdict, "replaced": old})
    jsave(BEST, b)


def main() -> int:
    say("loop started")
    while True:
        d = jload(LOOP)
        if d.get("paused"):
            say("loop is paused (recipes/LOOP.json 'paused'); nothing started"); return 0
        todo = [r for r in d["rounds"] if r.get("status") not in ("adopted", "parked", "failed", "skipped")
                and not str(r.get("status", "")).startswith("seed 1 judged; loop paused")]
        if not todo:
            say("all rounds done"); return 0
        r = todo[0]
        if not r.get("ready"):
            time.sleep(120); continue
        new_tests = d.get("new_tests", [])
        base = jload(BEST)["base"]["train"]
        if r.get("external_seed1"):                      # seed 1 trained and tested by its own chain
            flag = ROOT / "results" / f"{r['external_seed1']}_done"
            if not flag.exists():
                time.sleep(120); continue
            s1 = f"{r['name']}_s1"
        else:
            set_round(r["id"], status="seed 1 training")
            s1 = train_seed(r, 1, base)
            if s1 is None:
                set_round(r["id"], status="failed", note="seed 1 training failed"); say(r["id"], "seed 1 training FAILED"); return 1
        set_round(r["id"], status="seed 1 testing")
        if not test_model(s1, new_tests, r.get("engine_opts")):
            set_round(r["id"], status="failed", note="seed 1 suite failed"); say(r["id"], "seed 1 suite FAILED"); return 1
        w, l, line = compare([s1])
        set_round(r["id"], seed1=line, status="seed 1 judged")
        say(r["id"], "seed 1:", line)
        if jload(LOOP).get("pause_after_seed1"):         # Adam (20 Oct): stop after this verdict for his review
            set_round(r["id"], status="seed 1 judged; loop paused for Adam's review", verdict=line)
            d2 = jload(LOOP); d2["paused"] = True; jsave(LOOP, d2)
            say("paused for review after", r["id"], "seed 1"); return 0
        if not (l == 0 and w > 0):
            set_round(r["id"], status="parked", verdict=line); continue
        set_round(r["id"], status="seed 2 training")
        s2 = train_seed(r, 2, base)
        if s2 is None:
            set_round(r["id"], status="failed", note="seed 2 training failed"); say(r["id"], "seed 2 training FAILED"); return 1
        set_round(r["id"], status="seed 2 testing")
        if not test_model(s2, new_tests, r.get("engine_opts")):
            set_round(r["id"], status="failed", note="seed 2 suite failed"); say(r["id"], "seed 2 suite FAILED"); return 1
        test_model(s1, new_tests, r.get("engine_opts"))  # tests added after seed 1 was judged run on it too
        w, l, line = compare([s1, s2])
        say(r["id"], "two seeds:", line)
        if l == 0 and w > 0:
            adopt(r, [s1, s2], line)
            set_round(r["id"], status="adopted", verdict=line); say(r["id"], "ADOPTED: new BEST and BASE")
        else:
            set_round(r["id"], status="parked", verdict=line)


if __name__ == "__main__":
    sys.exit(main())
