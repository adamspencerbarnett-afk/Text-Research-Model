"""Engine checks (no training needed; any chat22 checkpoint):

  cache      the same conversation replayed with the history cache on and off gives IDENTICAL replies at every turn,
             through window scroll-out (long turns), memory-block changes, forget(), and verify > 1
  memory     a fact told early comes back as a [turn N] item once its turn has left the window; forget() clears it;
             add_fact() stores a fact that survives forget()
  order      memory items are placed facts first, then turns oldest to newest (the newest nearest the message)

    python -m chat22.tests.test_engine runs/chat22_smoke/smoke.pt [--device cpu]
"""
from __future__ import annotations

import argparse
import sys

from chat22 import window as W
from chat22.engine import Engine

LONG = (b"Here is something long about my week. " * 40).strip()


def conversation(eng, verify=0):
    out = []
    script = [b"My sister Kim lives in Oslo.", b"What is the capital of France?", LONG, LONG, LONG, LONG,
              b"Where does my sister live?", b"What is 12 plus 30?"]
    for j, m in enumerate(script):
        r, info = eng.reply(m, 40, verify=verify)
        out.append((r, info["items"], info["turns_in_window"]))
        eng.remember(m, r if r else b"Okay.")
        if j == 5:
            out.append(("reused", eng.last_reused))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--device", default="cpu")
    a = ap.parse_args()
    eng = Engine(a.model, a.device, dense=False)
    ok = True
    for verify in (0, 3):
        eng.forget(); eng.reuse = True; on = conversation(eng, verify)
        eng.forget(); eng.reuse = False; off = conversation(eng, verify)
        same = [x[:3] == y[:3] if x[0] != "reused" else True for x, y in zip(on, off)]
        reused = [x[1] for x in on if x[0] == "reused"]
        print(f"cache verify={verify}: identical replies {sum(same)}/{len(same)}; codes reused at turn 6: {reused}")
        ok &= all(same)
    eng.reuse = True; eng.forget()
    eng.remember(b"My sister Kim lives in Oslo.", b"Got it.")
    for _ in range(12):
        eng.remember(LONG, b"Okay.")
    _, info = eng.reply(b"Where does my sister Kim live?", 30)
    back = any("Oslo" in t and t.startswith("[turn 1]") for t in info["items"])
    print(f"memory: fact from turn 1 returned after 12 long turns: {back} (turns in window {info['turns_in_window']})")
    eng.add_fact(b"Who leads the research?", b"Adam Barnett leads the research.")
    eng.forget()
    _, info = eng.reply(b"Where does my sister Kim live?", 30)
    gone = not any("Oslo" in t for t in info["items"])
    _, info2 = eng.reply(b"Who leads the research?", 30)
    kept = any("Adam Barnett" in t for t in info2["items"])
    print(f"memory: forget() cleared the session: {gone}; add_fact survived forget(): {kept}")
    items = [W.Item(b"q3", b"a3", "turn", 9), W.Item(b"f1", b"x", "fact"), W.Item(b"q1", b"a1", "turn", 2), W.Item(b"f2", b"y", "fact")]
    order = [items[j] for j in W.place_order(items, [0, 1, 2, 3])]
    good_order = [(i.source, i.turn) for i in order] == [("fact", 0), ("fact", 0), ("turn", 2), ("turn", 9)]
    print(f"order: facts first, then turns oldest to newest: {good_order}")
    ok &= back and gone and kept and good_order
    print("ENGINE OK" if ok else "ENGINE CHECK FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
