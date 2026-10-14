"""Checks the lab chat (run_model_lab.py): cached replies equal the recompute replies, the window slide works,
streaming gives the same text, and the int8 option loads. CPU, 8 threads.

    .venv/Scripts/python.exe -B inference_lab/test_chat_lab.py
"""
import sys
import time
from pathlib import Path

LAB = Path(__file__).resolve().parent
sys.path.insert(0, str(LAB))
import run_model_lab as R  # noqa: E402

R.SETTINGS.update(DEVICE="cpu", THREADS=8)
R.say = lambda *a: None
MODELS = [("50M core", LAB.parent / "runs/core_P50/scale_50M_v0_8k.pt"),
          ("100M core", LAB.parent / "runs/core_P100/scale_custom_v0_8k.pt")]
short = b"The history of the city begins in the twelfth century, when"
heldout = sorted((LAB.parent / "data_core" / "eval_clean" / "heldout").glob("*.txt"))[0].read_bytes()
for name, path in MODELS:
    enc, model, dev = R.load_model(str(path))
    long = heldout[:200_000]
    long = enc.decode(enc.encode(long)[0][: model.ctx - 40])       # a prompt 40 codes short of the window
    for prompt, n in ((short, 60), (long, 120)):     # the second runs past the window: the slide is used
        if name.startswith("100M") and len(prompt) > 100:
            continue
        outs = {}
        for mode in ("recompute", "cache"):
            R.SETTINGS["CORE_DECODE"] = mode
            got = []
            t0 = time.perf_counter()
            out, stats = R.reply(enc, model, dev, prompt, n, 0.8, 40, 1, stream=got.append)
            outs[mode] = out
            print(f"{name} prompt {len(prompt)} B, n={n} {mode:9s} {time.perf_counter() - t0:6.1f}s  stream==reply: {b''.join(got) == out}  [{stats}]")
        same = outs["recompute"] == outs["cache"]
        k = next((i for i, (a, b) in enumerate(zip(outs["recompute"], outs["cache"])) if a != b), None)
        print(f"   same text: {same}" + ("" if same else f" (first difference at byte {k} of {len(outs['cache'])})"))
    print("  sample:", outs["cache"][:160].decode("utf-8", "replace").replace("\n", " "))
R.SETTINGS.update(CPU_INT8=True, CORE_DECODE="cache")
enc, model, dev = R.load_model(str(MODELS[0][1]))
out, stats = R.reply(enc, model, dev, short, 60, 0.8, 40, 1)
print("int8 50M:", stats, "|", out[:100].decode("utf-8", "replace").replace("\n", " "))
