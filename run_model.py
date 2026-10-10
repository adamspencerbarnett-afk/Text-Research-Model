"""Train a model on a folder of text files, or chat with a saved model. Menu driven.

    python run_model.py              # menu: train on a folder, or chat with a model
    python run_model.py --train DIR  # skip the menu
    python run_model.py --chat MODEL.pt

Training: pick a folder in the popup. Every *.txt file in it is training text (or, if the
folder has train/, heldout/ and ood/ subfolders, those are used as the research corpora are).
The dictionary is fitted on the training text, the model trains on the GPU when there is one,
and the model, its result file and the dictionary are saved in MODELS_SUBDIR inside that
folder. Chat: pick a .pt file in the popup; the encoder and the count tables are rebuilt from
the text the model was trained on, so keep the dataset folder where it was.

Settings are the SETTINGS block below. The research harness does the work
(compiler/experiments/train_lm.py), so these models are directly comparable with the results
in results/ and docs/.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------- SETTINGS
SETTINGS = dict(
    # --- model shape (research sizes: 128w/4L = 0.8M core; 256w/6L = 4.7M; 384w/8L = 14M; 512w/8L = 25M)
    WIDTH=256, LAYERS=6, HEADS=8, CTX=256,
    # --- encoding: "compiler" (the deterministic v0 dictionary) or "bpe" (standard tokenizer)
    ENCODING="compiler", VOCAB=8192,
    # --- exact n-gram count tables mixed into the output by the trust head; () for a plain model
    NGRAM_ORDERS=(2, 3), NGRAM_TOPK=64,
    # --- training: stop at BUDGET_S seconds of training, or earlier when the validation score
    #     has not improved by MIN_DELTA in PATIENCE evaluations (PATIENCE=0 never)
    BUDGET_S=600, EVAL_EVERY_S=60, PATIENCE=3, MIN_DELTA=0.002,
    BATCH=32, LR=1e-3, SEED=1,
    VALID_SHARE=0.02,       # last share of the training stream kept for validation, never trained on
    HELDOUT_FILES=1,        # files (alphabetically last) scored as held-out when there is no heldout/ folder
    # --- hardware: "auto" = GPU if available, else "cpu", or "cuda:0"
    DEVICE="auto", THREADS=0,
    # --- chat
    REPLY_TOKENS=120, TEMPERATURE=0.8, TOP_K=40,
    # --- chat with a Q&A model (memory + copy head; trained by compiler/experiments/m2_qa.py)
    QA_MAX_CODES=120,       # longest reply, in codes (about 3.5 bytes each)
    QA_NO_REPEAT=3,         # block any 3-code phrase from repeating inside a reply (0: off)
    QA_SHOW_MEMORY=True,    # print which stored question the answer was built from
    # --- places
    MODELS_SUBDIR="models",                    # created inside the dataset folder
    START_DIR=str(Path.home() / "OneDrive" / "Desktop" / "Text Files"),   # where the popups open
)
# ----------------------------------------------------------------------------- end of settings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import compiler_v0 as cv  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import train_lm  # noqa: E402


def say(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def pick_folder(title: str) -> str | None:
    from tkinter import Tk, filedialog
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title, initialdir=SETTINGS["START_DIR"], mustexist=True)
    root.destroy()
    return path or None


def pick_file(title: str, patterns) -> str | None:
    from tkinter import Tk, filedialog
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, initialdir=SETTINGS["START_DIR"], filetypes=patterns)
    root.destroy()
    return path or None


def device() -> str:
    d = SETTINGS["DEVICE"]
    if d == "auto":
        d = "cuda" if torch.cuda.is_available() else "cpu"
    if d.startswith("cuda") and not torch.cuda.is_available():
        say("no CUDA device found, using the CPU"); d = "cpu"
    if d.startswith("cuda"):
        say(f"GPU: {torch.cuda.get_device_name(d)}")
    return d


def native_encoder() -> str | None:
    """The fast C++ encoder (built on first use when a compiler is present); None means the
    Python encoder is used, which is fine for small datasets and slow for large ones."""
    exe = ROOT / "data" / ("cv0.exe" if os.name == "nt" else "cv0")
    if exe.exists():
        return str(exe)
    exe.parent.mkdir(exist_ok=True)
    src = ROOT / "compiler" / "native" / "cv0.cpp"
    for cc in ("g++", "clang++"):
        if shutil.which(cc):
            say(f"building the native encoder with {cc}")
            subprocess.run([cc, "-O2", "-std=c++17", "-o", str(exe), str(src)], check=True)
            return str(exe)
    say("no C++ compiler on the PATH: using the Python encoder (slow on large datasets)")
    return None


def dataset_files(folder: Path) -> dict[str, list[str]]:
    """Training, held-out and out-of-domain text files of a dataset folder."""
    if (folder / "train").is_dir():
        sets = {s: sorted(str(p) for p in (folder / s).glob("*.txt")) for s in ("train", "heldout", "ood")}
    else:
        files = sorted(str(p) for p in folder.glob("*.txt"))
        k = min(SETTINGS["HELDOUT_FILES"], max(0, len(files) - 1))
        sets = {"train": files[:len(files) - k] if k else files, "heldout": files[len(files) - k:] if k else [], "ood": []}
    if not sets["train"]:
        raise SystemExit(f"no .txt files to train on in {folder}")
    return sets


def load_tokens(enc: LE.Encoding, files: list[str], work: Path):
    """Concatenate the files in order (cached) and encode them with the dataset's encoder."""
    concat = work / "train_concat.txt"
    if not concat.exists():
        concat.write_bytes(b"".join(Path(p).read_bytes() for p in files))
    return enc.encode_file(str(concat), work)


def make_encoder(kind: str, vocab: int, train_files: list[str], work: Path, native: str | None) -> LE.Encoding:
    """Fit (once, on the training text only) and load the encoder named by the settings."""
    texts = lambda: (Path(p).read_bytes() for p in train_files)
    if kind == "compiler":
        path = work / f"v0_{vocab // 1024}k_plain.cv0d"
        if not path.exists():
            t0 = time.time()
            d, _ = cv.fit(texts, vocab, 0, 3, case_flags=False)
            d.save(str(path))
            say(f"fitted dictionary {path.name}: {d.vocab_size:,} IDs ({time.time() - t0:.0f}s)")
        return LE.V0Dict(str(path), native)
    if kind == "bpe":
        name = f"bpe_{vocab // 1024}k"
        path = work / f"{name}.merges.json"
        if not path.exists():
            t0 = time.time()
            LE.StandardBPE.fit(texts(), vocab, name).save(str(path))
            say(f"fitted {name}: {vocab:,} IDs ({time.time() - t0:.0f}s)")
        return LE.StandardBPE.load(str(path))
    raise SystemExit(f"ENCODING must be 'compiler' or 'bpe', not {kind!r}")


def load_encoder(info: dict, work: Path, native: str | None) -> LE.Encoding:
    """Rebuild a saved model's encoder from the checkpoint's description."""
    if info["name"] == "bytes":
        return LE.Bytes()
    if info.get("groups"):
        return LE.HashCodes.load(str(work / f"{info['name']}.json"))
    if info["name"].startswith("bpe_"):
        return LE.StandardBPE.load(str(work / f"{info['name']}.merges.json"))
    dict_path = Path(info["dictionary"])
    if not dict_path.exists():                 # the dataset folder was moved: look beside the model
        dict_path = work / dict_path.name
    return LE.V0Dict(str(dict_path), native)


# ----------------------------------------------------------------------------- train

def train(folder: str) -> Path:
    S = SETTINGS
    folder = Path(folder)
    sets = dataset_files(folder)
    models = folder / S["MODELS_SUBDIR"]; work = models / "work"
    work.mkdir(parents=True, exist_ok=True)
    train_bytes = sum(os.path.getsize(p) for p in sets["train"])
    say(f"dataset {folder}: {len(sets['train'])} training files ({train_bytes / 1e6:.1f} MB), "
        f"{len(sets['heldout'])} held-out, {len(sets['ood'])} out-of-domain")
    for p in sets["heldout"]:
        say(f"  held out: {Path(p).name}")
    if S["THREADS"]:
        torch.set_num_threads(S["THREADS"])
    dev = device()
    native = native_encoder()
    enc = make_encoder(S["ENCODING"], S["VOCAB"], sets["train"], work, native)
    t0 = time.time()
    tokens, lens = load_tokens(enc, sets["train"], work)
    say(f"{enc.name}: {len(tokens):,} codes for {train_bytes / 1e6:.1f} MB ({time.time() - t0:.0f}s to encode)")
    tag = "ng%d" % S["NGRAM_TOPK"] if S["NGRAM_ORDERS"] else "plain"
    name = f"{enc.name}_{tag}_{S['WIDTH']}w{S['LAYERS']}L_{time.strftime('%Y%m%d_%H%M')}"
    save_path = models / f"{name}.pt"
    result = train_lm.run(enc, tokens, lens, {"heldout": sets["heldout"], "ood": sets["ood"]}, work, S["BUDGET_S"],
                          S["WIDTH"], S["LAYERS"], S["HEADS"], S["CTX"], S["BATCH"], S["LR"], 0, S["SEED"],
                          S["EVAL_EVERY_S"], log=say, save_path=str(save_path), ngram_orders=tuple(S["NGRAM_ORDERS"]),
                          ngram_topk=S["NGRAM_TOPK"], valid_share=S["VALID_SHARE"], patience=S["PATIENCE"],
                          min_delta=S["MIN_DELTA"], device=dev)
    # What chat needs to rebuild the tables exactly: the files, their order and the validation split.
    ck = torch.load(save_path, map_location="cpu", weights_only=False)
    ck["run_model"] = {"dataset": str(folder), "train_files": sets["train"], "valid_share": S["VALID_SHARE"],
                       "settings": {k: (list(v) if isinstance(v, tuple) else v) for k, v in S.items()}}
    torch.save(ck, save_path)
    result["settings"] = ck["run_model"]["settings"]; result["dataset"] = ck["run_model"]["dataset"]
    (models / f"{name}.json").write_text(json.dumps(result, indent=1))
    ev = result["eval"]
    say(f"done. validation {ev.get('validation', {}).get('bits_per_byte')} bits/byte"
        + (f", held-out {ev['heldout_overall']['bits_per_byte']}" if ev.get("heldout_overall") else "")
        + f"; {result['training']['steps']} steps, stopped: {result['training']['stop_reason']}")
    say(f"saved {save_path} and {name}.json")
    return save_path


# ----------------------------------------------------------------------------- chat

def load_model(path: str):
    S = SETTINGS
    ck = torch.load(path, map_location="cpu", weights_only=False)
    info, shape, work = ck["encoding"], ck["model"], Path(path).parent / "work"
    native = native_encoder()
    enc = load_encoder(info, work, native)
    valid = np.fromiter(enc.reverse.keys(), dtype=np.int64, count=len(enc.reverse)) if isinstance(enc, LE.HashCodes) else None
    ngrams = []
    if ck.get("ngram_orders"):
        rm = ck.get("run_model")
        if rm is None:
            raise SystemExit("this model was not trained by run_model.py; use compiler/experiments/chat.py with --data")
        files = [p if Path(p).exists() else str(Path(rm["dataset"]) / Path(p).name) for p in rm["train_files"]]
        say(f"recounting the tables from {len(files)} training files")
        tokens, _ = load_tokens(enc, files, work)
        n_train = int(len(tokens) * (1 - rm["valid_share"]))
        ngrams = [train_lm.NgramTable(tokens[:n_train], o, ck.get("ngram_topk", 16)) for o in ck["ngram_orders"]]
    model = train_lm.LM(enc, shape["width"], shape["layers"], shape["heads"], shape["ctx"], shape["table_rows"], valid,
                        tuple(shape.get("table_orders", [2])), ngrams)
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or any(not k.startswith("table") for k in missing):
        raise SystemExit(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    dev = device()
    model.to(dev).eval()
    return enc, model, dev


@torch.no_grad()
def reply(enc, model, dev, prompt: bytes, n_tokens: int, temperature: float, top_k: int, seed: int):
    gen = torch.Generator().manual_seed(seed)
    p_tokens, _ = enc.encode(prompt)
    seq = [int(t) for t in p_tokens]
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        seq.append(model.sample_next(torch.tensor([seq[-model.ctx:]], device=dev), gen, temperature, top_k))
    dt = time.perf_counter() - t0
    out = enc.decode(seq[len(p_tokens):])
    return out, f"{n_tokens} codes, {len(out)} bytes in {dt:.1f}s ({len(out) / dt:.0f} B/s)"


def is_qa_model(path: str) -> bool:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return "args" in ck and "encoding" not in ck


def chat_qa(path: str) -> None:
    """Chat with a Q&A model: every question is looked up in the exchange memory, the closest
    stored exchange goes into the window, and the copy head can carry its answer into the reply.
    /add teaches it a new fact at once, with no training."""
    import m2_qa
    S = SETTINGS
    if S["THREADS"]:
        torch.set_num_threads(S["THREADS"])
    eng = m2_qa.QAEngine(path, native=native_encoder(), device=device(), log=say)
    print("Ask a question. Commands:\n"
          "  /add question || answer   store a new fact (answerable at once, no training)\n"
          "  /nomem question           answer without memory (the network alone)\n"
          "  /quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line.startswith("/add "):
            q, sep, ans = line[5:].partition("||")
            if not sep or not q.strip() or not ans.strip():
                print("format: /add question || answer"); continue
            i = eng.add(q.encode("utf-8"), ans.encode("utf-8"))
            print(f"stored as memory entry {i:,}; ask it now"); continue
        use_mem = True
        if line.startswith("/nomem "):
            line, use_mem = line[7:], False
        t0 = time.perf_counter()
        out, info = eng.ask(line.encode("utf-8"), S["QA_MAX_CODES"], S["QA_NO_REPEAT"], use_memory=use_mem)
        dt = time.perf_counter() - t0
        print(f"model> {out.decode('utf-8', 'replace')}")
        if S["QA_SHOW_MEMORY"]:
            src = f"{info['memory']} match ({info['similarity']:.2f}): {info['retrieved_question']}" if "retrieved_question" in info else info["memory"]
            calc = f"; executor computed {', '.join(info['computed'])}" if info["computed"] else ""
            print(f"       [memory: {src}{calc}; {info['codes']} codes in {dt:.1f}s]")


def chat(path: str) -> None:
    if is_qa_model(path):
        return chat_qa(path)
    S = SETTINGS
    enc, model, dev = load_model(path)
    pc = model.param_counts()
    say(f"loaded {Path(path).name}: {pc['total']:,} learned parameters, encoder {enc.name}, "
        f"{len(model.ngrams)} count tables, context {model.ctx} codes")
    print("The model continues your text in the style of its training data (it is a text model, not an assistant).\n"
          "Commands: /temp 0.8   /tokens 200   /topk 40   /seed 2   /quit")
    temp, n_tok, top_k, seed = S["TEMPERATURE"], S["REPLY_TOKENS"], S["TOP_K"], S["SEED"]
    while True:
        try:
            line = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line.strip():
            continue
        if line.startswith("/"):
            cmd, _, val = line[1:].partition(" ")
            if cmd in ("quit", "exit", "q"):
                break
            try:
                if cmd == "temp": temp = float(val)
                elif cmd == "tokens": n_tok = int(val)
                elif cmd == "topk": top_k = int(val)
                elif cmd == "seed": seed = int(val)
                else: print("unknown command"); continue
                print(f"temperature {temp}, reply {n_tok} codes, top-k {top_k}, seed {seed}")
            except ValueError:
                print("bad value")
            continue
        out, stats = reply(enc, model, dev, line.encode("utf-8"), n_tok, temp, top_k, seed)
        print(f"model> {out.decode('utf-8', 'replace')}\n       [{stats}]")


# ----------------------------------------------------------------------------- menu

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", metavar="DIR", help="train on this folder (no popup)")
    ap.add_argument("--chat", metavar="MODEL.pt", help="chat with this model (no popup)")
    a = ap.parse_args()
    if a.train:
        train(a.train); return 0
    if a.chat:
        chat(a.chat); return 0
    while True:
        print("\n  1  Train a model on a folder of text files\n  2  Chat with a saved model\n  3  Quit")
        choice = input("choose> ").strip()
        if choice == "1":
            folder = pick_folder("Choose the folder with the .txt files to train on")
            if folder:
                train(folder)
        elif choice == "2":
            path = pick_file("Choose a model (.pt)", [("Model", "*.pt"), ("All files", "*.*")])
            if path:
                chat(path)
        elif choice in ("3", "q", ""):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
