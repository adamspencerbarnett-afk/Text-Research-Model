"""Chat with the best model (C2b, menu option 3), train on a folder of text files, or chat with a saved model.

    python run_model.py              # menu (option 3 = chat with the best model)
    python run_model.py --chat runs/chat22/combo2b_5m_s1.pt   # chat with C2b directly
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
import codecs
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------------- SETTINGS
SETTINGS = dict(
    # --- model shape (research sizes: 128w/4L = 0.8M core; 256w/6L = 4.7M; 384w/8L = 14M; 512w/8L = 25M)
    #     CTX 512 is the research window (512 beat 256 on every Q&A group, October 2026)
    WIDTH=256, LAYERS=6, HEADS=8, CTX=512,
    # --- block: "legacy" (LayerNorm + GELU + position table) or "modern" (RoPE + RMSNorm + SwiGLU: 1-7% lower
    #     text error at equal steps, but at 5M no better on answers, probe 17/32 vs 20-21, half the reply speed)
    BLOCK="legacy",
    # --- bfloat16 training on the GPU (same quality, about twice the speed; ignored on the CPU)
    AMP=True,
    # --- encoding: "compiler" (the deterministic v0 dictionary) or "bpe" (standard tokenizer)
    ENCODING="compiler", VOCAB=8192,
    # --- exact n-gram count tables mixed into the output by the trust head; () for a plain model
    #     (they help up to about 5M parameters and hurt from 20M: use () for the larger shapes)
    NGRAM_ORDERS=(2, 3), NGRAM_TOPK=64,
    # --- training: stop at BUDGET_S seconds of training, or earlier when the validation score
    #     has not improved by MIN_DELTA in PATIENCE evaluations (PATIENCE=0 never)
    BUDGET_S=600, EVAL_EVERY_S=60, PATIENCE=3, MIN_DELTA=0.002,
    BATCH=32, LR=1e-3, SEED=1,
    VALID_SHARE=0.02,       # last share of the training stream kept for validation, never trained on
    HELDOUT_FILES=1,        # files (alphabetically last) scored as held-out when there is no heldout/ folder
    # --- hardware: "auto" = GPU if available, else "cpu", or "cuda:0"
    DEVICE="auto", THREADS=0,   # THREADS 0 = PyTorch's default; menu option 4 switches both while running
    # --- reply speed (inference lab, 14 October 2026: inference_lab/RESULTS.md)
    CORE_DECODE="cache",    # core chat: "cache" (KV cache: the window is processed once, then one code per step;
                            # same codes, 2-28x faster on the CPU) or "recompute" (the whole window for every code: the old path)
    STREAM=True,            # core chat: print the reply as it is generated
    CPU_INT8=False,         # CPU only: int8 matrix kernels (dynamic quantisation) for the core chat. Off: on this
                            # Xeon (AVX2, no int8 dot-product units) they were 1.5-2.5x slower than fp32 and changed replies
    # --- chat
    REPLY_TOKENS=120, TEMPERATURE=0.4, TOP_K=40,
    # --- chat with a Q&A model (memory + copy head; trained by compiler/experiments/m2_qa.py)
    QA_MAX_CODES=120,       # longest reply, in codes (about 3.5 bytes each)
    QA_NO_REPEAT=3,         # block any 3-code phrase from repeating inside a reply (0: off)
    QA_SHOW_MEMORY=True,    # print which stored question the answer was built from
    CHAT_REPLY_CODES=400,   # conversation mode (chat model v2): longest reply in codes (about 3.5 characters each);
                            # /codes N changes it while chatting. Replies stream as they are written.
    CHAT_SHOW_INFO=True,    # conversation mode: print how full the window is and which memory was used
    CHAT_VERIFY=3,          # conversation mode: when memory or a tool supplied evidence, write this many candidate replies and
                            # show the one best supported by the evidence (0: off, the first reply streams as written)
    CHAT_TOOLS=True,        # conversation mode: simple arithmetic and day/month questions are computed, and the result is
                            # placed in the memory slot for the model to read
    QA_DISPLAY="both",      # "both": a MEMORY answer (the model reading what memory returned) and a NETWORK answer (memory
                            # off: general knowledge), each with its scores; "single": the one answer of earlier versions
    QA_ABSTAIN_MATCH=0.15,  # the MEMORY answer says "I don't have knowledge of that" when the best stored question has a word
    QA_ABSTAIN_MEANING=0.55,#   match below this AND a meaning match below QA_ABSTAIN_MEANING (first values, 14 October; to tune)
    QA_CLARIFY_GAP=0.15,    # two stored facts within this of each other, telling apart by one word the question lacks: ask which
    QA_FAST="graph",        # "graph": CUDA-graph answer step (same replies, 3-9x faster on the GPU); True: the eager loop
    QA_MIN_SIM=0.0,         # memory gate: a nearest match below this similarity is ignored and the network
                            # answers alone. 0.0 for the 5M model (gate 0.5 drops the probe 21 -> 19/32);
                            # 0.5 for the 50M models (21 -> 24/32), whose network alone is good enough
    # --- places
    BEST_CHAT_MODEL="runs/chat22/combo2b_5m_s1.pt",   # the best chat model (menu option 3, 20 Oct): C2b = round 5's recipe + multi-part messages (recipes/BEST.json v3)
    CHAT22_SEARCH="v2",     # chat22 models: "v2" = search engine v2 (fact table, fast lane, decide); "v1" = the word + meaning search
    CHAT22_FACTS=None,      # chat22 models: another permanent store (.jsonl / .txt) instead of the one saved with the model
    CHAT22_HOP2=True,       # chat22 models: two-hop search ("Who wrote my favorite song?" -> the stored fact about that song); adopted 20 Oct (recipes/BEST.json)
    CHAT22_SPLIT=True,      # chat22 models: one memory search per question of a message ("Where does Kim live and what does she do?"); adopted with C2b
    CHAT_INFO="short",      # chat22 models: the line under each reply: "off", "short" (where the answer came from, window fill,
                            #   time) or "full" (also the search path, items, codes, first-word wait)
    BEST_MODEL="models/qa_best_5M_dense.pt",   # the best Q&A model in git (menu option 5): probe 24/32; needs sentence-transformers; the 50M models are local in runs/
    MODELS_SUBDIR="models",                    # created inside the dataset folder
    START_DIR=str(Path.home() / "OneDrive" / "Desktop" / "Text Files"),   # where the popups open
    MODEL_DIR="runs",       # where the model popup (menu option 2) opens; relative to this script's folder
)
# ----------------------------------------------------------------------------- end of settings

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "compiler"))
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
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


def pick_file(title: str, patterns, start: str | None = None) -> str | None:
    from tkinter import Tk, filedialog
    root = Tk(); root.withdraw(); root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, initialdir=start or SETTINGS["START_DIR"], filetypes=patterns)
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
    recorded = Path(info["dictionary"])
    # the recorded path may be relative to the project folder (research runs) or the dataset folder may have
    # moved: try it as recorded, from the project folder, beside the model, then in the model's work folder
    for dict_path in (recorded, ROOT / recorded, work.parent / recorded.name, work / recorded.name):
        if dict_path.exists():
            return LE.V0Dict(str(dict_path), native)
    raise SystemExit(f"dictionary {recorded.name} not found (looked as recorded, in {ROOT}, beside the model, and in {work})")


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
                          min_delta=S["MIN_DELTA"], device=dev, amp=S["AMP"] and dev.startswith("cuda"), block=S["BLOCK"])
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
            raise SystemExit("this model was not trained by run_model.py: chat with Q&A models (m2_qa.py) here, or core models through compiler/experiments/core_eval.py")
        files = [p if Path(p).exists() else str(Path(rm["dataset"]) / Path(p).name) for p in rm["train_files"]]
        say(f"recounting the tables from {len(files)} training files")
        tokens, _ = load_tokens(enc, files, work)
        n_train = int(len(tokens) * (1 - rm["valid_share"]))
        ngrams = [train_lm.NgramTable(tokens[:n_train], o, ck.get("ngram_topk", 16)) for o in ck["ngram_orders"]]
    model = train_lm.LM(enc, shape["width"], shape["layers"], shape["heads"], shape["ctx"], shape["table_rows"], valid,
                        tuple(shape.get("table_orders", [2])), ngrams, block=shape.get("block", "legacy"))
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    if unexpected or any(not k.startswith("table") for k in missing):
        raise SystemExit(f"checkpoint does not match the model: missing {missing}, unexpected {unexpected}")
    dev = device()
    if S["THREADS"]:
        torch.set_num_threads(S["THREADS"])
    model.to(dev).eval()
    if S["CPU_INT8"] and dev == "cpu":
        model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        say("int8 matrix kernels (CPU)")
    return enc, model, dev


@torch.no_grad()
def sample_from(model, h, x, gen, temperature: float, top_k: int) -> int:
    """train_lm.LM.sample_next from an already computed last hidden state h (for the KV-cached reply)."""
    def pick(logits):
        logits = logits / temperature
        if top_k:
            kth = torch.topk(logits, top_k).values[..., -1, None]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        return torch.multinomial(F.softmax(logits, -1).cpu(), 1, generator=gen).item()
    if model.coords:
        g = pick(model.head_g(h).masked_fill(~model.valid_group, float("-inf")))
        lm = model.head_m(torch.cat([h, model.emb_g(torch.tensor([g], device=h.device))], -1))
        return g * model.M + pick(lm.masked_fill(~model.valid[g], float("-inf")))
    logits = model.mixed_logits(h, x)
    if model.no_case_flags:
        logits[:, 256:258] = float("-inf")
    return pick(logits)


@torch.no_grad()
def reply(enc, model, dev, prompt: bytes, n_tokens: int, temperature: float, top_k: int, seed: int, stream=None):
    """Continue the prompt. With CORE_DECODE="cache" the window goes through the network once and each new
    code reuses the keys and values of the earlier ones; when the window is full its newest half is processed
    again. ``stream`` is called with each new piece of text as soon as it is generated."""
    gen = torch.Generator().manual_seed(seed)
    p_tokens, _ = enc.encode(prompt)
    seq = [int(t) for t in p_tokens] or [10]
    start = len(seq) if len(p_tokens) else 1
    use_cache = SETTINGS["CORE_DECODE"] == "cache"
    cache, base, done, shown, first = None, 0, 0, b"", None
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        x = torch.tensor([seq[-model.ctx:]], device=dev)
        if use_cache:
            try:
                if cache is None or len(seq) - base > model.ctx:      # first step, or the window is full
                    base = len(seq) - model.ctx // 2 if len(seq) > model.ctx else 0
                    h, cache = model.hidden_cached(torch.tensor([seq[base:]], device=dev), 0, None)
                else:                                                # only the newest code goes through
                    h, cache = model.hidden_cached(torch.tensor([seq[done:]], device=dev), done - base, cache)
                done = len(seq)
            except NotImplementedError:                              # hashed input tables: no cache for this model
                use_cache = False
        if use_cache:
            seq.append(sample_from(model, h[:, -1], x, gen, temperature, top_k))
        else:
            seq.append(model.sample_next(x, gen, temperature, top_k))
        if first is None:
            first = time.perf_counter() - t0
        if stream:
            text = enc.decode(seq[start:])
            if len(text) > len(shown):
                stream(text[len(shown):]); shown = text
    dt = time.perf_counter() - t0
    out = enc.decode(seq[start:])
    return out, (f"{n_tokens} codes, {len(out)} bytes in {dt:.1f}s ({len(out) / dt:.0f} B/s), "
                 f"first code after {1000 * first:.0f} ms, {SETTINGS['CORE_DECODE']} on {dev}")


def is_qa_model(path: str) -> bool:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return "args" in ck and "encoding" not in ck


def _cwords(stop, b: bytes) -> list[bytes]:
    """Content words of a text, lower case, possessive 's dropped."""
    out = []
    for w in re.findall(rb"[A-Za-z0-9][A-Za-z0-9#@.\-']*", b):
        w = w.lower().rstrip(b".")
        w = w[:-2] if w.endswith(b"'s") else w
        if w.decode("latin-1") not in stop and len(w) > 1:
            out.append(w)
    return out


def _matches(eng, q: bytes):
    """The best stored exchanges for q, best first: (combined score, word match, meaning match, index). The combined
    score is the memory's own ranking (word match + weighted meaning match when a sentence key is attached)."""
    r = eng.mem.retrieve(q, 5)
    L = eng.mem.learned
    cos = (L.vectors @ L.embed([q])[0]).float().cpu() if (L is not None and r) else None
    out = []
    for lit, i in r:
        c = float(cos[i]) if cos is not None else 0.0
        s = lit / getattr(L, "lit_std", 1.0) + (L.weight * c / getattr(L, "key_std", 1.0) if L is not None else 0.0)
        out.append((s, lit, c, i))
    return sorted(out, reverse=True)


@torch.no_grad()
def _confidence(eng, q: bytes, ans: bytes, evidence=None) -> float:
    """Average probability the model gave each code of its answer (0 to 1): how sure it was while writing it.
    Small models are often confidently wrong, so this is a rough signal, not a truth meter."""
    import m2_qa
    enc = eng.enc
    nb = enc.encode(m2_qa.render(*evidence))[0][-eng.a.neighbor_codes:] if evidence else np.array([], dtype=np.int64)
    own = np.append(enc.encode(m2_qa.render(q, ans))[0], m2_qa.EOS)
    ql = len(enc.encode(m2_qa.render(q, None))[0])
    full = np.concatenate([nb, eng.sep, own]).astype(np.int64)
    seq = full[-(eng.a.ctx + 1):]
    ev = max(0, len(nb) - (len(full) - len(seq)))
    start = len(seq) - len(own) + ql
    x = torch.tensor(seq[None, :-1], device=eng.dev); y = torch.tensor(seq[None, 1:], device=eng.dev)
    src = torch.zeros_like(x, dtype=torch.bool); src[0, :ev] = True
    nll = eng.model.nll(x, y, src, torch.zeros(1, device=eng.dev))[0, start - 1:]
    return float(torch.exp(-nll.float()).mean()) if len(nll) else 0.0


def _distinct(stop, x: bytes, y: bytes) -> str | None:
    """A word of stored question x that y lacks (a name first), to tell two stored facts apart."""
    yw = set(_cwords(stop, y))
    cand = [w.rstrip(b".?") for w in re.findall(rb"[A-Za-z0-9][A-Za-z0-9#@.\-']*", x) if _cwords(stop, w) and _cwords(stop, w)[0] not in yw]
    if not cand:
        return None
    w = ([c for c in cand if c[:1].isupper()] or cand)[0]
    return (w[:-2] if w.endswith(b"'s") else w).decode("utf-8", "replace")


def _block(label: str, text: str, details: str) -> str:
    import textwrap
    body = textwrap.fill(text or "(empty)", width=92, initial_indent=f"  {label:<8} ", subsequent_indent=" " * 11)
    return body + "\n" + " " * 11 + details


def strict_exact(eng):
    """Exact matches only when the question's content-word key has two or more words or the whole question is the same.
    The research key keeps content words only, so 'How are you today?' and 'Generate a to-do-list for today.' both
    became {today} and counted as the same question (found 14 October). Applied to this chat's engine only."""
    norm = lambda b: re.sub(rb"[^a-z0-9 ]", b"", b.lower()).strip()
    loose = eng.mem.exact_hit

    def exact_hit(q):
        i = loose(q)
        if i is not None and (len(eng.mem.key_of(q)) >= 2 or norm(eng.mem.exchanges[i][0]) == norm(q)):
            return i
        return None
    eng.mem.exact_hit = exact_hit


def qa_turn(eng, line: str, state: dict) -> str:
    """One question in the two-answer chat (QA_DISPLAY = "both"). The MEMORY answer is the model reading what memory
    returned; when memory holds nothing close it says so, and when two stored facts fit equally it asks which one.
    The NETWORK answer is the same model with memory off: its own general knowledge, with its confidence."""
    import m2_qa
    S, stop = SETTINGS, m2_qa.STOP
    t0 = time.perf_counter()
    nomem = line.startswith("/nomem ")
    q = (line[7:] if nomem else line).encode("utf-8")
    ask = lambda text, mem: eng.ask(text, S["QA_MAX_CODES"], S["QA_NO_REPEAT"], use_memory=mem, min_sim=S["QA_MIN_SIM"], fast=S["QA_FAST"])
    out = []
    # a reply to our clarifying question: answer from the option the user named
    if state.get("options") and not nomem:
        low = line.lower()
        chosen = [(w, i) for w, i in state["options"] if w.lower() in low]
        prev_q = state.pop("question"); state.pop("options")
        if len(chosen) == 1:
            w, i = chosen[0]
            stored_q, stored_a = eng.mem.exchanges[i]
            ans, _ = ask(stored_q, True)
            conf = _confidence(eng, stored_q, ans, (stored_q, stored_a))
            out.append(_block("MEMORY", ans.decode("utf-8", "replace"),
                              f"you chose {w} | confidence {conf:.2f} | from: \"{stored_q.decode('utf-8', 'replace')[:70]}\""))
            return "\n".join(out) + f"\n{'-' * 100}"
    if not nomem:
        m = _matches(eng, q)
        exact = eng.mem.exact_hit(q)
        best = m[0] if m else (0.0, 0.0, 0.0, None)
        # a question with fewer than two content words ('How are you today?' -> {today}) gets a word match of 1.0 from any
        # stored question with the same single word: judge it by meaning match alone
        word_ok = len(eng.mem.key_of(q)) >= 2 and best[1] >= S["QA_ABSTAIN_MATCH"]
        known = exact is not None or word_ok or best[2] >= S["QA_ABSTAIN_MEANING"]
        clarify = None
        if known and exact is None and len(m) > 1 and m[1][0] >= m[0][0] - S["QA_CLARIFY_GAP"]:
            q1, q2 = eng.mem.exchanges[m[0][3]][0], eng.mem.exchanges[m[1][3]][0]
            d1, d2 = _distinct(stop, q1, q2), _distinct(stop, q2, q1)
            w1, w2 = set(_cwords(stop, q1)), set(_cwords(stop, q2))
            # two facts about different people or things differ by exactly one word each (Sam's / Eli's birthday);
            # two wordings of the same question ("Why is the sky blue?" / "Explain why the sky appears blue") differ by more
            one_word = len(w1 - w2) == 1 and len(w2 - w1) == 1
            if one_word and d1 and d2 and d1.lower() != d2.lower() and d1.lower() not in line.lower() and d2.lower() not in line.lower():
                clarify = (d1, d2)
                state.update(question=q, options=[(d1, m[0][3]), (d2, m[1][3])])
        scores = f"word match {best[1]:.2f}" + (f" | meaning match {best[2]:.2f}" if eng.mem.learned is not None else "")
        if clarify:
            out.append(_block("MEMORY", f"Do you mean {clarify[0]} or {clarify[1]}?",
                              f"{scores} | two stored facts fit equally: reply with one of them"))
        elif not known:
            out.append(_block("MEMORY", "I don't have knowledge of that in my memory.",
                              f"{scores} | nothing stored is close enough ("
                              + ("one content word: judged by meaning match alone, " if len(eng.mem.key_of(q)) < 2 else
                                 f"word match below {S['QA_ABSTAIN_MATCH']}, ")
                              + f"meaning match below {S['QA_ABSTAIN_MEANING']})"))
        else:
            ans, info = ask(q, True)
            ev = eng.mem.exchanges[exact if exact is not None else best[3]]
            aw = _cwords(stop, ans); evw = set(_cwords(stop, ev[0] + b" " + ev[1]))
            grounded = 100 * sum(w in evw for w in aw) / max(1, len(aw))
            conf = _confidence(eng, q, ans, ev)
            src = info.get("retrieved_question", "")[:70]
            out.append(_block("MEMORY", ans.decode("utf-8", "replace"),
                              f"{'exact match' if exact is not None else scores} | from memory {grounded:.0f}% | confidence {conf:.2f} | from: \"{src}\""))
    if S["QA_DISPLAY"] == "both" or nomem:
        net, _ = ask(q, False)
        conf = _confidence(eng, q, net)
        out.append(_block("NETWORK", net.decode("utf-8", "replace"), f"confidence {conf:.2f} | general knowledge, not from memory"))
    out.append(" " * 11 + f"({time.perf_counter() - t0:.1f}s)\n{'-' * 100}")
    return "\n".join(out)


def chat_conversation(path: str) -> None:
    """Conversation chat for the 5M chat model v2 (compiler/experiments/chat_engine.py): replies stream as they are
    written; the window holds the stored facts memory returns plus as many recent turns as fit; every turn also goes into
    a session memory, so details that scrolled out of the window can still be recalled until you quit."""
    import chat_engine as CE
    S = SETTINGS
    if S["THREADS"]:
        torch.set_num_threads(S["THREADS"])
    eng = CE.ChatEngine(path, native=native_encoder(), device=device(), log=say)
    n_codes = S["CHAT_REPLY_CODES"]
    print(f"Conversation mode: window {eng.a.ctx:,} codes (about {eng.a.ctx * 7 // 2:,} characters). Commands:\n"
          "  /add question || answer   store a fact          /load [file.txt]   store many facts\n"
          "  /forget                   start a new conversation (clears the session memory)\n"
          "  /codes N                  longest reply in codes (now %d; stories need 400-600)\n"
          "  /quit" % n_codes)
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/forget":
            eng.forget(); print("new conversation: history and session memory cleared"); continue
        if line.startswith("/codes "):
            try:
                n_codes = max(16, int(line[7:])); print(f"replies up to {n_codes} codes")
            except ValueError:
                print("format: /codes 400")
            continue
        if line.startswith("/add "):
            q, sep, ans = line[5:].partition("||")
            if not sep or not q.strip() or not ans.strip():
                print("format: /add question || answer"); continue
            eng.add(q.strip().encode("utf-8"), ans.strip().encode("utf-8")); eng.base_n = len(eng.mem.exchanges)
            print("stored; ask it now"); continue
        if line == "/load" or line.startswith("/load "):
            f = line[6:].strip() or pick_file("Choose a text file of User:/Assistant: pairs", [("Text", "*.txt"), ("All files", "*.*")])
            if f:
                pairs = m2_qa_parse(Path(f).read_bytes())
                for q, ans in pairs:
                    eng.add(q, ans)
                eng.base_n = len(eng.mem.exchanges)
                print(f"stored {len(pairs):,} pairs")
            continue
        print("model> ", end="", flush=True)
        dec = codecs.getincrementaldecoder("utf-8")("replace")
        out, info = eng.reply(line.encode("utf-8"), n_codes, stream=lambda b: print(dec.decode(b), end="", flush=True),
                              verify=S["CHAT_VERIFY"], use_tools=S["CHAT_TOOLS"])
        print()
        eng.remember(line.encode("utf-8"), out)
        if S["CHAT_SHOW_INFO"]:
            mem = f"memory: {info['memory'][0][:50]}" if info["memory"] else "memory: none used"
            if info.get("tool"):
                mem = f"tool: {info['tool']} | " + mem
            if info.get("candidates_checked", 1) > 1:
                mem += f" | checked {info['candidates_checked']} replies"
            ses = f", recalled {info['session_recalled']} earlier turn(s)" if info["session_recalled"] else ""
            print(f"       [window {info['window_used']:,}/{info['window']:,} codes | {info['turns_in_window']} of {info['turns_total']} "
                  f"turns in the window{ses} | {mem} | {info['codes']} codes in {info['seconds']:.1f}s, "
                  f"first after {info['first_code_ms']} ms]")


def m2_qa_parse(raw: bytes):
    """User:/Assistant: blocks and '/add question || answer' lines of a text file."""
    import m2_qa
    raw = raw.replace(b"\r\n", b"\n")
    pairs = m2_qa.parse_exchanges(raw)
    for ln in raw.split(b"\n"):
        ln = ln.strip()
        if ln.startswith(b"/add ") and b"||" in ln:
            q, _, ans = ln[5:].partition(b"||")
            if q.strip() and ans.strip():
                pairs.append((q.strip(), ans.strip()))
    return pairs


def chat_qa(path: str) -> None:
    if (torch.load(path, map_location="cpu", weights_only=False).get("args") or {}).get("chat_v2"):
        return chat_conversation(path)
    """Chat with a Q&A model: every question is looked up in the exchange memory, the closest
    stored exchange goes into the window, and the copy head can carry its answer into the reply.
    /add teaches it a new fact at once, with no training."""
    import m2_qa
    S = SETTINGS
    if S["THREADS"]:
        torch.set_num_threads(S["THREADS"])
    eng = m2_qa.QAEngine(path, native=native_encoder(), device=device(), log=say)
    strict_exact(eng)
    print("Ask a question. Commands:\n"
          "  /add question || answer   store a new fact (answerable at once, no training)\n"
          "  /load [file.txt]          store every pair in a text file (popup when no file is given); either\n"
          "                            '/add question || answer' lines, or 'User: ...' / 'Assistant: ...' blocks\n"
          "  /nomem question           answer without memory (the network alone)\n"
          "  /quit")
    state: dict = {}            # a clarifying question waiting for the user's choice
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
        if line == "/load" or line.startswith("/load "):
            f = line[6:].strip() or pick_file("Choose a text file of User:/Assistant: pairs", [("Text", "*.txt"), ("All files", "*.*")])
            if not f:
                continue
            raw = Path(f).read_bytes().replace(b"\r\n", b"\n")
            pairs = m2_qa.parse_exchanges(raw)
            for ln in raw.split(b"\n"):            # also accept '/add question || answer' lines
                ln = ln.strip()
                if ln.startswith(b"/add ") and b"||" in ln:
                    q, _, ans = ln[5:].partition(b"||")
                    if q.strip() and ans.strip():
                        pairs.append((q.strip(), ans.strip()))
            for q, ans in pairs:
                eng.add(q, ans)
            print(f"stored {len(pairs):,} pairs from {Path(f).name}" if pairs else
                  f"no pairs found in {Path(f).name}: expected 'User: ...' then 'Assistant: ...', blank line between pairs"); continue
        if S["QA_DISPLAY"] == "both":
            print(qa_turn(eng, line, state)); continue
        use_mem = True
        if line.startswith("/nomem "):
            line, use_mem = line[7:], False
        t0 = time.perf_counter()
        out, info = eng.ask(line.encode("utf-8"), S["QA_MAX_CODES"], S["QA_NO_REPEAT"], use_memory=use_mem, min_sim=S["QA_MIN_SIM"], fast=S["QA_FAST"])
        dt = time.perf_counter() - t0
        print(f"model> {out.decode('utf-8', 'replace')}")
        if S["QA_SHOW_MEMORY"]:
            src = f"{info['memory']} match ({info['similarity']:.2f}): {info['retrieved_question']}" if "retrieved_question" in info else info["memory"]
            calc = f"; executor computed {', '.join(info['computed'])}" if info["computed"] else ""
            print(f"       [memory: {src}{calc}; {info['codes']} codes in {dt:.1f}s]")


def is_chat22_model(path: str) -> bool:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return (ck.get("config") or {}).get("layout") == "chat22"


def _item_label(item) -> str:
    """'[fact] User: What is X?\nAssistant: ...' -> 'stored fact "What is X?"'; '[turn 12] User: ...' -> 'turn 12 "..."'."""
    t = item.decode("utf-8", "replace") if isinstance(item, bytes) else str(item)
    m = re.match(r"\[(fact|turn (\d+))\]\s*(?:User:\s*)?(.*)", t, re.S)
    if not m:
        return '"' + t.strip().splitlines()[0][:50] + '"'
    kind = "stored fact" if m.group(1) == "fact" else f"turn {m.group(2)}"
    first = m.group(3).strip().splitlines()[0].strip()
    return f'{kind} "{first[:50]}{"…" if len(first) > 50 else ""}"'


def _source_line(info: dict, ctx: int, level: str, line: str = "") -> str:
    items = info.get("items") or []
    tool = None
    if line:
        from chat22 import tools as _T                      # the tool result is the first [fact] item; label it as a tool
        t = _T.run(line.encode("utf-8")); tool = t[1].decode("utf-8", "replace") if t else None
    if tool:
        src = "tool: " + tool[:60]
    elif info.get("verified", 0) > 1 and items:
        src = "memory: " + _item_label(items[0]) + (f" (+{len(items) - 1} more)" if len(items) > 1 else "") + f", {info['verified']} tries"
    elif items:
        src = "memory: " + _item_label(items[0]) + (f" (+{len(items) - 1} more)" if len(items) > 1 else "")
    else:
        src = "memory: nothing found, the model answered on its own"
    fill = 100 * info.get("window_codes", 0) // max(1, info.get("window", ctx))
    parts = [src, f"window {fill}%", f"{info.get('seconds', 0):.1f} s"]
    if level == "full":
        found = (info.get("found") or {}).get("path", "-")
        parts += [f"search {found}", f"{info.get('turns_in_window', 0)}/{info.get('turns_total', 0)} turns in window",
                  f"{info.get('codes', 0)} codes", f"first word {info.get('first_code_ms', 0)} ms"]
    return "     " + "  ·  ".join(parts)


class _Stream:
    """Prints a streamed reply cleanly: no leading space, no trailing blank lines, whitespace held until text follows."""

    def __init__(self):
        self.dec = codecs.getincrementaldecoder("utf-8")("replace"); self.started = False; self.held = ""

    def __call__(self, b: bytes) -> None:
        t = self.dec.decode(b)
        if not self.started:
            t = t.lstrip()
            if not t:
                return
            self.started = True
        t = self.held + t
        stripped = t.rstrip()
        self.held = t[len(stripped):]
        if stripped:
            print(stripped, end="", flush=True)


def chat_chat22(path: str) -> None:
    """Chat with a chat22 model (the clean package, 16 October on): replies stream; the window holds the recent turns and
    a short memory block (stored facts and earlier turns the search chose); every turn goes into the conversation memory,
    so anything said earlier can be recalled at any length. Search engine v2 by default (SETTINGS CHAT22_SEARCH)."""
    sys.path.insert(0, str(ROOT))
    from chat22.engine import Engine
    S = SETTINGS
    if S["THREADS"]:
        torch.set_num_threads(S["THREADS"])
    dev = device()
    if dev == "auto":
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    eng = Engine(path, dev, log=say, search=S["CHAT22_SEARCH"], facts_path=S["CHAT22_FACTS"], hop2=S["CHAT22_HOP2"], split=S["CHAT22_SPLIT"])
    n_codes = S["CHAT_REPLY_CODES"]; use_mem = True; level = S["CHAT_INFO"]
    print(f"\nChat with {Path(path).stem} (window {eng.ctx:,} codes, search {S['CHAT22_SEARCH']}, {len(eng.memory.facts.items):,} stored facts)\n"
          "  /add question || answer   store a permanent fact        /load [file.txt]   store many (User:/Assistant: pairs)\n"
          "  /forget                   start a new conversation      /memory off|on     answer without / with memory\n"
          "  /info off|short|full      the line under each reply     /codes N           reply length (stories: 500)\n"
          "  /quit\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); break
        if not line:
            continue
        if line in ("/quit", "/exit", "/q"):
            break
        if line == "/forget":
            eng.forget(); print("     new conversation: history and conversation memory cleared"); continue
        if line.startswith("/memory"):
            use_mem = "off" not in line; print(f"     memory {'on' if use_mem else 'off'}"); continue
        if line.startswith("/info"):
            level = next((w for w in ("off", "short", "full") if w in line), level); print(f"     info {level}"); continue
        if line.startswith("/codes "):
            try:
                n_codes = max(16, int(line[7:])); print(f"     replies up to {n_codes} codes")
            except ValueError:
                print("     format: /codes 400")
            continue
        if line.startswith("/add "):
            q, sep, ans = line[5:].partition("||")
            if not sep or not q.strip() or not ans.strip():
                print("     format: /add question || answer"); continue
            eng.add_fact(q.strip().encode("utf-8"), ans.strip().encode("utf-8")); print("     stored; ask it now"); continue
        if line == "/load" or line.startswith("/load "):
            f = line[6:].strip() or pick_file("Choose a text file of User:/Assistant: pairs", [("Text", "*.txt"), ("All files", "*.*")])
            if f:
                pairs = m2_qa_parse(Path(f).read_bytes())
                for q, ans in pairs:
                    eng.add_fact(q, ans)
                print(f"     stored {len(pairs):,} pairs")
            continue
        print("bot> ", end="", flush=True)
        out, info = eng.reply(line.encode("utf-8"), n_codes, use_memory=use_mem, use_tools=S["CHAT_TOOLS"], stream=_Stream())
        print()
        eng.remember(line.encode("utf-8"), out)
        if level != "off":
            print(_source_line(info, eng.ctx, level, line))


def chat(path: str) -> None:
    if is_chat22_model(path):
        return chat_chat22(path)
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
        if S["STREAM"]:
            print("model> ", end="", flush=True)
            dec = codecs.getincrementaldecoder("utf-8")("replace")
            out, stats = reply(enc, model, dev, line.encode("utf-8"), n_tok, temp, top_k, seed,
                               stream=lambda b: print(dec.decode(b), end="", flush=True))
            print(f"\n       [{stats}]")
        else:
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
        best = ROOT / SETTINGS["BEST_CHAT_MODEL"]; best_qa = ROOT / SETTINGS["BEST_MODEL"]
        print("\n  1  Train a model on a folder of text files\n  2  Chat with a saved model (popup)\n"
              f"  3  Chat with the best chat model ({SETTINGS['BEST_CHAT_MODEL']})\n"
              f"  4  Device and threads (now {SETTINGS['DEVICE']}, threads {SETTINGS['THREADS'] or 'default'})\n"
              f"  5  Chat with the best Q&A model ({SETTINGS['BEST_MODEL']})\n  6  Quit")
        choice = input("choose> ").strip()
        if choice == "1":
            folder = pick_folder("Choose the folder with the .txt files to train on")
            if folder:
                train(folder)
        elif choice == "2":
            path = pick_file("Choose a model (.pt)", [("Model", "*.pt"), ("All files", "*.*")], str(ROOT / SETTINGS["MODEL_DIR"]))
            if path:
                chat(path)
        elif choice == "3":
            if best.exists():
                chat(str(best))
            else:
                print(f"not found: {best}")
        elif choice == "4":
            gpu = torch.cuda.is_available()
            d = input(f"device: auto, cpu{', cuda' if gpu else ''} [{SETTINGS['DEVICE']}]> ").strip().lower()
            if d in ("auto", "cpu") or (gpu and d.startswith("cuda")):
                SETTINGS["DEVICE"] = d
            t = input(f"CPU threads, 0 = default (this machine has {os.cpu_count()}) [{SETTINGS['THREADS']}]> ").strip()
            if t.isdigit():
                SETTINGS["THREADS"] = int(t)
            print(f"device {SETTINGS['DEVICE']}, threads {SETTINGS['THREADS'] or 'default'}: used by the next chat or training")
        elif choice == "5":
            if best_qa.exists():
                chat(str(best_qa))
            else:
                print(f"not found: {best_qa}")
        elif choice in ("6", "q", ""):
            return 0


if __name__ == "__main__":
    raise SystemExit(main())
