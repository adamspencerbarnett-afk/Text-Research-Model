"""Speed check for search engine v2's parts, on this machine (CPU, and a short GPU check): what each part would cost per
question, against today's 9-10 ms (word key) and 34-40 ms (word + MiniLM) on the CPU."""
import json, sys, time
from pathlib import Path
import numpy as np, torch
ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "compiler" / "experiments", ROOT / "compiler", ROOT): sys.path.insert(0, str(p))
torch.set_num_threads(6)
from chat22 import encoder as E
from chat22.memory import key_of
from chat22.blocks import Block, RMSNorm
res = {}
pool = [json.loads(l) for l in open(ROOT / "data_chat22/facts_pool.jsonl", encoding="utf-8")]
# 1. vectorised word search: a sparse (facts x words) idf matrix; one sparse product per question
import scipy.sparse as sp
vocab, rows, cols = {}, [], []
keys = [key_of(r["key"].encode("utf-8", "replace")) for r in pool]
for i, k in enumerate(keys):
    for w in k:
        rows.append(i); cols.append(vocab.setdefault(w, len(vocab)))
df = np.bincount(cols, minlength=len(vocab)); idf = np.log((1 + len(pool)) / (1 + df))
M = sp.csr_matrix((idf[cols], (rows, cols)), shape=(len(pool), len(vocab)))
qs = [b"Where does my sister live?", b"What is the wifi password at the library?", b"Which school is my grandmother in?"] * 20
t0 = time.perf_counter()
for q in qs:
    qv = np.zeros(len(vocab)); 
    for w in key_of(q):
        if w in vocab: qv[vocab[w]] = 1.0
    s = M @ qv; top = np.argpartition(-s, 50)[:50]
res["word search, vectorised (137,944 facts), ms"] = round(1000 * (time.perf_counter() - t0) / len(qs), 2)
# 2. fact table: (who, what) -> newest value; a Python dict
table = {(f"person{i}", a): (f"value{i}", i) for i in range(20000) for a in ("city", "job")}
t0 = time.perf_counter()
for i in range(10000):
    table.get((f"person{i % 20000}", "city"))
res["fact table lookup, ms"] = round(1000 * (time.perf_counter() - t0) / 10000, 4)
# 3. the judge: our core's blocks scoring (question + candidate) pairs in one batch, full 6 layers and 2 layers
core = torch.load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
sh = core["model"]
for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
    for layers in (6, 2):
        emb = torch.nn.Embedding(8192, sh["width"]).to(dev)
        blocks = torch.nn.ModuleList(Block(sh["width"], sh["heads"]) for _ in range(layers)).to(dev)
        head = torch.nn.Linear(sh["width"], 1).to(dev)
        for n in (40,):
            x = torch.randint(0, 8192, (n, 96), device=dev)
            with torch.no_grad():
                for _ in range(2):
                    h = emb(x)
                    for b in blocks: h = b(h)
                    head(h[:, -1])
                if dev == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(10):
                    h = emb(x)
                    for b in blocks: h = b(h)
                    head(h[:, -1])
                if dev == "cuda": torch.cuda.synchronize()
            res[f"judge {layers} layers, {n} candidates x 96 codes, {dev}, ms"] = round(1000 * (time.perf_counter() - t0) / 10, 1)
# 4. the meaning key on the GPU: MiniLM query vector + one matrix product over the pool
import dense_keys
V = torch.from_numpy(np.load(ROOT / "data_chat22/facts_pool_minilm.npy"))
for dev in (["cpu"] + (["cuda"] if torch.cuda.is_available() else [])):
    dk = dense_keys.DenseKeys(dev, 2.0); Vd = V.to(dev)
    dk.embed([b"warm up"]); t0 = time.perf_counter()
    for q in qs:
        v = dk.embed([q])[0]; s = Vd @ v; torch.topk(s, 50)
    if dev == "cuda": torch.cuda.synchronize()
    res[f"meaning key query + 137,944-fact product, {dev}, ms"] = round(1000 * (time.perf_counter() - t0) / len(qs), 1)
(ROOT / "research_side/engine_v2_speed.json").write_text(json.dumps(res, indent=1))
for k, v in res.items(): print(f"{v:>9}  {k}")
