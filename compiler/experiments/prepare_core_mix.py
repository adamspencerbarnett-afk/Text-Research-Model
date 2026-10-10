"""Build the core-training corpora: clean modern text, conversation, evidence windows, worked
math and question rewordings, deduplicated and cut into size-matched mixes.

    python -I compiler/experiments/prepare_core_mix.py wiki   ENWIK9.txt            data_core/src/wiki.txt
    python -I compiler/experiments/prepare_core_mix.py books  data_v2/train         data_core/src/books.txt
    python -I compiler/experiments/prepare_core_mix.py oasst  oasst_ready.trees.jsonl.gz data_core/src/oasst.txt
    python -I compiler/experiments/prepare_core_mix.py ultrachat ULTRACHAT_DIR      data_core/src/ultrachat.txt
    python -I compiler/experiments/prepare_core_mix.py fineweb FINEWEB.parquet      data_core/src/fineweb.txt
    python -I compiler/experiments/prepare_core_mix.py metamath MetaMathQA-395K.json data_core/src/math.txt
    python -I compiler/experiments/prepare_core_mix.py qqp    QQP_DIR               data_core/src/qqp.txt
    python -I compiler/experiments/prepare_core_mix.py evidence data_core/src/wiki.txt,data_core/src/fineweb.txt data_core/src/evidence.txt
    python -I compiler/experiments/prepare_core_mix.py mix    data_core/src  data_core/mix_D3_5M  --mb 400 --recipe D3

Every source file holds documents separated by a line containing only the form feed character
(\\f), so documents never run into each other and can be counted, deduplicated and split.
Held-out documents (1% of each source, chosen by a hash of the document, never by position) go
to the mix's heldout/ folder, one file per source, so fluency can be measured per kind of text.
See docs/CORE_TRAINING_PLAN.md for why each part exists.
"""
from __future__ import annotations

import argparse
import collections
import glob
import gzip
import hashlib
import html
import json
import re
import sys
from pathlib import Path

DOC = "\n\f\n"          # document separator inside source files

# ----------------------------------------------------------------------------- Wikipedia

NS_SKIP = re.compile(r"^(Wikipedia|Category|Template|Image|File|User|Talk|Help|Portal|MediaWiki|Special|[A-Za-z ]+ talk):")
LINK_DROP = re.compile(r"\[\[(?:Image|File|Category|[a-z]{2,3}(?:-[a-z]+)?):[^\[\]]*(?:\[\[[^\]]*\]\][^\[\]]*)*\]\]")


def strip_nested(text: str, open_s: str, close_s: str) -> str:
    """Remove every (possibly nested) open_s ... close_s block."""
    out, depth, i, n = [], 0, 0, len(text)
    lo, lc = len(open_s), len(close_s)
    while i < n:
        if text.startswith(open_s, i):
            depth += 1; i += lo; continue
        if depth and text.startswith(close_s, i):
            depth -= 1; i += lc; continue
        if not depth:
            out.append(text[i])
        i += 1
    return "".join(out)


def clean_wiki_article(raw: str) -> str:
    t = html.unescape(html.unescape(raw))
    t = re.sub(r"<!--.*?-->", "", t, flags=re.S)
    t = re.sub(r"<ref[^>/]*/>", "", t)
    t = re.sub(r"<ref[^>]*>.*?</ref>", "", t, flags=re.S)
    t = re.sub(r"<(math|gallery|timeline|table|div|span|small|sup|sub|center|font|br|references|nowiki)[^>]*>", " ", t, flags=re.I)
    t = re.sub(r"</?[a-zA-Z][^>]{0,200}>", "", t)
    t = strip_nested(t, "{{", "}}")
    t = strip_nested(t, "{|", "|}")
    t = LINK_DROP.sub("", t)
    t = re.sub(r"\[\[([^\[\]|]*)\|([^\[\]]*)\]\]", r"\2", t)
    t = re.sub(r"\[\[([^\[\]]*)\]\]", r"\1", t)
    t = re.sub(r"\[https?://\S+ ([^\]]*)\]", r"\1", t)
    t = re.sub(r"\[https?://\S+\]", "", t)
    t = re.sub(r"'{2,}", "", t)
    paras, cur = [], []
    for line in t.split("\n"):
        s = line.strip()
        if not s:
            if cur: paras.append(" ".join(cur)); cur = []
            continue
        m = re.match(r"^(=+)\s*(.*?)\s*\1$", s)
        if m:                                           # a heading becomes its own short paragraph
            if cur: paras.append(" ".join(cur)); cur = []
            if m.group(2).lower() not in ("references", "external links", "see also", "notes", "further reading", "bibliography", "sources"):
                paras.append(m.group(2))
            else:
                break                                   # the rest of the article is lists of links
            continue
        if s.startswith("|") or s.startswith("!"):
            continue
        if re.match(r"^[*#:;]", s):                     # a list item is its own line, not merged into prose
            if cur: paras.append(" ".join(cur)); cur = []
            paras.append(re.sub(r"^[*#:;]+\s*", "", s))
            continue
        cur.append(s)
    if cur: paras.append(" ".join(cur))
    paras = [re.sub(r"\s+", " ", p).strip() for p in paras]
    paras = [p for p in paras if len(p) > 1 and not re.fullmatch(r"[\W\d_]+", p)]
    return "\n\n".join(paras)


def cmd_wiki(src: str, out: str, min_chars: int = 500) -> None:
    raw = Path(src).read_text(encoding="utf-8", errors="replace")
    docs, skipped = [], collections.Counter()
    for m in re.finditer(r"<page>\s*<title>(.*?)</title>.*?<text[^>]*>(.*?)</text>", raw, re.S):
        title, body = html.unescape(m.group(1)), m.group(2)
        if NS_SKIP.match(title): skipped["namespace"] += 1; continue
        if body.lstrip()[:9].upper() == "#REDIRECT": skipped["redirect"] += 1; continue
        text = clean_wiki_article(body)
        if len(text) < min_chars: skipped["short"] += 1; continue
        docs.append(title + "\n\n" + text)
    write_docs(out, docs)
    print(f"wiki: {len(docs):,} articles kept; skipped {dict(skipped)}")


# ----------------------------------------------------------------------------- books

def cmd_books(src_dir: str, out: str) -> None:
    """Join hard-wrapped lines inside paragraphs; paragraphs stay separated by blank lines."""
    docs = []
    for p in sorted(glob.glob(f"{src_dir}/*.txt")):
        t = Path(p).read_text(encoding="utf-8", errors="replace").replace("\r\n", "\n")
        paras = [re.sub(r"\s*\n\s*", " ", q).strip() for q in re.split(r"\n\s*\n", t)]
        docs.append("\n\n".join(q for q in paras if q))
    write_docs(out, docs)
    print(f"books: {len(docs)} unwrapped")


# ----------------------------------------------------------------------------- conversation

def render_turns(turns: list[tuple[str, str]]) -> str:
    """A conversation in the chat format the model answers in: 'User: ...' / 'Assistant: ...'."""
    return "\n".join(("User: " if r == "user" else "Assistant: ") + re.sub(r"\n{3,}", "\n\n", t.strip()) for r, t in turns)


def cmd_oasst(src: str, out: str) -> None:
    """OpenAssistant oasst1 trees, English only; at each assistant turn the top-ranked reply."""
    docs = []
    with gzip.open(src, "rt", encoding="utf-8") as f:
        for line in f:
            tree = json.loads(line)
            node = tree["prompt"]
            if node.get("lang") != "en":
                continue
            turns = []
            while node:
                turns.append(("user" if node["role"] == "prompter" else "assistant", node["text"]))
                kids = [k for k in node.get("replies", []) if k.get("lang") == "en" and not k.get("deleted")]
                if not kids:
                    break
                node = min(kids, key=lambda k: k.get("rank") if k.get("rank") is not None else 99)
            if len(turns) >= 2 and turns[-1][0] == "user":
                turns = turns[:-1]
            if len(turns) >= 2:
                docs.append(render_turns(turns))
    write_docs(out, docs)
    print(f"oasst: {len(docs):,} English conversations")


def cmd_ultrachat(src_dir: str, out: str) -> None:
    import pyarrow.parquet as pq
    docs = []
    for p in sorted(glob.glob(f"{src_dir}/*.parquet")):
        for batch in pq.ParquetFile(p).iter_batches(columns=["messages"], batch_size=4096):
            for msgs in batch.column(0).to_pylist():
                turns = [(m["role"], m["content"]) for m in msgs if m["role"] in ("user", "assistant") and m["content"].strip()]
                if len(turns) >= 2:
                    docs.append(render_turns(turns))
    write_docs(out, docs)
    print(f"ultrachat: {len(docs):,} conversations")


# ----------------------------------------------------------------------------- web, math, questions

def cmd_fineweb(src: str, out: str, max_docs: int = 0) -> None:
    import pyarrow.parquet as pq
    docs = []
    for batch in pq.ParquetFile(src).iter_batches(columns=["text"], batch_size=8192):
        for t in batch.column(0).to_pylist():
            t = re.sub(r"[ \t]+", " ", t.replace("\r\n", "\n"))
            t = re.sub(r"\n{3,}", "\n\n", t).strip()
            if len(t) >= 300:
                docs.append(t)
        if max_docs and len(docs) >= max_docs:
            break
    write_docs(out, docs[:max_docs] if max_docs else docs)
    print(f"fineweb-edu: {len(docs):,} documents")


def cmd_metamath(src: str, out: str) -> None:
    """MetaMathQA: question + step-by-step answer, final line 'Answer: N' (the calculator's format)."""
    rows = json.loads(Path(src).read_text(encoding="utf-8"))
    docs = []
    for r in rows:
        resp = r["response"].strip()
        m = re.search(r"The answer is:?\s*(.+)$", resp)
        body = resp[:m.start()].strip() if m else resp
        final = m.group(1).strip() if m else None
        if not final:
            continue
        docs.append(render_turns([("user", r["query"]), ("assistant", body + "\nAnswer: " + final)]))
    write_docs(out, docs)
    print(f"metamath: {len(docs):,} worked problems")


def cmd_qqp(src_dir: str, out: str) -> None:
    """Quora question pairs marked as duplicates (research use): 'Question: a / In other words: b'.
    The validation split is kept apart as the reword-retrieval test (data_core/eval/qqp_val.jsonl)."""
    import pyarrow.parquet as pq
    t = pq.read_table(f"{src_dir}/train.parquet").to_pydict()
    docs = [f"Question: {a.strip()}\nIn other words: {b.strip()}" for a, b, l in zip(t["question1"], t["question2"], t["label"]) if l == 1]
    write_docs(out, docs)
    v = pq.read_table(f"{src_dir}/validation.parquet").to_pydict()
    ev = Path(out).parent.parent / "eval"; ev.mkdir(parents=True, exist_ok=True)
    with open(ev / "qqp_val.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for a, b, l in zip(v["question1"], v["question2"], v["label"]):
            if l == 1:
                f.write(json.dumps({"q1": a.strip(), "q2": b.strip()}) + "\n")
    print(f"qqp: {len(docs):,} duplicate pairs for training; validation pairs written to {ev / 'qqp_val.jsonl'}")


# ----------------------------------------------------------------------------- evidence windows

WORD = re.compile(r"[A-Za-z]{3,}")
COMMON = set("the and for that with this from are was were have has had not but they their which been more also into than its other some such only when what where who will would there these can about one all may any most many".split())


def para_key(p: str) -> frozenset:
    return frozenset(w for w in (x.lower() for x in WORD.findall(p)) if w not in COMMON)


def cmd_evidence(srcs: str, out: str, max_pairs: int = 400_000, seed: int = 1) -> None:
    """Pairs of related paragraphs from different documents: 'evidence paragraph' then 'target
    paragraph'. Relation = idf-weighted shared content words (the exchange memory's measure),
    at least 0.15 and at most 0.8 (near-copies are skipped). Each paragraph is used as evidence
    at most once. Held-out pairs (for the evidence-gain test) go to data_core/eval/evidence_val.jsonl."""
    import math
    import random
    rng = random.Random(seed)
    paras: list[tuple[int, str]] = []
    for src in srcs.split(","):
        for d_i, doc in enumerate(read_docs(src)):
            for p in doc.split("\n\n"):
                if 200 <= len(p) <= 450:                 # a pair must fit one 256-code window (about 900 bytes)
                    paras.append((hash((src, d_i)), p))
    rng.shuffle(paras)
    paras = paras[: max_pairs * 4]
    keys = [para_key(p) for _, p in paras]
    df = collections.Counter(w for k in keys for w in k)
    n = len(paras)
    idf = {w: math.log((1 + n) / (1 + c)) for w, c in df.items()}
    index: dict[str, list[int]] = collections.defaultdict(list)
    for i, k in enumerate(keys):
        for w in k:
            if df[w] <= 0.002 * n:                      # only distinctive words index (keeps it fast)
                index[w].append(i)
    used, pairs = set(), []
    for i in range(n):
        if len(pairs) >= max_pairs:
            break
        scores = collections.Counter()
        for w in keys[i]:
            for j in index.get(w, ()):
                if j != i and paras[j][0] != paras[i][0] and j not in used:
                    scores[j] += idf[w]
        if not scores:
            continue
        j, s = scores.most_common(1)[0]
        union = sum(idf[w] for w in keys[i] | keys[j])
        sim = s / max(union, 1e-9)
        if 0.15 <= sim <= 0.8:
            used.add(j)
            pairs.append((paras[j][1], paras[i][1], round(sim, 3)))
    cut = int(len(pairs) * 0.99)
    write_docs(out, [f"{e}\n\n{t}" for e, t, _ in pairs[:cut]])
    ev = Path(out).parent.parent / "eval"; ev.mkdir(parents=True, exist_ok=True)
    with open(ev / "evidence_val.jsonl", "w", encoding="utf-8", newline="\n") as f:
        for e, t, s in pairs[cut:]:
            f.write(json.dumps({"evidence": e, "target": t, "sim": s}) + "\n")
    print(f"evidence: {cut:,} training pairs, {len(pairs) - cut:,} held out; median similarity "
          f"{sorted(s for _, _, s in pairs)[len(pairs) // 2] if pairs else 0}")


# ----------------------------------------------------------------------------- mixes

RECIPES = {   # shares of the mix by bytes; see docs/CORE_TRAINING_PLAN.md
    "D1": {"wiki": 0.50, "fineweb": 0.35, "books": 0.15},
    "D2": {"wiki": 0.40, "fineweb": 0.30, "books": 0.10, "oasst": 0.05, "ultrachat": 0.15},
    "D3": {"wiki": 0.30, "fineweb": 0.22, "books": 0.08, "oasst": 0.04, "ultrachat": 0.08, "evidence": 0.18, "math": 0.05, "qqp": 0.05},
}


def norm_hash(p: str) -> bytes:
    return hashlib.md5(re.sub(r"\W+", " ", p.lower()).strip().encode()).digest()


def cmd_mix(src_dir: str, out: str, mb: float, recipe: str) -> None:
    shares = RECIPES[recipe]
    outp = Path(out)
    for s in ("train", "heldout", "ood"):
        (outp / s).mkdir(parents=True, exist_ok=True)
    seen: set = set()
    manifest = {"recipe": recipe, "target_mb": mb, "sources": {}}
    for name, share in shares.items():
        docs = read_docs(f"{src_dir}/{name}.txt")
        budget = int(mb * 1e6 * share)
        train, held, size, dup = [], [], 0, 0
        for d in docs:
            h = int.from_bytes(hashlib.sha256(d.encode()).digest()[:4], "little")
            if h % 100 == 0:                              # 1% held out by document hash
                if sum(len(x) for x in held) < 2_000_000:
                    held.append(d)
                continue
            paras = d.split("\n\n")
            keep = []
            for p in paras:
                k = norm_hash(p)
                if len(p) > 60 and k in seen:
                    dup += 1; continue
                seen.add(k); keep.append(p)
            if not keep:
                continue
            d2 = "\n\n".join(keep)
            if size + len(d2) > budget:
                break
            train.append(d2); size += len(d2)
        (outp / "train" / f"{name}.txt").write_bytes(("\n\n".join(train) + "\n").encode("utf-8"))
        (outp / "heldout" / f"{name}.txt").write_bytes(("\n\n".join(held) + "\n").encode("utf-8"))
        manifest["sources"][name] = {"share": share, "train_bytes": size, "train_docs": len(train), "heldout_bytes": sum(len(x) for x in held),
                                     "paragraphs_dropped_as_duplicates": dup, "budget_bytes": budget, "budget_met": size >= 0.95 * budget}
        print(f"  {name:10s} {size/1e6:7.1f} MB of {budget/1e6:7.1f} MB  ({len(train):,} docs; {dup:,} duplicate paragraphs dropped)")
    (outp / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"mix {recipe} -> {out}: {sum(v['train_bytes'] for v in manifest['sources'].values())/1e6:.1f} MB")


# ----------------------------------------------------------------------------- io

def write_docs(out: str, docs: list[str]) -> None:
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_bytes(DOC.join(d.replace("\f", " ") for d in docs).encode("utf-8"))


def read_docs(path: str) -> list[str]:
    return [d for d in Path(path).read_text(encoding="utf-8").split(DOC) if d.strip()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd"); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--mb", type=float, default=400); ap.add_argument("--recipe", default="D3")
    ap.add_argument("--max-docs", type=int, default=0); ap.add_argument("--max-pairs", type=int, default=400_000)
    a = ap.parse_args()
    if a.cmd == "wiki": cmd_wiki(a.src, a.out)
    elif a.cmd == "books": cmd_books(a.src, a.out)
    elif a.cmd == "oasst": cmd_oasst(a.src, a.out)
    elif a.cmd == "ultrachat": cmd_ultrachat(a.src, a.out)
    elif a.cmd == "fineweb": cmd_fineweb(a.src, a.out, a.max_docs)
    elif a.cmd == "metamath": cmd_metamath(a.src, a.out)
    elif a.cmd == "qqp": cmd_qqp(a.src, a.out)
    elif a.cmd == "evidence": cmd_evidence(a.src, a.out, a.max_pairs)
    elif a.cmd == "mix": cmd_mix(a.src, a.out, a.mb, a.recipe)
    else:
        raise SystemExit(f"unknown command {a.cmd}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
