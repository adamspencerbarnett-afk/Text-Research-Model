"""data_real/: a test folder of real, non-fiction datasets (Adam, 17 Oct 2026). Nothing here replaces the training data in
use; it is built beside it so real data can be tested against the current mix. Only datasets already on disk (no downloads
until the 100M). Where text is chosen, documents whose spelling is mostly British are left out (modern American English).

  qa/         real Q&A in place of Alpaca (machine-written, partly invented): Dolly (human-written by Databricks staff;
              creative writing and brainstorming left out), SQuAD v1.1 with its Wikipedia paragraph, ARC (Easy + Challenge),
              OpenBookQA (answer + its supporting science fact) and CommonsenseQA written as answers with their options
  knowledge/  the permanent fact store: Wikipedia (knowledge_wiki), OpenBookQA's 1,326 core science facts, clean sentences
              from the ARC science corpus
  text/       plain text for rehearsal: Wikipedia + FineWeb-Edu (in place of the 1800s Gutenberg novels) + the related-
              paragraph evidence pairs + conversations, whole documents, the old sample's sizes
  long/       long non-fiction documents (Wikipedia + FineWeb-Edu, no novels) for long windows and document reading;
              narrative non-fiction (history and biography articles) for the continuation half of the story kind
  look/       real look-alike facts for training (in place of facts_pool's invented skill facts and web junk)
  convo/      the de-duplicated dialogues (copied), plus the human-written OpenAssistant threads on their own
  skills/     the conversation-memory skills (copied unchanged: invented on purpose, conversation memory only)
  eval/       held-out sets for every new source

    python -m chat22.data.build_real [--only qa,knowledge,text,long,look,convo,skills] [--seed 1]
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UP = ROOT / "data_uploads"
OUT = ROOT / "data_real"
P100 = ROOT / "data_core" / "mix_P100" / "train"

UK = ["colour", "favour", "flavour", "honour", "labour", "neighbour", "behaviour", "humour", "harbour", "centre", "theatre",
      "metre", "litre", "fibre", "organise", "organisation", "realise", "recognise", "apologise", "analyse", "programme",
      "defence", "offence", "travelled", "travelling", "cancelled", "jewellery", "grey", "tyre", "aluminium", "catalogue",
      "cheque", "kerb", "mould", "plough", "practise", "licence", "enrol", "whilst", "amongst"]
US = ["color", "favor", "flavor", "honor", "labor", "neighbor", "behavior", "humor", "harbor", "center", "theater",
      "liter", "fiber", "organize", "organization", "realize", "recognize", "apologize", "analyze", "defense",
      "offense", "traveled", "traveling", "canceled", "jewelry", "gray", "aluminum", "catalog", "mold", "plow", "enroll"]
UK_RX = re.compile(r"\b(" + "|".join(UK) + r")s?\b", re.I)
US_RX = re.compile(r"\b(" + "|".join(US) + r")s?\b", re.I)


def american(text: str) -> bool:
    """Keep unless British spellings outnumber American ones (and there are at least two)."""
    uk = len(UK_RX.findall(text))
    return uk < 2 or uk <= len(US_RX.findall(text))


BLANK_RX = re.compile(r"\n\s*\n")


def ex(q: str, a: str) -> str:
    """One exchange; inner blank lines are folded (a blank line separates exchanges)."""
    q, a = BLANK_RX.sub("\n", q.strip()), BLANK_RX.sub("\n", a.strip())
    return "User: " + q + "\nAssistant: " + a + "\n\n"


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


# ------------------------------------------------------------------------------------------------------------- qa
def build_qa(rng: random.Random) -> dict:
    train, held, rep = [], [], {}
    # Dolly: human-written; creative writing and brainstorming left out (not facts)
    rows = [json.loads(l) for l in open(UP / "dolly" / "databricks-dolly-15k.jsonl", encoding="utf-8")]
    keep = [r for r in rows if r["category"] not in ("creative_writing", "brainstorming") and r["response"].strip()
            and american(r["instruction"] + " " + r["context"] + " " + r["response"])]
    rng.shuffle(keep)
    cut = len(keep) // 20
    for i, r in enumerate(keep):
        q = r["instruction"].strip() + ("\nContext: " + r["context"].strip() if r["context"].strip() else "")
        (held if i < cut else train).append(ex(q, r["response"]))
    rep["dolly"] = {"kept": len(keep), "of": len(rows), "held_out": cut}
    # SQuAD: the prepared passage + question -> answer span (train / held-out = the dev split)
    for split, dst in (("train", train), ("heldout", held)):
        t = (ROOT / "data_qa_squad" / split / f"qa_{split}.txt").read_text(encoding="utf-8")
        parts = [p for p in t.split("\n\n") if p.strip()]
        dst += [p.strip() + "\n\n" for p in parts]
        rep[f"squad_{split}"] = len(parts)

    # multiple-choice science and common sense, written as answers with their options shown
    def mc(path, dst, name, fact=False):
        n = 0
        for l in open(path, encoding="utf-8"):
            r = json.loads(l)
            qd = r["question"]
            ch = {c["label"]: c["text"] for c in qd["choices"]}
            if r.get("answerKey") not in ch:
                continue
            opts = "; ".join(c["text"] for c in qd["choices"])
            ans = ch[r["answerKey"]].strip()
            ans = ans[:1].upper() + ans[1:]
            if fact and r.get("fact1"):
                f1 = r["fact1"].strip()
                ans = ans.rstrip(".") + ". " + f1[:1].upper() + f1[1:].rstrip(".") + "."
            else:
                ans = ans.rstrip(".") + "."
            dst.append(ex(qd["stem"].strip() + " (Options: " + opts + ")", ans)); n += 1
        rep[name] = n
    A = UP / "arc" / "ARC-V1-Feb2018-2"
    for lvl in ("Easy", "Challenge"):
        mc(A / f"ARC-{lvl}" / f"ARC-{lvl}-Train.jsonl", train, f"arc_{lvl.lower()}_train")
        mc(A / f"ARC-{lvl}" / f"ARC-{lvl}-Dev.jsonl", train, f"arc_{lvl.lower()}_dev")
        mc(A / f"ARC-{lvl}" / f"ARC-{lvl}-Test.jsonl", held, f"arc_{lvl.lower()}_test")
    O = UP / "openbookqa" / "OpenBookQA-V1-Sep2018" / "Data" / "Additional"
    mc(O / "train_complete.jsonl", train, "openbookqa_train", fact=True)
    mc(O / "dev_complete.jsonl", train, "openbookqa_dev", fact=True)
    mc(O / "test_complete.jsonl", held, "openbookqa_test", fact=True)
    C = UP / "commonsenseqa"
    mc(C / "train_rand_split.jsonl", train, "commonsenseqa_train")
    mc(C / "dev_rand_split.jsonl", held, "commonsenseqa_dev")
    # downloaded 17 Oct with Adam's approval (data_uploads/): HotpotQA (two supporting Wikipedia paragraphs), Natural
    # Questions open (real Google searches, short answers), SciQ (science with its support paragraph), BoolQ (yes/no)
    import pyarrow.parquet as pq
    def rows_of(pattern):
        out = []
        for f in sorted(x for pat in pattern.split("|") for x in UP.glob(pat)):
            out += pq.read_table(f).to_pylist()
        return out
    def cap(t):
        t = str(t).strip()
        return t[:1].upper() + t[1:]
    def hotpot(r):
        ctx = dict(zip(r["context"]["title"], r["context"]["sentences"]))
        paras = []
        for t in dict.fromkeys(r["supporting_facts"]["title"]):
            if t in ctx:
                paras.append(t + ": " + "".join(ctx[t]).strip())
        return ("Passage: " + " ".join(paras) + "\nQuestion: " + r["question"], cap(r["answer"]).rstrip(".") + ".") if paras else None
    def sciq(r):
        if not r["support"].strip():
            opts = [r["correct_answer"], r["distractor1"], r["distractor2"], r["distractor3"]]; rng.shuffle(opts)
            return r["question"] + " (Options: " + "; ".join(opts) + ")", cap(r["correct_answer"]).rstrip(".") + "."
        return "Passage: " + r["support"].strip() + "\nQuestion: " + r["question"], cap(r["correct_answer"]).rstrip(".") + "."
    def boolq(r):
        q = cap(r["question"]).rstrip("?") + "?"
        return "Passage: " + r["passage"].strip() + "\nQuestion: " + q, "Yes." if r["answer"] else "No."
    def nq(r):
        return cap(r["question"]).rstrip("?") + "?", cap(r["answer"][0]).rstrip(".") + "."
    for name, fn, tr, he in (("hotpotqa", hotpot, "hotpotqa/distractor_train_*.parquet", "hotpotqa/distractor_validation_*.parquet"),
                             ("nq_open", nq, "nq_open/train_*.parquet", "nq_open/validation_*.parquet"),
                             ("sciq", sciq, "sciq/train_*.parquet", "sciq/validation_*.parquet|sciq/test_*.parquet"),
                             ("boolq", boolq, "boolq_hf/train_*.parquet", "boolq_hf/validation_*.parquet")):
        for pattern, dst, tag in ((tr, train, "train"), (he, held, "heldout")):
            n = 0
            for r in rows_of(pattern):
                qa = fn(r)
                if qa and american(qa[0] + " " + qa[1]):
                    dst.append(ex(*qa)); n += 1
            rep[f"{name}_{tag}"] = n
    rng.shuffle(train)
    seen_train = set(train)                               # an exchange in both splits stays only in training
    rep["heldout_dropped_as_in_train"] = sum(h in seen_train for h in held)
    held = [h for h in held if h not in seen_train]
    write(OUT / "shared" / "qa" / "qa_train.txt", "".join(train))
    write(OUT / "shared" / "eval" / "qa_heldout.txt", "".join(held))
    rep["train_exchanges"], rep["heldout_exchanges"] = len(train), len(held)
    return rep


# ------------------------------------------------------------------------------------------------------- knowledge
PERSON_RX = re.compile(r"\b(I|me|my|we|our|us|you|your|i'm|we're|you're)\b", re.I)
OPINION_RX = re.compile(r"\b(but|however|should|must|unfair|believe|think|probably|maybe|perhaps|God|President|Clinton|Obama|Bush|Trump|"
                        r"liberal|conservative|democrat|republican|church|bible)\b", re.I)
STATEMENT_RX = re.compile(r"\b(is|are|can|cause|causes|produce|produces|contain|contains|consist|consists|form|forms|"
                          r"absorb|absorbs|use|uses|need|needs|help|helps|make|makes|live|lives|grow|grows)\b")


def clean_sentence(t: str) -> bool:
    """A short, general science statement: no person, no opinion, no broken characters."""
    if not (40 <= len(t) <= 180) or not t[:1].isupper() or t[-1:] != "." or any(x in t for x in (";", "http", "|", "  ", "(vb", "\ufffd", '"', "?")):
        return False
    if PERSON_RX.search(t) or OPINION_RX.search(t) or not STATEMENT_RX.search(t):
        return False
    return sum(c.isalpha() for c in t) / len(t) > 0.8 and american(t)


def build_knowledge(rng: random.Random, n_arc: int = 120_000) -> dict:
    rows = [json.loads(l) for l in open(ROOT / "data_chat22" / "knowledge_wiki.jsonl", encoding="utf-8")]
    n_wiki = len(rows)
    for l in open(UP / "openbookqa" / "OpenBookQA-V1-Sep2018" / "Data" / "Main" / "openbook.txt", encoding="utf-8"):
        t = l.strip().strip('"')
        if t:
            t = t[:1].upper() + t[1:].rstrip(".") + "."
            rows.append({"q": "Science fact", "a": t, "key": t, "source": "openbook"})
    n_ob = len(rows) - n_wiki
    seen, arc = set(), []
    with open(UP / "arc" / "ARC-V1-Feb2018-2" / "ARC_Corpus.txt", encoding="utf-8", errors="replace") as f:
        for l in f:
            t = l.strip()
            k = t.lower()
            if k not in seen and clean_sentence(t):
                seen.add(k); arc.append(t)
    rng.shuffle(arc)
    rows += [{"q": "Science fact", "a": t, "key": t, "source": "arc_corpus"} for t in arc[:n_arc]]
    write(OUT / "shared" / "knowledge" / "knowledge_real.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return {"rows": len(rows), "wikipedia": n_wiki, "openbook_facts": n_ob, "arc_corpus_clean": len(arc), "arc_used": min(n_arc, len(arc))}


# ------------------------------------------------------------------------------------------------------- text, long
def docs_of(name: str) -> list[str]:
    return [d.strip() for d in (P100 / f"{name}.txt").read_bytes().decode("utf-8", "replace").split("\f") if d.strip()]


SEEN = None


def seen_by_cores() -> set:
    """FineWeb-Edu documents already in the P100 core mix (both cores were pre-trained on it): 81% of the file's first
    100,000 documents. Unseen documents are preferred: new text teaches more than a re-read."""
    global SEEN
    if SEEN is None:
        SEEN = {norm_key(d) for d in (P100 / "fineweb.txt").read_bytes().decode("utf-8", "replace").split("\f") if d.strip()}
    return SEEN


def norm_key(t: str) -> str:
    return re.sub(r"\s+", " ", t).strip()[:120]


def fineweb_edu_docs(min_score: float = 3.0, unseen: bool = False):
    import pyarrow.parquet as pq
    seen = seen_by_cores() if unseen else set()
    f = pq.ParquetFile(UP / "fineweb_edu" / "000_00000.parquet")
    for b in f.iter_batches(batch_size=20_000, columns=["text", "score", "language"]):
        d = b.to_pydict()
        for t, s, lang in zip(d["text"], d["score"], d["language"]):
            if lang == "en" and s >= min_score and american(t) and not (unseen and norm_key(t) in seen):
                yield t.strip()


# plain-text rehearsal per model size (MB): about one pass at the size's step budget (5M: 48k steps, 100M: 40k, 11% of
# windows of 2,049 codes, ~3.6 bytes per code); FineWeb-Edu from documents the cores never saw
TEXT_MB = {"5m": {"wiki": 130, "fineweb_edu": 340, "evidence": 74, "ultrachat": 111},
           "100m": {"wiki": 130, "fineweb_edu": 600, "evidence": 74, "ultrachat": 111}}
LONG_MIN_CHARS = {"5m": 28_000, "100m": 16_000}


def build_text(rng: random.Random, size: str = "5m") -> dict:
    """Wikipedia, unseen FineWeb-Edu (in place of the 1800s novels), evidence pairs, conversations; whole documents."""
    shares = TEXT_MB[size]
    parts, rep = [], {}
    # held-out text of the core mix and of the clean eval set: a document sharing any long paragraph with it stays out
    held_keys = held_paragraph_keys()
    rep["held_out_documents_excluded_keys"] = len(held_keys)
    for name, mb in shares.items():
        src = fineweb_edu_docs(unseen=True) if name == "fineweb_edu" else iter(rng.sample(docs_of(name), k=len(docs_of(name))))
        got, n, dropped, excluded = [], 0, 0, 0
        for d in src:
            if n >= mb * 1_000_000:
                break
            if name in ("wiki", "fineweb_edu") and not american(d):
                dropped += 1; continue
            if touches(d, held_keys):
                excluded += 1; continue
            got.append(d); n += len(d.encode("utf-8"))
        parts += got
        rep[name] = {"documents": len(got), "mb": round(n / 1e6, 1), "dropped_british": dropped, "held_out_overlap_dropped": excluded}
    rng.shuffle(parts)
    data = "\f".join(parts)
    name = f"rehearse_REAL_whole_{len(data.encode('utf-8')) // 1_000_000}MB_seed1.txt"
    for old in (OUT / size / "text").glob("rehearse_REAL_*.txt"):
        old.unlink()
    write(OUT / size / "text" / name, data)
    rep["file"] = name
    return rep



PARA_RX = re.compile(r"\n\s*\n|\f")
LINE_RX = re.compile(r"\n")


def held_paragraph_keys() -> set:
    keys = set()
    for hp in list((ROOT / "data_core" / "mix_P100" / "heldout").glob("*.txt")) + list((ROOT / "data_core" / "eval_clean").rglob("*.txt")):
        for para in PARA_RX.split(hp.read_bytes().decode("utf-8", "replace")):
            para = para.strip()
            if len(para) >= 200:
                keys.add(para[:120])
    return keys


def touches(doc: str, keys: set) -> bool:
    return any(len(q) >= 200 and q.strip()[:120] in keys for q in LINE_RX.split(doc))


NARR = re.compile(r"\b(was born|born in|died in|early life|biography|history of|century|war|empire|founded in|reign)\b", re.I)


def build_long(rng: random.Random, size: str = "5m") -> dict:
    l8, narr, rep = [], [], {}
    held = held_paragraph_keys()
    for name in ("wiki", "fineweb_edu"):
        src = fineweb_edu_docs(2.5) if name == "fineweb_edu" else docs_of("wiki")
        a = b = 0
        for d in src:
            if len(d) > LONG_MIN_CHARS[size] and american(d) and not touches(d, held):
                l8.append(d); a += 1
                if name == "wiki" and len(NARR.findall(d[:3000])) >= 3:
                    narr.append(d); b += 1
        rep[name] = {"over_8k": a, "narrative": b}
    write(OUT / size / "long" / "long_over8k_real.txt", "\f".join(l8))
    write(OUT / size / "long" / "narrative_nonfiction.txt", "\f".join(narr))
    rep["over_8k_total"], rep["narrative_total"] = len(l8), len(narr)
    rep["over_32k"] = sum(len(d) > 112_000 for d in l8)
    rep["min_chars"] = LONG_MIN_CHARS[size]
    return rep


# ------------------------------------------------------------------------------------------------------- look-alikes
def build_look(rng: random.Random) -> dict:
    rows = [json.loads(l) for l in open(ROOT / "data_chat22" / "knowledge_wiki.jsonl", encoding="utf-8")]
    out = [{"q": r["q"], "a": r["a"], "key": r["key"], "source": "wiki"} for r in rows]
    for p in (OUT / "shared" / "qa" / "qa_train.txt").read_text(encoding="utf-8").split("\n\n"):
        if p.startswith("User: ") and "\nAssistant: " in p:
            q, a = p[6:].split("\nAssistant: ", 1)
            if len(q) < 300 and len(a) < 400:
                out.append({"q": q, "a": a, "key": q, "source": "qa_real"})
    write(OUT / "shared" / "look" / "facts_pool_real.jsonl", "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))
    return {"rows": len(out)}


# ------------------------------------------------------------------------------------------------------- convo, skills
def build_convo(rng: random.Random) -> dict:
    (OUT / "shared" / "convo").mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "data_chat22" / "dialogues_train.jsonl", OUT / "shared" / "convo" / "dialogues_train.jsonl")
    n = 0
    with gzip.open(UP / "oasst1" / "2023-04-12_oasst_ready.trees.jsonl.gz", "rt", encoding="utf-8") as f, \
         open(OUT / "shared" / "convo" / "oasst_human_en.jsonl", "w", encoding="utf-8") as g:
        for l in f:
            tree = json.loads(l)
            if tree.get("prompt", {}).get("lang") != "en":
                continue
            turns, node = [], tree["prompt"]
            while node:
                turns.append(["user" if node["role"] == "prompter" else "assistant", node["text"]])
                kids = [k for k in node.get("replies", []) if not k.get("deleted")]
                kids.sort(key=lambda k: (k.get("rank") is None, k.get("rank") or 0))
                node = kids[0] if kids else None
            if len(turns) >= 2 and american(" ".join(t for _, t in turns)):
                g.write(json.dumps({"turns": turns}, ensure_ascii=False) + "\n"); n += 1
    return {"dialogues_copied": sum(1 for _ in open(OUT / "shared" / "convo" / "dialogues_train.jsonl", encoding="utf-8")), "oasst_human_en": n}


def build_skills(rng: random.Random) -> dict:
    (OUT / "shared" / "skills").mkdir(parents=True, exist_ok=True)
    for f in ("train_v4c.jsonl", "test_v2.json", "test_v4_rewrite.json"):
        shutil.copy2(ROOT / "data_skills" / f, OUT / "shared" / "skills" / f)
    return {"copied": 3, "note": "invented on purpose (conversation memory); never in the permanent store"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="qa,knowledge,text,long,look,convo,skills"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--size", default="5m", choices=["5m", "100m"], help="for text and long")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    man_p = OUT / "manifest.json"
    man = json.loads(man_p.read_text()) if man_p.exists() else {}
    for part in a.only.split(","):
        t0 = time.time()
        fn = globals()[f"build_{part}"]
        key = f"{part}_{a.size}" if part in ("text", "long") else part
        man[key] = fn(random.Random(a.seed), a.size) if part in ("text", "long") else fn(random.Random(a.seed))
        part = key
        man[part]["seconds"] = round(time.time() - t0)
        man_p.write_text(json.dumps(man, indent=1))
        print(part, json.dumps(man[part])[:600], flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
