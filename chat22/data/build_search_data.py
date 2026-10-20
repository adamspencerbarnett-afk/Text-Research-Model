"""Training and test data for search engine v2 (docs/SEARCH_ENGINE_V2.md), written to data_chat22/search_v2/.

  extract   statements -> fact-table records (who, attribute, value, correction?, link?), with ground truth from the
            fact generator: "My sister Kim lives in Oslo." -> (my sister kim, city, Oslo); "Actually, Kim moved to Bergen"
            -> a correction; "My sister is called Kim" -> a link (sister -> Kim). Train: training names and phrasings;
            test: the reserved names and phrasings (G3.setup(test=True)).
  query     questions -> (type, who, attribute): personal facts with ground truth, general questions (Alpaca),
            chit-chat, arithmetic / calendar (tool). Train and test as above.
  judge     (question, memory item, label) pairs for the judge: 1 = the item answers the question, 0 = it does not; hard
            negatives come from what the real search returned (look-alikes, outdated values, same-topic stored facts).
            Sources: the any-length session rows (longconv_s1_*.jsonl), the skills records (gold by answer overlap),
            Q&A (own exchange vs its nearest neighbours). Held out: the last 2% of session rows, the last 2,000 skills.

    python -m chat22.data.build_search_data [--only extract,query,judge]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data_chat22" / "search_v2"
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402

WORD = re.compile(r"[a-z0-9][a-z0-9@.\-#']*")
STOPW = {"the", "a", "an", "is", "are", "of", "to", "in", "and", "your", "my", "yes", "no", "now", "it", "on", "at", "for"}


def words(s: str) -> set:
    return {w.strip(".'") for w in WORD.findall(s.lower()) if w.strip(".'") not in STOPW and len(w) > 1}


def extract_rows(rng, n: int) -> list:
    rows = []
    while len(rows) < n:
        f = G.Fact(rng)
        rec = {"who": f.ent["id"], "attribute": f.field, "value": str(f.value), "kind": f.kind}
        rows.append({"text": f.stored()[1], "record": rec, "type": "statement"})
        if f.can_update() and rng.random() < 0.3:
            (_, new_s), _ = f.update()
            rows.append({"text": "Actually, " + new_s[0].lower() + new_s[1:], "record": dict(rec, value=str(f.new_value)), "type": "correction"})
        if rng.random() < 0.15:
            rel, nm = G.pick(rng, G.RELS), G.pick(rng, G.NAMES)
            rows.append({"text": G.pick(rng, [f"My {rel}'s name is {nm}.", f"My {rel} is called {nm}."]),
                         "record": {"link": f"my {rel}", "name": nm}, "type": "link"})
    return rows[:n]


def query_rows(rng, n: int, alpaca: list) -> list:
    greet = ["Hi", "Hello!", "How are you today?", "Thanks!", "Good night.", "Hey there", "Bye for now."]
    math = ["What is 12 plus 30?", "What is 7 times 8?", "I have three apples and eat one. How many are left?",
            "What day comes after Tuesday?", "What month is before June?"]
    rows = []
    while len(rows) < n:
        r = rng.random()
        if r < 0.6:
            f = G.Fact(rng)
            rows.append({"text": f.ask(), "type": "personal_fact", "who": f.ent["id"], "attribute": f.field})
        elif r < 0.85:
            rows.append({"text": alpaca[int(rng.integers(len(alpaca)))], "type": "general", "who": None, "attribute": None})
        elif r < 0.93:
            rows.append({"text": G.pick(rng, greet), "type": "chat", "who": None, "attribute": None})
        else:
            rows.append({"text": G.pick(rng, math), "type": "tool", "who": None, "attribute": None})
    return rows


def judge_pairs_from_longconv(lines: list) -> list:
    out = []
    for l in lines:
        r = json.loads(l)
        val = (r.get("value") or "").lower(); tw = words(r["target"])
        for q, a, src, t in r["items"]:
            text = (q + " " + a)
            low = text.lower()
            hit = (val in low) if val else (not r["absent"] and len(words(low) & tw) >= max(1, len(tw) // 2))
            out.append({"q": r["q"], "item": text[:600], "source": src, "label": int(bool(hit) and not r["absent"]),
                        "from": "longconv:" + r["source"]})
    return out


def judge_pairs_from_skills(recs: list) -> list:
    out = []
    answerable = {"single", "nearmiss", "update", "update_earlier", "combo", "followup", "short", "yesno", "list"}
    for w in recs:
        tw = words(w["target"]); ov = [len(words(c[1]) & tw) for c in w["cands"]] or [0]
        for j, c in enumerate(w["cands"]):
            good = w["kind"] in answerable and ov[j] > 0 and (ov[j] == max(ov) or w["kind"] == "list")
            out.append({"q": w["q"], "item": (c[0] + " " + c[1])[:600], "source": "fact", "label": int(good), "from": "skills:" + w["kind"]})
    return out


def judge_pairs_from_qa(rng, n: int) -> list:
    sys.path.insert(0, str(ROOT))
    from chat22 import encoder as E
    qa = E.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
    nb = np.load(ROOT / "runs" / "chat5m_v2" / "dense_top4.npy")
    out = []
    for i in rng.choice(len(qa), n, replace=False):
        q, a = qa[int(i)]
        if not a.strip():
            continue
        out.append({"q": q.decode("utf-8", "replace"), "item": (q + b" " + a).decode("utf-8", "replace")[:600], "source": "fact", "label": 1, "from": "qa:own"})
        for j in nb[int(i)][:2]:
            q2, a2 = qa[int(j)]
            out.append({"q": q.decode("utf-8", "replace"), "item": (q2 + b" " + a2).decode("utf-8", "replace")[:600], "source": "fact", "label": 0, "from": "qa:neighbour"})
    return out


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(rows)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="extract,query,judge")
    a = ap.parse_args()
    man = {}
    sys.path.insert(0, str(ROOT))
    from chat22 import encoder as E
    alpaca = [q.decode("utf-8", "replace") for q, _ in E.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())]
    alpaca_test = [q.decode("utf-8", "replace") for q, _ in E.parse_exchanges((ROOT / "data_qa" / "heldout" / "qa_heldout.txt").read_bytes())]
    for part in a.only.split(","):
        if part in ("extract", "query"):
            for split, test in (("train", False), ("test", True)):
                G3.setup(test=test)
                rng = np.random.default_rng(1 if not test else 2)
                rows = extract_rows(rng, 60000 if not test else 3000) if part == "extract" else \
                       query_rows(rng, 60000 if not test else 3000, alpaca if not test else alpaca_test)
                man[f"{part}_{split}"] = write(f"{part}_{split}.jsonl", rows)
        if part == "judge":
            G3.setup(test=False)
            rng = np.random.default_rng(3)
            lc = []
            for f in sorted((ROOT / "data_chat22").glob("longconv_s1_*.jsonl")):
                if f.name.endswith("_99.jsonl"):
                    continue
                lc += open(f, encoding="utf-8").readlines()
            cut = max(1, len(lc) // 50) if lc else 0
            sk = [json.loads(l) for l in open(ROOT / "data_skills" / "train_v4c.jsonl", encoding="utf-8")]
            train = judge_pairs_from_longconv(lc[:-cut] if lc else []) + judge_pairs_from_skills(sk[:-2000]) + judge_pairs_from_qa(rng, 30000)
            test = judge_pairs_from_longconv(lc[-cut:] if lc else []) + judge_pairs_from_skills(sk[-2000:])
            rng.shuffle(train)
            man["judge_train"] = write("judge_train.jsonl", train); man["judge_test"] = write("judge_test.jsonl", test)
            man["judge_positive_share"] = round(sum(r["label"] for r in train) / max(1, len(train)), 3)
            man["judge_longconv_rows_used"] = len(lc)
    (OUT / "manifest.json").write_text(json.dumps(man, indent=1))
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
