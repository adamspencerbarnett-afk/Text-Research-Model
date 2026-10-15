"""Audit every dataset the 5M chat model v2 trains on (15 October 2026), before training.

    python compiler/experiments/audit_chat_data.py [--sample 20000]

Checks, per source:
  skills v4     kinds, duplicates, empty fields, test leakage (test v1 names, held-out names reserved for test v2 and the
                rewriting test, identical questions), answer present in the evidence for every answerable kind,
                'My ...' left in answers to the user, typo share, window length in codes (cut windows)
  conversations turn order, empty turns, very long replies, non-English share, duplicates, overlap with the held-out set
  Alpaca Q&A    duplicates, empty answers, overlap with the skills tests
  recall        generated long-conversation windows: fact kept in the window or moved to the memory slot, answer present,
                reserved names, window length
Writes results/chat5m_data_audit.json and prints the findings.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import chat_engine as CE  # noqa: E402
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402
import ladder_encodings as LE  # noqa: E402
import m2_qa  # noqa: E402
import testset as T  # noqa: E402

WORD = re.compile(r"[a-z0-9][a-z0-9@.\-#']*")
STOPW = set(m2_qa.STOP) | {"your", "yes", "no", "now", "is", "are"}
ANSWERABLE = {"single", "nearmiss", "update", "update_earlier", "combo", "followup", "short", "yesno", "list"}


def words(s: str) -> set[str]:
    return {w.strip(".'") for w in WORD.findall(s.lower()) if w.strip(".'") not in STOPW and len(w) > 1}


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--sample", type=int, default=20000)
    ap.add_argument("--write-clean", action="store_true", help="write data_skills/train_v4c.jsonl: duplicates and windows whose "
                    "question is word for word a test question removed")
    a = ap.parse_args()
    rng = np.random.default_rng(3)
    core = __import__("torch").load(ROOT / "runs/core_mt5m/scale_custom_v0_8k.pt", map_location="cpu", weights_only=False)
    enc = LE.V0Dict(str(ROOT / core["encoding"]["dictionary"]), str(ROOT / "data" / "cv0.exe"))
    sep = enc.encode(CE.SEP_TEXT)[0]
    rep = {}

    # ---------------------------------------------------------------- skills v4
    G3.setup(test=False); NAMES_TRAIN = set(G.NAMES); RESERVED = set(G3.RESERVED)
    v1_names = {w.lower() for w in T.TEST_NAMES}
    reserved = {w.lower() for w in RESERVED}
    test_qs = {q for _, q, _, _ in T.items()}
    for f in ("test_v2.json", "test_v4_rewrite.json"):
        test_qs |= {i[1] for i in json.loads((ROOT / "data_skills" / f).read_text())["items"]}
    test_qs_n = {re.sub(r"[^a-z0-9 ]", "", q.lower()).strip() for q in test_qs}
    kinds, dup, empty, leak_v1, leak_res, leak_q, gold_miss, my_left, typo, lens, cut = (collections.Counter(), 0, 0, collections.Counter(),
                                                                                         collections.Counter(), 0, collections.Counter(), 0, 0, [], 0)
    seen = set(); miss_examples = []; clean = []
    lines = (ROOT / "data_skills" / "train_v4.jsonl").read_text(encoding="utf-8").splitlines()
    for li, l in enumerate(lines):
        w = json.loads(l); kinds[w["kind"]] += 1
        key = (w["q"], w["target"], tuple(map(tuple, w["cands"])))
        is_dup = key in seen; dup += is_dup; seen.add(key)
        same_q = re.sub(r"[^a-z0-9 ]", "", w["q"].lower()).strip() in test_qs_n
        if not is_dup and not same_q:
            clean.append(l)
        if not w["q"].strip() or not w["target"].strip():
            empty += 1
        txt = " ".join([w["q"], w["target"]] + [x for c in w["cands"] + w["turns"] for x in c])
        toks = set(re.findall(r"[A-Za-z][a-z]+", txt))
        for n in toks:
            if n.lower() in reserved and n in RESERVED:
                leak_res[n] += 1
        for n in {t.lower() for t in toks} & v1_names:
            leak_v1[n] += 1
        leak_q += re.sub(r"[^a-z0-9 ]", "", w["q"].lower()).strip() in test_qs_n
        if w["kind"] in ANSWERABLE:
            ev = set().union(*[words(c[0] + " " + c[1]) for c in w["cands"]]) if w["cands"] else set()
            tw = words(w["target"]) - words(w["q"]) - {"yes", "no"}
            if tw and not tw <= ev:
                gold_miss[w["kind"]] += 1
                if len(miss_examples) < 6:
                    miss_examples.append({"kind": w["kind"], "q": w["q"], "target": w["target"], "missing": sorted(tw - ev)[:5]})
        if w["kind"] not in ("clarify", "unknown") and w["target"].startswith("My "):
            my_left += 1
        if li < a.sample:
            codes, ev_len, pl, kept = CE.build_window(enc, sep, [(B(c[0]), B(c[1])) for c in w["cands"]],
                                                      [(B(t[0]), B(t[1])) for t in w["turns"]], B(w["q"]), B(w["target"]), 2048)
            lens.append(len(codes)); cut += len(codes) > 2049
    # typo share: questions that no longer parse into dictionary words are hard to count; use the generator's own marker
    # (build_skills_data adds typos to 10% of questions after generation): count questions with a doubled or dropped letter
    if a.write_clean:
        with open(ROOT / "data_skills" / "train_v4c.jsonl", "w", encoding="utf-8") as fo:
            fo.writelines(x + "\n" for x in clean)
    rep["skills_v4"] = {"windows": len(lines), "clean_windows": len(clean), "kinds": dict(kinds), "duplicates": dup, "empty": empty,
                        "test_v1_name_hits": dict(leak_v1.most_common(8)), "reserved_name_hits": dict(leak_res.most_common(8)),
                        "identical_test_questions": leak_q, "answer_not_in_evidence": dict(gold_miss), "answer_not_in_evidence_examples": miss_examples,
                        "answers_starting_My": my_left, "codes_p50_p99_max": [int(np.percentile(lens, 50)), int(np.percentile(lens, 99)), int(max(lens))],
                        "windows_over_2049": cut}

    # ---------------------------------------------------------------- conversations
    bad_order = empty_t = long_r = nonen = 0
    seen_first, dup_c, lens_r = set(), 0, []
    held_first = {json.loads(l)["turns"][0][1][:200] for l in open(ROOT / "data_mt" / "heldout" / "dialogues.jsonl", encoding="utf-8")}
    overlap = 0
    n = 0
    for l in open(ROOT / "data_mt" / "train" / "dialogues.jsonl", encoding="utf-8"):
        t = json.loads(l)["turns"]; n += 1
        roles = [r for r, _ in t]
        bad_order += any(roles[i] == roles[i + 1] for i in range(len(roles) - 1)) or roles[0] != "user"
        empty_t += any(not x.strip() for _, x in t)
        first = t[0][1][:200]; dup_c += first in seen_first; seen_first.add(first); overlap += first in held_first
        if n <= a.sample:
            for r, x in t:
                if r == "assistant":
                    c = len(enc.encode(x.encode("utf-8", "replace"))[0]); lens_r.append(c); long_r += c > 1024
            txt = " ".join(x for _, x in t)
            nonen += (sum(ch.isascii() for ch in txt) / max(1, len(txt))) < 0.9
    rep["conversations"] = {"threads": n, "bad_turn_order": bad_order, "empty_turns": empty_t, "duplicate_first_messages": dup_c,
                            "first_message_in_heldout": overlap, f"replies_over_1024_codes_in_first_{a.sample}": long_r,
                            "reply_codes_p50_p99": [int(np.percentile(lens_r, 50)), int(np.percentile(lens_r, 99))],
                            f"non_english_threads_in_first_{a.sample}": nonen}

    # ---------------------------------------------------------------- Alpaca
    ax = m2_qa.parse_exchanges((ROOT / "data_qa" / "train" / "qa_train.txt").read_bytes())
    qs = collections.Counter(q for q, _ in ax)
    rep["alpaca"] = {"exchanges": len(ax), "duplicate_questions": sum(c - 1 for c in qs.values() if c > 1),
                     "empty_answers": sum(not x.strip() for _, x in ax),
                     "skills_test_questions_present": sum(re.sub(rb"[^a-z0-9 ]", b"", q.lower()).strip().decode() in test_qs_n for q, _ in ax)}
    print(json.dumps(rep, indent=1))
    (ROOT / "results" / "chat5m_data_audit.json").write_text(json.dumps(rep, indent=1))
    return 0


def B(x):
    return x if isinstance(x, bytes) else x.encode("utf-8", "replace")


if __name__ == "__main__":
    raise SystemExit(main())
