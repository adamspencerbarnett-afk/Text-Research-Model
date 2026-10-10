"""Chat probe: the same conversation battery for every Q&A model, so models can be compared by
reading what they say, not only by F1.

    python compiler/experiments/chat_probe.py models/qa_best_5M.pt [--device cuda]

Writes docs/chat_probes/<model>.md (the transcript, with what memory returned for each answer)
and results/probe_<model>.json (the same plus an automatic keyword check). The grading notes
are written by hand in docs/CHAT_NOTES.md after reading the transcript.

Sections (written from what the models were trained on, 10 October 2026):
  stored     questions that are in the Alpaca training data, asked verbatim
  reworded   the same questions in other words (tests retrieval + copy under paraphrase)
  knowledge  general facts never given as such
  heldout    held-out Alpaca instructions (in no memory)
  taught     facts added with /add during the chat, then asked verbatim and reworded
  arithmetic small sums in words
  chat       everyday conversation
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "compiler" / "experiments"))
sys.path.insert(0, str(ROOT / "compiler"))
import m2_qa  # noqa: E402

TEACH = [
    ("What is the name of Adam's research model?", "Adam's research model is called the Text Research Model."),
    ("What GPU does the research computer use?", "The research computer uses an NVIDIA RTX 3090 GPU."),
    ("When was Version 1 of the compiler frozen?", "Version 1 of the compiler was frozen on 9 October 2026."),
]

# (section, question, keywords that a correct answer contains; any one of each group, groups AND-ed)
PROBES = [
    ("stored", "Name a food that is traditionally served during Thanksgiving in the US.", [["turkey"]]),
    ("stored", "Given the input, provide an example to demonstrate the concept of gravity.", [["fall", "drop"]]),
    ("stored", "Create a list of 10 verbs related to cooking", [["boil", "bake", "fry", "roast", "chop"]]),
    ("stored", "Name some computer viruses", [["wannacry", "stuxnet", "melissa", "mydoom", "iloveyou", "code red"]]),
    ("stored", "Come up with two statistics related to the US population.", [["million", "%", "percent"]]),
    ("reworded", "What food do Americans usually eat on Thanksgiving?", [["turkey"]]),
    ("reworded", "Can you give me an example that shows how gravity works?", [["fall", "drop"]]),
    ("reworded", "List ten verbs you would use when cooking.", [["boil", "bake", "fry", "roast", "chop"]]),
    ("reworded", "What are some well-known computer viruses?", [["wannacry", "stuxnet", "melissa", "mydoom", "iloveyou", "code red"]]),
    ("reworded", "Tell me two facts about the population of the United States.", [["million", "%", "percent"]]),
    ("knowledge", "What is the largest planet in our solar system?", [["jupiter"]]),
    ("knowledge", "Who wrote Romeo and Juliet?", [["shakespeare"]]),
    ("knowledge", "What is photosynthesis?", [["light", "sun"], ["plant"]]),
    ("knowledge", "What is the boiling point of water in Celsius?", [["100"]]),
    ("knowledge", "What is the capital of Japan?", [["tokyo"]]),
    ("heldout", "Generate a creative poem describing the night sky.", [["star", "moon", "night"]]),
    ("heldout", "Come up with a creative sentence to describe a summer day.", [["sun", "summer", "warm", "hot"]]),
    ("heldout", "List 5 famous Italian dishes.", [["pizza", "pasta", "lasagna", "risotto", "spaghetti"]]),
    ("heldout", "Tell me the most common hairstyle for men.", [["cut", "short"]]),
    ("heldout", 'Print the following statement: "I want to learn to code".', [["i want to learn to code"]]),
    ("taught", "What is the name of Adam's research model?", [["text research model"]]),
    ("taught", "What GPU does the research computer use?", [["3090"]]),
    ("taught", "When was Version 1 of the compiler frozen?", [["9 october", "october 9"]]),
    ("taught", "What is Adam's research model called?", [["text research model"]]),
    ("taught", "Which graphics card is in the research computer?", [["3090"]]),
    ("taught", "On what date was Version 1 frozen?", [["9 october", "october 9"]]),
    ("arithmetic", "What is 12 plus 30?", [["42"]]),
    ("arithmetic", "If I have 5 apples and eat 2, how many are left?", [["3", "three"]]),
    ("arithmetic", "What is 7 times 8?", [["56"]]),
    ("chat", "Hello, how are you today?", [["fine", "good", "well", "great", "doing"]]),
    ("chat", "Can you help me write a short email to my boss asking for a day off?", [["dear", "hi", "hello"], ["day off", "leave", "day"]]),
    ("chat", "What should I cook for dinner tonight?", [["chicken", "pasta", "salad", "soup", "rice", "fish", "vegetable"]]),
]


def check(answer: str, groups) -> bool:
    low = answer.lower()
    return all(any(k in low for k in g) for g in groups)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("--device", default="cpu"); ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--no-repeat", type=int, default=3); ap.add_argument("--max-codes", type=int, default=120)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    native = ROOT / "data" / "cv0.exe"
    eng = m2_qa.QAEngine(a.model, native=str(native) if native.exists() else None, device=a.device)
    for q, ans in TEACH:
        eng.add(q.encode(), ans.encode())
    rows = []
    for section, q, groups in PROBES:
        t0 = time.perf_counter()
        out, info = eng.ask(q.encode(), a.max_codes, a.no_repeat)
        alone, _ = eng.ask(q.encode(), a.max_codes, a.no_repeat, use_memory=False)
        text, text_alone = out.decode("utf-8", "replace"), alone.decode("utf-8", "replace")
        rows.append({"section": section, "question": q, "answer": text, "answer_without_memory": text_alone,
                     "keywords_ok": check(text, groups), "keywords_ok_without_memory": check(text_alone, groups),
                     "memory": info["memory"], "similarity": info["similarity"],
                     "retrieved_question": info.get("retrieved_question"), "seconds": round(time.perf_counter() - t0, 2)})
    stem = Path(a.model).stem
    summary = {}
    for r in rows:
        s = summary.setdefault(r["section"], {"n": 0, "ok": 0, "ok_without_memory": 0})
        s["n"] += 1; s["ok"] += r["keywords_ok"]; s["ok_without_memory"] += r["keywords_ok_without_memory"]
    (ROOT / "results" / f"probe_{stem}.json").write_text(json.dumps({"model": a.model, "summary": summary, "rows": rows}, indent=1))
    md = [f"# Chat probe: `{a.model}`\n\n*{time.strftime('%Y-%m-%d %H:%M')}. Greedy decoding, repeated 3-code phrases blocked, "
          f"replies up to {a.max_codes} codes. Facts in the 'taught' section were added with /add just before, with no training. "
          "Each question is asked with memory (the model as designed) and without (the network alone). "
          "The keyword check is a rough automatic flag; the grades are in docs/CHAT_NOTES.md.*\n\n",
          "| section | keyword check, with memory | without memory |\n| --- | ---: | ---: |\n"]
    md += [f"| {k} | {v['ok']}/{v['n']} | {v['ok_without_memory']}/{v['n']} |\n" for k, v in summary.items()]
    cur = None
    for i, r in enumerate(rows, 1):
        if r["section"] != cur:
            cur = r["section"]; md.append(f"\n## {cur}\n")
        mem = f"{r['memory']} ({r['similarity']:.2f}): {r['retrieved_question']}" if r["retrieved_question"] else r["memory"]
        md.append(f"\n**{i}. {r['question']}**\n\n> {r['answer'] or '(empty)'}\n\n"
                  f"- memory used: {mem}\n- keywords: {'yes' if r['keywords_ok'] else 'no'}\n"
                  f"- network alone: {r['answer_without_memory'][:300] or '(empty)'}\n")
    out_md = ROOT / "docs" / "chat_probes" / f"{stem}.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_bytes("".join(md).encode("utf-8"))
    print(json.dumps(summary)); print(f"wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
