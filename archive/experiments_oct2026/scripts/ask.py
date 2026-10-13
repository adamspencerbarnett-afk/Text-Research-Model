"""Ask a chat-trained model a list of questions through its compiler and print the answers.

    python compiler/experiments/ask.py CHECKPOINT --data DATA_QA [--native ./cv0] [--questions q.txt]

Each question is wrapped as "User: <question>\nAssistant:" (the training format); generation
stops at the next "User:" or after --tokens codes. Prints the answers and writes them as JSON.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chat  # noqa: E402

DEFAULT = ["What is the capital of France?", "What color is the sky?", "Name three fruits.", "What is 2 + 2?",
           "Give three tips for staying healthy.", "Who wrote Pride and Prejudice?", "Translate 'good morning' into Spanish.",
           "Why is the ocean salty?", "Write one sentence about a dog.", "What is the opposite of hot?"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint"); ap.add_argument("--data"); ap.add_argument("--native"); ap.add_argument("--questions")
    ap.add_argument("--tokens", type=int, default=80); ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-k", type=int, default=20); ap.add_argument("--seed", type=int, default=1); ap.add_argument("--out")
    a = ap.parse_args()
    questions = [l.strip() for l in open(a.questions) if l.strip()] if a.questions else DEFAULT
    enc, model = chat.load(a.checkpoint, a.native, a.data)
    rows = []
    for q in questions:
        out, stats = chat.reply(enc, model, f"User: {q}\nAssistant:".encode("utf-8"), a.tokens, a.temperature, a.top_k, a.seed)
        text = out.decode("utf-8", "replace")
        text = text.split("\nUser:")[0].split("User:")[0].strip()
        rows.append({"question": q, "answer": text, "bytes_per_s": stats["bytes_per_s"]})
        print(f"Q: {q}\nA: {text}\n")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
