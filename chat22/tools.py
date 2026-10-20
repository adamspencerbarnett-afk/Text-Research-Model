"""Answer-time tools (copied unchanged from compiler/experiments/chat_engine.py): simple arithmetic and the day or
month after / before another. The result goes into the memory block as a [fact] item; the model reads and copies it.
Phone numbers, codes and times never trigger."""
from __future__ import annotations

import re


def run(msg: bytes):
    """(question, result) for the memory block, or None."""
    text = msg.decode("utf-8", "replace").lower()
    r = arithmetic(text) or calendar(text)
    return (msg, r.encode()) if r else None


NUMS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
        "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18,
        "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "hundred": 100}
NUM = r"(\d+(?:\.\d+)?|" + "|".join(NUMS) + r")"
OPS = {"plus": "+", "+": "+", "add": "+", "minus": "-", "-": "-", "times": "*", "x": "*", "*": "*", "multiplied by": "*",
       "divided by": "/", "/": "/", "over": "/"}
TAKE = r"(?:eat|ate|eats|lose|lost|give away|gave away|give|gave|use|used|spend|spent|sell|sold|throw away|break|broke)"
GET = r"(?:buy|bought|get|got|find|found|receive|received|add|added|pick|picked)"


def _n(s: str) -> float:
    return float(NUMS[s]) if s in NUMS else float(s)


def _fmt(v: float) -> str:
    return str(int(v)) if float(v).is_integer() else f"{v:.4g}"


def arithmetic(t: str) -> str | None:
    """'What is 12 plus 30?', '7 x 8', 'If I have three apples and eat one, how many are left?' -> a short worked result."""
    # word operators, or a symbol with spaces on both sides: phone numbers (555-0142), codes and times never match
    m = (re.search(r"\b" + NUM + r"\s+(multiplied by|divided by|plus|minus|times|over)\s+" + NUM + r"\b", t)
         or re.search(r"(?<![\d:.\-])" + NUM + r"\s+([-+*/x])\s+" + NUM + r"(?![\d:.\-])", t))
    if m:
        x, op, y = _n(m.group(1)), OPS[m.group(2)], _n(m.group(3))
        if op == "/" and y == 0:
            return None
        v = {"+": x + y, "-": x - y, "*": x * y, "/": x / y if y else 0}[op]
        word = {"+": "plus", "-": "minus", "*": "times", "/": "divided by"}[op]
        return f"{_fmt(x)} {word} {_fmt(y)} is {_fmt(v)}."
    m = re.search(r"\bhave\s+" + NUM + r"\s+([a-z]+).*?\b" + TAKE + r"\s+" + NUM, t)
    if m:
        x, thing, y = _n(m.group(1)), m.group(2), _n(m.group(3))
        return f"{_fmt(x)} {thing} minus {_fmt(y)} leaves {_fmt(x - y)} {thing}."
    m = re.search(r"\bhave\s+" + NUM + r"\s+([a-z]+).*?\b" + GET + r"\s+" + NUM, t)
    if m:
        x, thing, y = _n(m.group(1)), m.group(2), _n(m.group(3))
        return f"{_fmt(x)} {thing} plus {_fmt(y)} makes {_fmt(x + y)} {thing}."
    return None


DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]


def calendar(t: str) -> str | None:
    """'What comes after Tuesday?', 'Which month is before March?' -> the neighbouring day or month."""
    m = re.search(r"\b(after|before|follows|precedes)\s+(" + "|".join(DAYS + MONTHS) + r")\b", t)
    if not m:
        return None
    seq = DAYS if m.group(2) in DAYS else MONTHS
    i = seq.index(m.group(2)); step = 1 if m.group(1) in ("after", "follows") else -1
    return f"The {'day' if seq is DAYS else 'month'} {'after' if step > 0 else 'before'} {m.group(2).title()} is {seq[(i + step) % len(seq)].title()}."
