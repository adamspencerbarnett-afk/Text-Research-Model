"""One scorer for every chat22 test (the test review, 16 October: three abstain lists, substring keys, no wrong-value
veto, no intervals).

  abstained(text)        says it does not have / know the fact (curly apostrophes normalised; "you don't have to ..."
                         and similar are NOT abstentions)
  has(text, key)         whole-word match; a key of 1-3 characters or a number must be a whole token
  right(text, want, wrong)  every 'want' group present (any one alternative per group) and NO 'wrong' value present
  asks(text)             a clarifying question ("Do you mean X or Y?")
  rate(k, n)             share with a 95% Wilson interval
  diff_holds(k1, n1, k0, n0)  the new share is not lower than the reference by more than 2 standard errors of the
                         difference (the guard's rule)"""
from __future__ import annotations

import math
import re

ABSTAIN = [r"\bdon'?t have (that|any|this|it|information|knowledge|a record)", r"\bdo not have (that|any|this|it|information|knowledge)",
           r"\b(i )?don'?t know\b", r"\bnot in my memory\b", r"\bhaven'?t told me\b", r"\bno information (about|on)\b",
           r"\bi'?m not sure\b", r"\bcould you tell me\b", r"\bnot (mentioned|stated) (in|anywhere)"]
NOT_ABSTAIN = [r"\byou don'?t have to\b", r"\bdon'?t know (if|whether) you\b", r"\bif you don'?t know\b"]


def norm(text) -> str:
    t = text.decode("utf-8", "replace") if isinstance(text, bytes) else text
    return t.replace("’", "'").replace("‘", "'").lower()


def abstained(text) -> bool:
    t = norm(text)
    for p in NOT_ABSTAIN:
        t = re.sub(p, " ", t)
    return any(re.search(p, t) for p in ABSTAIN)


def has(text, key: str) -> bool:
    t, k = norm(text), key.lower().strip()
    if not k:
        return False
    if len(k) <= 3 or k.replace(".", "").replace(",", "").isdigit():
        return re.search(r"(?<![a-z0-9])" + re.escape(k) + r"(?![a-z0-9])", t) is not None
    return re.search(r"(?<![a-z0-9])" + re.escape(k), t) is not None


def right(text, want_groups, wrong=()) -> bool:
    ok = all(any(has(text, k) for k in g) for g in want_groups)
    return ok and not any(has(text, w) for w in wrong if w)


def asks(text) -> bool:
    t = norm(text)
    return "?" in t and (" or " in t or "which" in t or "do you mean" in t)


def rate(k: int, n: int) -> dict:
    if n == 0:
        return {"k": 0, "n": 0, "share": None, "ci95": None}
    p, z = k / n, 1.96
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return {"k": k, "n": n, "share": round(p, 3), "ci95": [round(max(0, c - h), 3), round(min(1, c + h), 3)]}


def diff_holds(k1: int, n1: int, k0: int, n0: int) -> bool:
    if n1 == 0 or n0 == 0:
        return False
    p1, p0 = k1 / n1, k0 / n0
    se = math.sqrt(max(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0, 1e-9))
    return p1 >= p0 - 2 * se
