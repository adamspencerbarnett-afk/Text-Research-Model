"""Skills generator version 4 (15 October 2026, Adam: "answers in its own words in the form you asked for").
Version 3 plus rewriting: the reply is built from the stored fact but not copied from it.

  short        "Just the answer: where does Kim live?"                 -> "Oslo."
  yesno        "Is it true that Kim lives in Lyon?"                     -> "No, Kim lives in Oslo."  (or "Yes, ...")
  list         two facts about one person, "List Kim's job and city."   -> "- Job: nurse" / "- City: Oslo"
  perspective  every reply about the user's own people speaks to the user: stored "My sister Kim lives in Oslo."
               is answered "Your sister Kim lives in Oslo." (version 3 copied "My ..." back)

    import gen_skills_v4 as G4
    G4.setup(test=False); w = G4.make(rng, alpaca)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_skills as G  # noqa: E402
import gen_skills_v3 as G3  # noqa: E402

pick = G.pick
SHORT = ["Just the answer: {q}", "In a few words, {ql}", "Answer briefly: {q}", "Short answer please: {q}", "One or two words: {q}"]
YESNO = ["Is it true that {s}?", "Am I right that {s}?", "Is it correct that {s}?", "Can you confirm that {s}?"]
LIST = ["List {ep} {f1} and {f2}.", "Give me {ep} {f1} and {f2} as a list.", "What are {ep} {f1} and {f2}? Use a list.",
        "Make a short list of {ep} {f1} and {f2}."]
FIELD_NAMES = {"city": "city", "job": "job", "food": "favourite food", "colour": "favourite colour", "sport": "favourite sport",
               "band": "favourite band", "drink": "favourite drink", "car": "car", "phone": "phone number", "birthday": "birthday",
               "pet": "pet's name", "age": "age", "allergy": "allergy", "hobby": "hobby", "work": "workplace", "surname": "last name",
               "book": "favourite book", "school": "school", "language": "language", "instrument": "instrument", "address": "address",
               "email": "email"}
MIX4 = ["v3"] * 7 + ["short", "yesno", "list"]


def setup(test: bool = False):
    G3.setup(test=test)


def your(s: str) -> str:
    """The user's own people, spoken to the user: 'My sister ...' -> 'Your sister ...'."""
    s = re.sub(r"^My ", "Your ", s)
    return re.sub(r"\bmy\b", "your", s)


def short_value(f) -> str:
    v = f.value
    if f.kind == "t" and f.field in ("when", "deadline"):
        v = f"{f.fmt['d']} at {v}" if f.field == "when" else f.fmt["d"]
    elif f.field == "price":
        v = f"{v} dollars"
    elif f.field == "age":
        v = f"{v} years old"
    return v[0].upper() + v[1:] + "."


def lower_first(s: str) -> str:
    """Lower-case a leading 'My', 'The', 'Your' ... so a sentence can follow 'Is it true that'; names stay as they are."""
    w = s.split(" ", 1)[0]
    return w.lower() + s[len(w):] if w in ("My", "The", "Your", "Parking", "Just", "In", "Answer", "Short", "One") else s


def make(rng, alpaca=None):
    kind = pick(rng, MIX4)
    if kind == "short":
        f = G.Fact(rng)
        cands = [f.stored()] + [G._other(rng, {f.ent["id"]}).stored() for _ in range(int(rng.integers(0, 3)))]
        rng.shuffle(cands)
        q = f.ask()
        return dict(kind="short", cands=cands, turns=[], q=pick(rng, SHORT).format(q=q, ql=lower_first(q)), target=short_value(f))
    if kind == "yesno":
        f = G.Fact(rng)
        while f.kind == "t" and f.field in ("when", "deadline"):
            f = G.Fact(rng)
        true = f.stored()[1]
        cands = [f.stored()] + [G._other(rng, {f.ent["id"]}).stored() for _ in range(int(rng.integers(0, 2)))]
        rng.shuffle(cands)
        if rng.random() < 0.5:
            claim, target = true, "Yes, " + lower_first(your(true))
        else:
            wrong = f.spec[6](rng)
            while wrong == f.value:
                wrong = f.spec[6](rng)
            claim, target = f._f(f.spec[1], wrong), "No, " + lower_first(your(true))
        claim = lower_first(claim.rstrip("."))          # in the user's own voice: "my sister Kim lives in Lyon"
        return dict(kind="yesno", cands=cands, turns=[], q=pick(rng, YESNO).format(s=claim), target=target)
    if kind == "list":
        ent = G.person(rng)
        fields = [k for k in G.P if k in FIELD_NAMES and G.P[k][1].startswith(("{E}", "{Ep}"))]
        f1, f2 = [str(x) for x in rng.choice(fields, 2, replace=False)]
        a, b = G.Fact(rng, "p", f1, dict(ent)), G.Fact(rng, "p", f2, dict(ent))
        cands = [a.stored(), b.stored()] + [G._other(rng, {ent["id"]}).stored() for _ in range(int(rng.integers(0, 2)))]
        rng.shuffle(cands)
        q = pick(rng, LIST).format(ep=ent["ep"], f1=FIELD_NAMES[f1], f2=FIELD_NAMES[f2])
        target = f"- {FIELD_NAMES[f1][0].upper() + FIELD_NAMES[f1][1:]}: {short_value(a)[:-1]}\n- {FIELD_NAMES[f2][0].upper() + FIELD_NAMES[f2][1:]}: {short_value(b)[:-1]}"
        return dict(kind="list", cands=cands, turns=[], q=q, target=target)
    w = G3.make(rng, alpaca)
    if w["kind"] not in ("clarify", "unknown"):
        w["target"] = your(w["target"])
    return w
