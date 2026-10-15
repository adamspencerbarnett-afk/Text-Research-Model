"""Skills generator version 3 (14 October 2026, for the 100,000-example skills-arm dataset). Extends gen_skills:
more fields, many more ways of asking, casual wrappers ("Remind me, ..."), facts that combine two details, typos, and
a reserved split (one name in five, one way of asking per field) that only the held-out test v2 uses, so v2 measures
people and phrasings the training never saw.

    import gen_skills_v3 as G3
    G3.setup(test=False)          # training: reserved names and asks excluded
    w = G3.make(rng, alpaca)      # same window format as gen_skills.make
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen_skills as G  # noqa: E402

pick, person, Fact = G.pick, G.person, G.Fact

MORE_ASKS = {
    "city": ["What city is {e} living in?", "Where does {e} live these days?", "Do you know where {e} lives?", "In which city does {e} live?",
             "Which place does {e} call home?"],
    "job": ["What's {ep} line of work?", "What job does {e} have?", "What does {e} do?", "How does {e} earn a living?", "What is {e} working as?"],
    "food": ["What's {ep} {fav} meal?", "What dish does {e} enjoy most?", "Which food does {e} love?", "What would {e} order for dinner?"],
    "colour": ["What's {ep} {fav} {col}?", "Which {col} is {ep} {fav}?", "What {col} does {e} prefer?"],
    "sport": ["Which sport is {ep} {fav}?", "What sport does {e} love?", "What does {e} like to play?"],
    "band": ["What music does {e} like?", "Which group does {e} listen to?", "What's {ep} {fav} band?"],
    "drink": ["What's {ep} usual drink?", "What does {e} usually order to drink?", "Which drink does {e} like best?"],
    "car": ["What's {ep} car?", "Which car is {ep}?", "What vehicle does {e} drive?", "What kind of car does {e} own?"],
    "phone": ["What number do I call for {e}?", "What's the best number for {e}?", "How do I reach {e} by phone?", "Do you have {ep} number?"],
    "birthday": ["What day is {ep} birthday?", "When does {e} celebrate a birthday?", "On what date was {e} born?"],
    "pet": ["What is {ep} {pet} named?", "What do they call {ep} {pet}?", "Do you know the name of {ep} {pet}?"],
    "age": ["How old is {e} now?", "What is {ep} age?", "Do you know how old {e} is?"],
    "allergy": ["Is {e} allergic to anything?", "What can't {e} eat or touch?", "Does {e} have an allergy?"],
    "hobby": ["What is {e} into?", "How does {e} spend free time?", "What's {ep} favourite pastime?"],
    "work": ["Where is {e} employed?", "What company does {e} work at?", "Where does {e} have a job?"],
    "surname": ["What surname does {e} have?", "What's {ep} last name?", "What family name does {e} have?"],
    "when": ["When does the {t} happen?", "What day and time is the {t}?", "When is the {t} scheduled?", "At what time is the {t}?"],
    "where": ["Where does the {t} take place?", "Where is the {t} happening?", "In which room is the {t}?"],
    "code": ["What's the code to the {t}?", "What do I enter to open the {t}?", "What is the {t} combination?"],
    "price": ["What's the cost of the {t}?", "How much do I pay for the {t}?", "How much does the {t} cost?"],
    "opens": ["What time is the {t} open from?", "At what time does the {t} open?", "When can I get into the {t}?"],
}
BOOKS = ["Dune", "Emma", "Beloved", "Ulysses", "Middlemarch", "The Hobbit", "Rebecca", "Persuasion", "Kindred", "Stoner", "Matilda", "Watership Down"]
SCHOOLS = ["Maple Grove School", "Riverside Academy", "Saint Clare Primary", "Hillcrest High", "Oakfield College", "Northgate School", "Brookside Academy"]
LANGS = ["Portuguese", "Swahili", "Korean", "Dutch", "Polish", "Turkish", "Greek", "Hindi", "Finnish", "Tagalog", "Arabic", "Czech"]
INSTRUMENTS = ["cello", "trumpet", "ukulele", "drums", "violin", "flute", "saxophone", "banjo", "harp", "clarinet"]
ROADS = ["Road", "Lane", "Avenue", "Close", "Drive", "Way", "Crescent", "Terrace"]
PLACES = ["office", "cabin", "studio", "library", "clinic", "warehouse", "lake house", "workshop"]
P3 = {
    "book": ("What is {ep} {fav} book?", "{Ep} {fav} book is {v}.", ["Which book does {e} love most?", "What book does {e} like best?", "What's {ep} {fav} novel?"],
             ["What is {ep} {fav} book now?"], ["{Ep} {fav} book is now {v}."], ["Which book did {e} like before?"], lambda r: pick(r, BOOKS)),
    "school": ("Which school does {e} go to?", "{E} goes to {v}.", ["Where does {e} go to school?", "What school is {e} at?", "Which school is {e} in?"],
               ["Which school does {e} go to now?", "Did {e} change schools?"], ["{E} moved to {v}.", "{E} now goes to {v}."],
               ["Which school did {e} go to before?"], lambda r: pick(r, SCHOOLS)),
    "language": ("What language does {e} speak?", "{E} speaks {v}.", ["Which language does {e} know?", "What language is {e} fluent in?", "Does {e} speak another language?"],
                 ["What language is {e} learning now?"], ["{E} is now learning {v}."], None, lambda r: pick(r, LANGS)),
    "instrument": ("What instrument does {e} play?", "{E} plays the {v}.", ["Which instrument does {e} play?", "Does {e} play an instrument?", "What does {e} play in the band?"],
                   ["What instrument does {e} play now?"], ["{E} switched to the {v}."], ["What instrument did {e} play before?"], lambda r: pick(r, INSTRUMENTS)),
    "address": ("What is {ep} address?", "{Ep} address is {v}.", ["Where does {e} live exactly?", "What's {ep} home address?", "Which street does {e} live on?"],
                ["What is {ep} new address?"], ["{Ep} new address is {v}."], ["What was {ep} old address?"],
                lambda r: f"{int(r.integers(2, 300))} {pick(r, G.STREETS)} {pick(r, ROADS)}"),
    "email": ("What is {ep} email address?", "{Ep} email is {v}.", ["How do I email {e}?", "What email does {e} use?", "What's {ep} email?"],
              ["What is {ep} new email?"], ["{Ep} new email is {v}."], ["What was {ep} old email?"],
              lambda r: f"{pick(r, G.NAMES).lower()}.{pick(r, G.SURNAMES).lower()}@{pick(r, ['mailbox', 'postbox', 'inbox'])}.com"),
}
TH3 = {
    "wifi": (PLACES, "What is the wifi password at the {t}?", "The wifi password at the {t} is {v}.",
             ["What's the internet password at the {t}?", "How do I get online at the {t}?", "What is the {t} wifi password?"],
             ["What is the new wifi password at the {t}?"], ["The wifi password at the {t} was changed to {v}."], ["What was the old wifi password at the {t}?"],
             lambda r: f"{pick(r, G.PETNAMES).lower()}{int(r.integers(10, 99))}"),
    "parking": (PLACES, "Where do I park at the {t}?", "Parking for the {t} is in lot {v}.",
                ["Which lot do I use at the {t}?", "Where is parking for the {t}?", "Where can I leave the car at the {t}?"],
                ["Where do I park at the {t} now?"], ["Parking for the {t} moved to lot {v}."], ["Where did we park at the {t} before?"],
                lambda r: pick(r, list("ABCDEFGH")) + str(int(r.integers(1, 9)))),
    "deadline": (["budget report", "funding proposal", "website update", "safety audit", "sales plan", "tax return", "photo album", "lab report"],
                 "When is the {t} due?", "The {t} is due on {d}.", ["What is the deadline for the {t}?", "By when do I need the {t}?", "When do I have to hand in the {t}?"],
                 ["Was the {t} deadline moved?", "When is the {t} due now?"], ["The {t} deadline moved to {d}.", "The {t} is now due on {d}."],
                 ["When was the {t} due originally?"], lambda r: pick(r, G.DAYS)),
}
WRAP = ["{q}", "{q}", "{q}", "{q}", "Quick question: {q}", "Remind me, {ql}", "Hey, {ql}", "I forgot. {q}", "Sorry, {ql}",
        "{q} I can't remember.", "Do you remember? {q}", "Can you help? {q}", "One more thing: {ql}"]
NAMES_ALL = list(G.NAMES)
RESERVED = set(NAMES_ALL[::5])
_on = {"v3": False}


def setup(test: bool = False):
    """Training (test=False): reserved names and each field's last way of asking are never used. Test v2 (test=True):
    only reserved names and only the reserved way of asking."""
    G.P.update(P3); G.TH.update(TH3)
    for k, extra in MORE_ASKS.items():
        spec = G.P.get(k) or G.TH.get(k)
        i = 2 if k in G.P else 3
        for x in extra:
            if x not in spec[i]:
                spec[i].append(x)
    for d, i in [(G.P, 2), (G.TH, 3)]:
        for k, spec in d.items():
            full = BANK.setdefault(k, list(spec[i]))
            spec[i][:] = [full[-1]] if test else full[:-1]
    G.NAMES[:] = [n for n in NAMES_ALL if (n in RESERVED) == test]
    G.MIX[:] = ["single"] * 4 + ["nearmiss"] * 3 + ["update"] * 3 + ["unknown"] * 2 + ["clarify"] * 4
    _on["v3"] = True


BANK: dict = {}


def wrap(rng, q: str) -> str:
    w = pick(rng, WRAP)
    return w.format(q=q, ql=q[0].lower() + q[1:])


def combo(rng):
    """A stored answer that carries two details of one person, asked about one of them."""
    ent = person(rng)
    fields = [k for k in G.P if G.P[k][1].startswith("{E}")]
    f1, f2 = rng.choice(fields, 2, replace=False)
    a, b = Fact(rng, "p", str(f1), dict(ent)), Fact(rng, "p", str(f2), dict(ent))
    sa, sb = a.stored()[1], b.stored()[1]
    ans = sa[:-1] + ", and " + sb[0].lower() + sb[1:]
    return (f"What do you know about {ent['e']}?", ans), pick(rng, [a, b]).ask(), ans


def make(rng, alpaca=None):
    """A version 3 window: gen_skills.make plus a combined-fact kind, casual wrappers on the asked question."""
    if rng.random() < 0.08:
        stored, ask, ans = combo(rng)
        cands = [stored] + [G._other(rng, set()).stored() for _ in range(int(rng.integers(0, 2)))]
        rng.shuffle(cands)
        w = dict(kind="combo", cands=cands, turns=[], q=ask, target=ans)
    else:
        w = G.make(rng, alpaca)
    if w["kind"] != "followup":
        w["q"] = wrap(rng, w["q"])
    return w
