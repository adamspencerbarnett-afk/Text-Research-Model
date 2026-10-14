"""Synthetic skill windows for training (14 October 2026). Invented facts about invented people, events and things;
no name or distinctive value of the skills test (testset.TEST_NAMES) is ever used, so the test measures the skill,
not memorised facts.

make(rng, alpaca) returns {kind, cands: [(q, a)] oldest first, turns: [(q, a)], q, target} for one window:
  single    the fact among 0-2 unrelated ones, asked in other words            -> its answer
  nearmiss  the fact beside 1-2 that differ in one name or one field           -> its answer
  update    an older and a newer value (newer stored later)                     -> the newer, or the older when asked
  unknown   facts about other people / other fields, or unrelated exchanges     -> "I don't have that in my memory."
  clarify   two facts that fit an under-specified question equally              -> "Do you mean X or Y?"
  followup  the clarifying turn in the window, the user names one               -> that one's answer
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import testset  # noqa: E402

BAN = testset.TEST_NAMES
_ok = lambda xs: [x for x in xs if not any(w.lower().strip(".,'") in BAN for w in x.split())]

NAMES = _ok("""Kim Ivan Sofia Mateo Ingrid Rahul Yusuf Chloe Bruno Talia Kenji Amara Lucas Freya Dmitri Leila Hugo Maya Oscar
Imani Pablo Sienna Tariq Wren Felipe Ayla Jonah Esme Kofi Lara Nico Isla Rohan Vera Silas Noor Emil Gemma Ravi_ Callum Ines Malik
Petra Quinn Aditi Bram Celia Dario Elif Farah Gideon Hazel Idris Juno Kai Lucia Mateus Nadia_ Odette Pia Ronan Saskia Tobias Uma
Viktor Willa Xavier Yara Zane Aurora Basil Cyrus Daphne Ezra Flora Gabe Hector Iris Jasper Keira Lionel Magnus Nell Orla Pierce
Rhea Stellan Thea Ulric Vida Wesley Ximena Yosef Zara Alba Bea Cato Dov Edie Fenn Gia Hal Ivy Jude Kit Lux Milo Nia Otto Pip""".replace("_", "x").split())
RELS = ["sister", "brother", "cousin", "aunt", "uncle", "boss", "neighbour", "friend", "coworker", "roommate", "son", "daughter",
        "grandmother", "grandfather", "niece", "nephew", "partner", "trainer", "tutor", "colleague"]
CITIES = _ok("""Oslo Lyon Leeds Quito Hobart Kraków Porto Cork Bergen Halifax Dayton Fresno Tulsa Boise Reno Omaha Savannah Raleigh
Toledo Spokane Duluth Hamburg Turin Seville Ghent Aarhus Tampere Gdansk Bilbao Valencia Nantes Graz Brno Malmo Perth Adelaide Wellington
Nagoya Busan Pune Mysore Cusco Rosario Medellin Kingston Halifax Calgary Regina Moncton""".split())
JOBS = ["nurse", "pilot", "plumber", "teacher", "architect", "chef", "pharmacist", "electrician", "librarian", "lawyer", "baker",
        "journalist", "firefighter", "vet", "accountant", "carpenter", "translator", "paramedic", "engineer", "florist", "barber",
        "physiotherapist", "data analyst", "graphic designer", "tour guide", "zookeeper", "welder", "midwife", "surveyor", "optician"]
FOODS = ["pad thai", "lasagne", "sushi", "paella", "ramen", "falafel", "goulash", "pierogi", "risotto", "tacos al pastor", "pho",
         "shepherd's pie", "biryani", "moussaka", "dumplings", "fish and chips", "enchiladas", "gnocchi", "bibimbap", "jambalaya"]
COLOURS = ["purple", "teal", "maroon", "navy", "lime", "coral", "beige", "crimson", "turquoise", "lavender", "mustard", "olive", "pink", "grey"]
SPORTS = ["tennis", "cricket", "hockey", "volleyball", "badminton", "golf", "baseball", "swimming", "cycling", "fencing", "rowing",
          "climbing", "handball", "table tennis", "lacrosse", "squash"]
BANDS = ["Radiohead", "ABBA", "Queen", "Muse", "Blondie", "Oasis", "Daft Punk", "Fleetwood Mac", "The Cure", "Pixies", "Keane",
         "Gorillaz", "Metallica", "Blur", "Pulp", "Weezer", "Journey", "Toto", "Train", "Garbage"]
DRINKS = ["espresso", "lemonade", "mint tea", "orange juice", "hot chocolate", "iced coffee", "ginger ale", "chai", "sparkling water", "cola"]
CARS = ["Toyota", "Ford", "Kia", "Nissan", "Audi", "Fiat", "Peugeot", "Hyundai", "Tesla", "Jeep", "Skoda", "Mini", "Lexus", "Renault"]
PETS = ["dog", "cat", "rabbit", "parrot", "hamster", "goldfish", "tortoise", "ferret"]
PETNAMES = _ok("""Rex Luna Mochi Ziggy Nala Waffles Bean Olive Ginger Scout Pickles Toffee Banjo Juniper Nugget Peanut_ Sprout Maple_
Bandit Clover Dot Fudge Gizmo Hazel_ Inky Jinx Koko Loki Muffin Noodle Oreo Pumpkin Rolo Sushi Tango Ursa Velvet Whiskers""".replace("_", "y").split())
HOBBIES = ["pottery", "birdwatching", "chess", "knitting", "salsa dancing", "rock climbing", "baking bread", "photography",
           "gardening", "woodworking", "kayaking", "painting", "juggling", "stargazing", "fishing", "origami"]
ALLERGIES = ["shellfish", "cats", "pollen", "dairy", "eggs", "wheat", "bee stings", "penicillin", "soy", "sesame", "dust", "kiwi"]
COMPANIES = ["the post office", "the hospital", "a bike shop", "the town hall", "a law firm", "the airport", "a software company",
             "the bakery on Elm Road", "the museum", "a garden centre", "the train station", "a bookshop", "the city zoo", "a dental clinic"]
SURNAMES = _ok("""Novak Ferreira Lindqvist Achebe Kowalski Brennan Takeda Moreau Haddad Petrov Silva Okoro Jansen Romano Dubois
Nakamura Gallo Abara Holm Rossi Varga Eriksen Costa Ibarra Kaplan Laine Mendez Nolan Osei Quist Reyes Sato Tanaka Ueda""".split())
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
EVENTS = ["staff meeting", "book group", "soccer practice", "piano lesson", "team dinner", "parent evening", "board meeting",
          "dance class", "study group", "vet appointment", "car service", "chess club", "choir practice", "budget review", "swim lesson"]
QUALS = [("Monday", "Friday"), ("morning", "evening"), ("first", "second"), ("junior", "senior"), ("weekly", "monthly"),
         ("north", "south"), ("early", "late"), ("small", "big"), ("indoor", "outdoor"), ("beginner", "advanced")]
THINGS = ["garage", "alarm", "safe", "storage unit", "bike lock", "side door", "mailbox", "shed", "printer room", "staff room"]
ITEMS = ["monthly pass", "car wash", "haircut", "yearly plan", "music lesson", "taxi to the airport", "parking permit", "gym membership"]
PLACES_OPEN = ["pool", "post office", "corner shop", "bakery", "laundromat", "bike shop", "library", "climbing gym"]
ROOMS = ["room {n}", "the {c} room", "the hall on {s} Street", "the cafe on {s} Street", "the community centre", "the back office"]
STREETS = ["Oak", "Mill", "Bridge", "Station", "Church", "Elmwood", "Hill", "Market", "Queen", "Water"]
ABSTAIN = ["I don't have that in my memory.", "I don't have that in my memory. Could you tell me?"]


def _art(w):
    return "an" if w[0].lower() in "aeiou" else "a"


def person(rng, named=None):
    """(question form, sentence-start form, possessive) for a person, e.g. ('my sister Kim', 'My sister Kim', "my sister Kim's")."""
    n = named or NAMES[rng.integers(len(NAMES))]
    form = rng.integers(3)
    if form == 0:
        e = n
    elif form == 1 and named is None:
        e = f"my {RELS[rng.integers(len(RELS))]}"
    else:
        e = f"my {RELS[rng.integers(len(RELS))]} {n}"
    return {"e": e, "E": e[0].upper() + e[1:], "ep": e + "'s", "Ep": (e + "'s")[0].upper() + (e + "'s")[1:], "id": e}


def colourword(rng):
    return ("colour", "favourite") if rng.random() < 0.5 else ("color", "favorite")


def pick(rng, xs):
    return xs[rng.integers(len(xs))]


def time_(rng):
    h = int(rng.integers(6, 21)); m = pick(rng, ["00", "15", "30", "45"])
    return f"{h % 12 or 12}:{m} {'am' if h < 12 else 'pm'}" if m != "00" else f"{h % 12 or 12} {'am' if h < 12 else 'pm'}"


def phone(rng):
    return f"555-{int(rng.integers(200, 999)):04d}"


def room(rng):
    return pick(rng, ROOMS).format(n=int(rng.integers(2, 60)), c=pick(rng, COLOURS), s=pick(rng, STREETS))


# person fields: (stored question, answer, asks, update question, update answer, earlier-value asks, value maker)
P = {
    "city": ("Where does {e} live?", "{E} lives in {v}.", ["Which city does {e} live in?", "What town is {e} in?", "Where is {e} living these days?", "Where's {e} based?"],
             ["Where does {e} live now?", "Where did {e} move to?"], ["{E} moved to {v}.", "{E} now lives in {v}."],
             ["Where did {e} live before?", "Which city did {e} live in before moving?"], lambda r: pick(r, CITIES)),
    "job": ("What is {ep} job?", "{E} works as {art} {v}.", ["What does {e} do for a living?", "What is {ep} profession?", "What kind of work does {e} do?"],
            ["What is {ep} new job?", "Did {e} change jobs?"], ["{E} now works as {art} {v}.", "{E} changed jobs and is now {art} {v}."],
            ["What did {e} do before?", "What was {ep} old job?"], lambda r: pick(r, JOBS)),
    "food": ("What is {ep} {fav} food?", "{Ep} {fav} food is {v}.", ["What food does {e} like best?", "What does {e} love to eat?", "Which dish does {e} like most?"],
             ["What is {ep} {fav} food now?"], ["{Ep} {fav} food is now {v}."], ["What was {ep} {fav} food before?"], lambda r: pick(r, FOODS)),
    "colour": ("What is {ep} {fav} {col}?", "{Ep} {fav} {col} is {v}.", ["Which {col} does {e} like best?", "What {col} does {e} like most?", "What is {e}'s {fav} {col}?"],
               ["What is {ep} {fav} {col} now?"], ["{Ep} {fav} {col} is now {v}."], ["What {col} did {e} like before?"], lambda r: pick(r, COLOURS)),
    "sport": ("What is {ep} {fav} sport?", "{Ep} {fav} sport is {v}.", ["Which sport does {e} like best?", "What sport does {e} enjoy most?"],
              ["What sport does {e} play now?"], ["{E} switched to {v}."], ["What sport did {e} play before?"], lambda r: pick(r, SPORTS)),
    "band": ("What is {ep} {fav} band?", "{Ep} {fav} band is {v}.", ["Which band does {e} like best?", "Who does {e} listen to most?"],
             ["What does {e} listen to now?"], ["{Ep} {fav} band is now {v}."], ["Which band did {e} like before?"], lambda r: pick(r, BANDS)),
    "drink": ("What is {ep} {fav} drink?", "{Ep} {fav} drink is {v}.", ["What does {e} like to drink most?", "Which drink does {e} prefer?"],
              ["What does {e} drink now?"], ["{E} now prefers {v}."], ["What did {e} use to drink?"], lambda r: pick(r, DRINKS)),
    "car": ("What car does {e} drive?", "{E} drives {art} {v}.", ["Which car does {e} have?", "What does {e} drive?", "What kind of car is {ep}?"],
            ["What car does {e} drive now?", "Did {e} get a new car?"], ["{E} now drives {art} {v}.", "{E} sold the old car and bought {art} {v}."],
            ["What car did {e} have before?"], lambda r: f"{pick(r, COLOURS)} {pick(r, CARS)}"),
    "phone": ("What is {ep} phone number?", "{Ep} phone number is {v}.", ["How can I call {e}?", "What's {ep} number?", "Can you give me {ep} phone number?"],
              ["What is {ep} new number?", "Did {e} change numbers?"], ["{Ep} new phone number is {v}.", "{E} changed numbers to {v}."],
              ["What was {ep} old number?"], phone),
    "birthday": ("When is {ep} birthday?", "{Ep} birthday is on {v}.", ["What date is {ep} birthday?", "When was {e} born?", "When should I wish {e} a happy birthday?"],
                 None, None, None, lambda r: f"{pick(r, MONTHS)} {int(r.integers(1, 29))}"),
    "pet": ("What is the name of {ep} {pet}?", "{Ep} {pet} is called {v}.", ["What's {ep} {pet} called?", "What did {e} name the {pet}?", "What name does {ep} {pet} have?"],
            None, None, None, lambda r: pick(r, PETNAMES)),
    "age": ("How old is {e}?", "{E} is {v} years old.", ["What age is {e}?", "How many years old is {e}?"],
            ["How old is {e} now?"], ["{E} just turned {v}."], None, lambda r: str(int(r.integers(3, 90)))),
    "allergy": ("What is {e} allergic to?", "{E} is allergic to {v}.", ["Does {e} have any allergies?", "What allergy does {e} have?"],
                None, None, None, lambda r: pick(r, ALLERGIES)),
    "hobby": ("What is {ep} hobby?", "{Ep} hobby is {v}.", ["What does {e} do for fun?", "What does {e} like doing in their free time?"],
              ["What is {ep} hobby now?"], ["{Ep} new hobby is {v}."], ["What hobby did {e} have before?"], lambda r: pick(r, HOBBIES)),
    "work": ("Where does {e} work?", "{E} works at {v}.", ["Which place does {e} work for?", "Where is {ep} workplace?"],
             ["Where does {e} work now?"], ["{E} now works at {v}."], ["Where did {e} work before?"], lambda r: pick(r, COMPANIES)),
    "surname": ("What is {ep} last name?", "{Ep} last name is {v}.", ["What's {ep} surname?", "What is {ep} family name?"],
                ["What is {ep} last name now?"], ["{Ep} last name is now {v}."], ["What was {ep} last name before?"], lambda r: pick(r, SURNAMES)),
}
# thing fields: the entity is "the <thing>"
TH = {
    "when": (EVENTS, "When is the {t}?", "The {t} is on {d} at {v}.", ["What day is the {t}?", "When do we have the {t}?", "What time is the {t}?"],
             ["Has the {t} moved?", "When is the {t} now?"], ["The {t} moved to {d} at {v}.", "The {t} is now on {d} at {v}."],
             ["When was the {t} before it moved?"], time_),
    "where": (EVENTS, "Where is the {t}?", "The {t} is in {v}.", ["Where do we meet for the {t}?", "Where is the {t} held?", "Which room is the {t} in?"],
              ["Where is the {t} now?", "Did the {t} change rooms?"], ["The {t} moved to {v}.", "The {t} is now in {v}."],
              ["Where was the {t} before?"], room),
    "code": (THINGS, "What is the code for the {t}?", "The {t} code is {v}.", ["How do I open the {t}?", "What's the {t} code?", "Can you remind me of the {t} code?"],
             ["What is the new code for the {t}?", "Was the {t} code changed?"], ["The {t} code was changed to {v}.", "The new {t} code is {v}."],
             ["What was the old {t} code?"], lambda r: str(int(r.integers(1000, 9999)))),
    "price": (ITEMS, "How much is the {t}?", "The {t} costs {v} dollars.", ["What does the {t} cost?", "What's the price of the {t}?"],
              ["How much is the {t} now?", "Did the {t} get more expensive?"], ["The {t} now costs {v} dollars.", "The price of the {t} went up to {v} dollars."],
              ["How much did the {t} cost before?"], lambda r: str(int(r.integers(5, 120)))),
    "opens": (PLACES_OPEN, "What time does the {t} open?", "The {t} opens at {v}.", ["When does the {t} open?", "How early is the {t} open?"],
              ["What time does the {t} open now?"], ["The {t} now opens at {v}."], ["What time did the {t} use to open?"], time_),
}


class Fact:
    """One invented fact: an entity, a field and a value; renders its stored exchange, asks, updates."""

    def __init__(self, rng, kind=None, field=None, ent=None, value=None, extra=None):
        self.rng = rng
        self.kind = kind or ("p" if rng.random() < 0.65 else "t")
        if self.kind == "p":
            self.field = field or pick(rng, list(P))
            self.ent = ent or person(rng)
            spec = P[self.field]
            fav, col = (extra or {}).get("fav"), (extra or {}).get("col")
            if fav is None:
                col, fav = colourword(rng)
            self.fmt = dict(self.ent, fav=fav, col=col, pet=(extra or {}).get("pet") or pick(rng, PETS))
        else:
            self.field = field or pick(rng, list(TH))
            spec = TH[self.field]
            self.ent = ent or {"t": pick(rng, spec[0]), "id": None}
            self.ent["id"] = self.ent["t"]
            self.fmt = dict(self.ent, d=pick(rng, DAYS))
        self.spec = spec if self.kind == "p" else spec[1:]
        self.value = value or self.spec[6](rng)

    def _f(self, s, v=None):
        v = self.value if v is None else v
        return s.format(**self.fmt, v=v, art=_art(v))

    def stored(self):
        return (self._f(self.spec[0]), self._f(self.spec[1]))

    def ask(self):
        return self._f(pick(self.rng, self.spec[2])) if self.rng.random() < 0.75 else self._f(self.spec[0])

    def can_update(self):
        return self.spec[3] is not None

    def update(self):
        """(new stored exchange, its answer, an earlier-value ask or None)."""
        nv = self.spec[6](self.rng)
        while nv == self.value:
            nv = self.spec[6](self.rng)
        if self.kind == "t":
            self.fmt["d"] = pick(self.rng, DAYS)
        q = self._f(pick(self.rng, self.spec[3]), nv); a = self._f(pick(self.rng, self.spec[4]), nv)
        prev = self._f(pick(self.rng, self.spec[5])) if self.spec[5] else None
        return (q, a), prev


def _other(rng, avoid_ids, avoid_fields=()):
    for _ in range(50):
        f = Fact(rng)
        if f.ent["id"] not in avoid_ids and f.field not in avoid_fields:
            return f
    return Fact(rng)


def make(rng, alpaca=None):
    kind = pick(rng, ["single"] * 2 + ["nearmiss"] * 3 + ["update"] * 3 + ["unknown"] * 2 + ["clarify"] * 2)
    f = Fact(rng)
    if kind == "single":
        cands = [f.stored()] + [_other(rng, {f.ent["id"]}).stored() for _ in range(int(rng.integers(0, 3)))]
        rng.shuffle(cands)
        return dict(kind=kind, cands=cands, turns=[], q=f.ask(), target=f.stored()[1])
    if kind == "nearmiss":
        others = []
        for _ in range(int(rng.integers(1, 3))):
            if f.kind == "p" and rng.random() < 0.5:       # same field, another person
                g = Fact(rng, "p", f.field, person(rng), extra={"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]})
            elif f.kind == "p":                              # same person, another field
                g = Fact(rng, "p", pick(rng, [k for k in P if k != f.field]), dict(f.ent))
            else:                                            # same kind of thing, another one
                g = Fact(rng, "t", f.field, {"t": pick(rng, [t for t in TH[f.field][0] if t != f.ent["t"]])})
            others.append(g.stored())
        cands = [f.stored()] + others
        rng.shuffle(cands)
        return dict(kind=kind, cands=cands, turns=[], q=f.ask(), target=f.stored()[1])
    if kind == "update":
        while not f.can_update():
            f = Fact(rng)
        old = f.stored(); ask_now = f.ask()
        new, prev = f.update()
        cands = [old, new]
        if rng.random() < 0.4:
            cands.insert(int(rng.integers(0, 3)), _other(rng, {f.ent["id"]}).stored())
        if prev and rng.random() < 0.25:
            return dict(kind="update_earlier", cands=cands, turns=[], q=prev, target=old[1])
        return dict(kind=kind, cands=cands, turns=[], q=ask_now, target=new[1])
    if kind == "unknown":
        r = rng.random()
        if r < 0.35 and f.kind == "p":       # the person is stored, but not this field
            cands = [Fact(rng, "p", g, dict(f.ent)).stored() for g in rng.choice([k for k in P if k != f.field], int(rng.integers(1, 3)), replace=False)]
        elif r < 0.7:                          # this field is stored, for other people or things
            cands = [_other(rng, {f.ent["id"]}).stored() for _ in range(int(rng.integers(1, 4)))]
            if f.kind == "p":
                cands[0] = Fact(rng, "p", f.field, person(rng), extra={"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]}).stored()
        elif alpaca is not None:               # only unrelated general exchanges came back
            cands = [alpaca(rng) for _ in range(int(rng.integers(1, 4)))]
        else:
            cands = []
        rng.shuffle(cands)
        return dict(kind=kind, cands=cands, turns=[], q=f.ask(), target=pick(rng, ABSTAIN))
    # clarify / followup: two facts that the question cannot tell apart
    if f.kind == "p":
        rel = pick(rng, RELS)
        n1, n2 = rng.choice(NAMES, 2, replace=False)
        ex = {"fav": f.fmt["fav"], "col": f.fmt["col"], "pet": f.fmt["pet"]}
        mk = lambda n: {"e": f"my {rel} {n}", "E": f"My {rel} {n}", "ep": f"my {rel} {n}'s", "Ep": f"My {rel} {n}'s", "id": n}
        a1, a2 = Fact(rng, "p", f.field, mk(n1), extra=ex), Fact(rng, "p", f.field, mk(n2), extra=ex)
        gen = Fact(rng, "p", f.field, {"e": f"my {rel}", "E": f"My {rel}", "ep": f"my {rel}'s", "Ep": f"My {rel}'s", "id": rel}, extra=ex)
        labels = (str(n1), str(n2)); clar = f"Do you mean {n1} or {n2}?"
        picks = [lambda l: f"{l}.", lambda l: f"I mean {l}.", lambda l: f"{l}, please."]
    else:
        t = f.ent["t"]; q1, q2 = pick(rng, QUALS)
        a1 = Fact(rng, "t", f.field, {"t": f"{q1} {t}"}); a2 = Fact(rng, "t", f.field, {"t": f"{q2} {t}"})
        gen = Fact(rng, "t", f.field, {"t": t})
        labels = (q1, q2); clar = f"Do you mean the {q1} {t} or the {q2} {t}?"
        picks = [lambda l: f"The {l} one.", lambda l: f"{l[0].upper() + l[1:]}.", lambda l: f"I mean the {l} {t}."]
    cands = [a1.stored(), a2.stored()]
    ask = gen.ask()
    if rng.random() < 0.5:
        return dict(kind="clarify", cands=cands, turns=[], q=ask, target=clar)
    j = int(rng.integers(2))
    return dict(kind="followup", cands=cands, turns=[(ask, clar)], q=pick(rng, picks)(labels[j]), target=cands[j][1])


if __name__ == "__main__":
    import numpy as np
    rng = np.random.default_rng(0)
    for _ in range(14):
        w = make(rng)
        print(w["kind"], "|", " || ".join(f"{q} -> {a}" for q, a in w["cands"]), "|", w["turns"], "| Q:", w["q"], "| T:", w["target"])
