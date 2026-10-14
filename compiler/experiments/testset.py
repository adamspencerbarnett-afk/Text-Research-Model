"""The skills test: 100 questions no model has seen (14 October 2026). Every fact is invented, so the answers
can only come from memory, and the right reply to an unstored question is to say so.

Groups (keyword-graded; any keyword of each group, every group needed):
  taught     20  one stored fact, asked in other words
  update     20  an older and a newer fact stored (in that order); 15 ask the current value, 5 the earlier one
  nearmiss   20  two facts that differ in one word (Anna / Anne, front / back gate); each asked once
  unknown    20  nothing stored; the right reply says it does not know (several ask about people who ARE stored,
                 but about something else)
  clarify    10  two facts fit the question equally (my brother Sam / my brother Eli); the right reply asks which
  followup   10  the user then names one; the right reply answers for that one
"""

ABSTAIN = ["don't have", "do not have", "don't know", "do not know", "not sure", "no information", "not in my memory",
           "haven't told", "have not told", "could you tell me", "can you tell me", "i'm not aware", "no record"]

TAUGHT = [
    ("What is my cousin Lena's job?", "My cousin Lena works as a dentist.", "What does Lena do for a living?", [["dentist"]]),
    ("What is the wifi password at the cabin?", "The wifi password at the cabin is maple42.", "What's the password for the internet at the cabin?", [["maple42"]]),
    ("Where did I park my car today?", "You parked on level 3 of the Elm Street garage.", "Which level of the garage is my car on?", [["3", "three"]]),
    ("What is my landlord's name?", "Your landlord is called Mr. Okafor.", "Who is my landlord?", [["okafor"]]),
    ("What time does the pharmacy close?", "The pharmacy closes at 7 pm.", "Until when is the pharmacy open?", [["7"]]),
    ("What is the name of Jenna's cat?", "Jenna's cat is called Pepper.", "What did Jenna name her cat?", [["pepper"]]),
    ("How old is my nephew Arlo?", "Arlo is six years old.", "What age is Arlo?", [["six", "6"]]),
    ("Which airline is my flight to Denver with?", "Your flight to Denver is with Frontier.", "What airline am I flying to Denver on?", [["frontier"]]),
    ("What is my locker number at the gym?", "Your gym locker number is 218.", "Which locker is mine at the gym?", [["218"]]),
    ("What is Marco's favourite sport?", "Marco's favourite sport is rugby.", "Which sport does Marco like best?", [["rugby"]]),
    ("When is my eye doctor appointment?", "Your eye doctor appointment is on the 14th at 3:30 pm.", "What day do I see the eye doctor?", [["14"]]),
    ("What is the name of our project server?", "The project server is called Atlas-7.", "What do we call the project server?", [["atlas"]]),
    ("Where does my aunt Rosa live?", "Aunt Rosa lives in Tucson.", "Which city is Aunt Rosa in?", [["tucson"]]),
    ("What allergy does Theo have?", "Theo is allergic to peanuts.", "Is Theo allergic to anything?", [["peanut"]]),
    ("What is the code for the office door?", "The office door code is 5590.", "How do I open the office door?", [["5590"]]),
    ("What colour is Hana's bike?", "Hana's bike is orange.", "What colour bike does Hana ride?", [["orange"]]),
    ("Who is coaching the football team this year?", "Coach Delgado is coaching the football team this year.", "Who's the football coach now?", [["delgado"]]),
    ("What is my insurance policy number?", "Your insurance policy number is K-48213.", "Can you remind me of my policy number?", [["48213"]]),
    ("Which room is the book club in?", "The book club meets in room 9 of the library.", "Where does the book club meet?", [["9", "nine"]]),
    ("What is Gus's middle name?", "Gus's middle name is Everett.", "What's the middle name of Gus?", [["everett"]]),
]

# (older fact, newer fact, question, keywords of the right answer, keywords of the wrong one)
UPDATE = [
    (("What is my favorite drink?", "My favorite drink is coffee."), ("What is my favorite drink now?", "My favorite drink is now green tea."),
     "What do I like to drink most?", [["tea"]], ["coffee"]),
    (("What is Nadia's phone number?", "Nadia's phone number is 555-0142."), ("What is Nadia's new phone number?", "Nadia changed her number to 555-0199."),
     "How can I call Nadia?", [["0199"]], ["0142"]),
    (("Where is the team lunch?", "The team lunch is at Rosie's Diner."), ("Has the team lunch moved?", "The team lunch moved to the Blue Fern cafe."),
     "Where are we having the team lunch?", [["fern"]], ["rosie"]),
    (("What time is the school pickup?", "School pickup is at 3 pm."), ("What time is school pickup now?", "School pickup is now at 2:15 pm."),
     "When should I pick up the kids from school?", [["2:15"]], ["3 pm"]),
    (("Who is my manager?", "My manager is Linda Park."), ("Who is my new manager?", "My new manager is Victor Hale."),
     "Who do I report to?", [["victor", "hale"]], ["linda"]),
    (("What car does Omar drive?", "Omar drives a silver Honda."), ("What car does Omar drive now?", "Omar sold the Honda and now drives a blue Subaru."),
     "Which car is Omar driving these days?", [["subaru"]], []),
    (("Where does my sister Mia live?", "Mia lives in Portland."), ("Where did Mia move to?", "Mia moved to Austin last month."),
     "Which city does Mia live in?", [["austin"]], ["portland"]),
    (("What is the deadline for the report?", "The report is due on Friday."), ("Was the report deadline changed?", "The report deadline was pushed to next Wednesday."),
     "When is the report due?", [["wednesday"]], ["friday"]),
    (("What is my gym schedule?", "I go to the gym on Mondays and Thursdays."), ("What is my new gym schedule?", "I now go to the gym on Tuesdays and Saturdays."),
     "Which days do I go to the gym?", [["tuesday"], ["saturday"]], ["monday"]),
    (("How many people are coming to the party?", "Twelve people are coming to the party."), ("How many people are coming to the party now?", "Fifteen people are now coming to the party."),
     "What's the head count for the party?", [["fifteen", "15"]], ["twelve"]),
    (("What is Ben's email address?", "Ben's email is ben.k@mailbox.com."), ("What is Ben's new email?", "Ben's new email is benk.work@mailbox.com."),
     "What email should I use for Ben?", [["benk.work"]], []),
    (("Which hotel are we staying at?", "We are staying at the Harbor Inn."), ("Did the hotel change?", "We switched to the Cedar Lodge."),
     "What hotel are we booked at?", [["cedar"]], ["harbor"]),
    (("What is Zoe's favorite band?", "Zoe's favorite band is Coldplay."), ("What does Zoe listen to now?", "Zoe's favorite band is now Arctic Monkeys."),
     "Which band does Zoe like best?", [["arctic"]], ["coldplay"]),
    (("What day is trash collection?", "Trash is collected on Tuesday."), ("Has trash day changed?", "Trash collection moved to Thursday."),
     "When do they pick up the trash?", [["thursday"]], ["tuesday"]),
    (("What is the price of the monthly plan?", "The monthly plan costs 20 dollars."), ("What does the monthly plan cost now?", "The monthly plan now costs 25 dollars."),
     "How much is the monthly plan?", [["25"]], ["20 dollars"]),
    # the earlier value is asked for: picking the newest by rule would be wrong here
    (("Where does Leo work?", "Leo works at the bakery on Main Street."), ("Where does Leo work now?", "Leo now works at the city library."),
     "Where did Leo work before the library?", [["bakery"]], []),
    (("What is my address?", "My address is 12 Birch Lane."), ("What is my new address?", "My new address is 48 Willow Road."),
     "What was my old address?", [["birch"]], ["willow"]),
    (("What is Priya's last name?", "Priya's last name is Shah."), ("What is Priya's married name?", "After the wedding Priya's last name is Mehta."),
     "What was Priya's last name before she married?", [["shah"]], ["mehta"]),
    (("What color are the kitchen walls?", "The kitchen walls are yellow."), ("What color did we paint the kitchen?", "We painted the kitchen walls white."),
     "What color were the kitchen walls before we painted them?", [["yellow"]], ["white"]),
    (("What phone do I have?", "I have an iPhone 11."), ("What phone did I upgrade to?", "I upgraded to an iPhone 15."),
     "What phone did I have before the upgrade?", [["11"]], []),
]
UPDATE_KIND = ["current"] * 15 + ["earlier"] * 5

# pairs of facts that differ in one word; both are asked
NEARMISS = [
    (("What is Anna's car?", "Anna drives a red Mazda."), ("What is Anne's car?", "Anne drives a white Volvo."),
     ("What does Anna drive?", [["mazda"]]), ("What does Anne drive?", [["volvo"]])),
    (("When is Tom's birthday?", "Tom's birthday is on March 3."), ("When is Tim's birthday?", "Tim's birthday is on August 19."),
     ("What date is Tom's birthday?", [["march"]]), ("What date is Tim's birthday?", [["august"]])),
    (("What is the morning shift time?", "The morning shift starts at 6 am."), ("What is the evening shift time?", "The evening shift starts at 4 pm."),
     ("When does the evening shift begin?", [["4"]]), ("When does the morning shift begin?", [["6"]])),
    (("What is the Wi-Fi network at home?", "The home network is called NestNet."), ("What is the Wi-Fi network at work?", "The work network is called OfficeHub."),
     ("Which network do I join at work?", [["officehub"]]), ("Which network do I join at home?", [["nestnet"]])),
    (("What is my son's shoe size?", "My son's shoe size is 4."), ("What is my daughter's shoe size?", "My daughter's shoe size is 7."),
     ("What size shoes does my daughter wear?", [["7"]]), ("What size shoes does my son wear?", [["4"]])),
    (("Who is the lead on Project Falcon?", "Grace leads Project Falcon."), ("Who is the lead on Project Hawk?", "Devon leads Project Hawk."),
     ("Who runs Project Hawk?", [["devon"]]), ("Who runs Project Falcon?", [["grace"]])),
    (("What is the front gate code?", "The front gate code is 1357."), ("What is the back gate code?", "The back gate code is 2468."),
     ("How do I open the back gate?", [["2468"]]), ("How do I open the front gate?", [["1357"]])),
    (("What is Uncle Raj's favorite food?", "Uncle Raj loves lamb curry."), ("What is Uncle Ravi's favorite food?", "Uncle Ravi loves fish tacos."),
     ("What food does Ravi like best?", [["taco"]]), ("What food does Raj like best?", [["curry"]])),
    (("What platform does the 8:10 train leave from?", "The 8:10 train leaves from platform 2."), ("What platform does the 8:40 train leave from?", "The 8:40 train leaves from platform 5."),
     ("Which platform is the 8:40 train on?", [["5"]]), ("Which platform is the 8:10 train on?", [["2"]])),
    (("What is the name of Carla's horse?", "Carla's horse is called Thunder."), ("What is the name of Carla's dog?", "Carla's dog is called Biscuit."),
     ("What's Carla's horse called?", [["thunder"]]), ("What's Carla's dog called?", [["biscuit"]])),
]

UNKNOWN = ["What is Lena's favourite colour?", "How old is Jenna's cat?", "What is Marco's phone number?", "What is my boss's birthday?",
           "Where does my cousin Felix live?", "What is the code for the garage?", "When is the plumber coming?", "What is Hana's last name?",
           "Which school does Arlo go to?", "What is the name of Theo's sister?", "What is my bank account number?",
           "What time does the farmers market open on Sunday?", "What did I eat for breakfast?", "What is Gus's favourite movie?",
           "Where did I put my passport?", "What time is my haircut?", "What is the name of my neighbour's dog?",
           "How much did I pay for the couch?", "What is Coach Delgado's first name?", "Which gate does my flight to Denver leave from?"]

# (fact 1, fact 2, ambiguous question, keywords naming each option, follow-up naming option 2, keywords of option 2's answer)
CLARIFY = [
    (("When is my brother Sam's birthday?", "Sam's birthday is on June 9."), ("When is my brother Eli's birthday?", "Eli's birthday is on November 21."),
     "When is my brother's birthday?", [["sam"], ["eli"]], "I mean Eli.", [["november"]]),
    (("What is the address of the downtown office?", "The downtown office is at 200 King Street."), ("What is the address of the airport office?", "The airport office is at 7 Runway Drive."),
     "What's the office address?", [["downtown"], ["airport"]], "The airport one.", [["runway"]]),
    (("What is Kate Moreno's phone number?", "Kate Moreno's number is 555-0110."), ("What is Kate Lin's phone number?", "Kate Lin's number is 555-0177."),
     "What is Kate's number?", [["moreno"], ["lin"]], "Kate Lin.", [["0177"]]),
    (("What time is the yoga class on Monday?", "Monday yoga is at 6 pm."), ("What time is the yoga class on Wednesday?", "Wednesday yoga is at 7 am."),
     "What time is yoga class?", [["monday"], ["wednesday"]], "Wednesday.", [["7"]]),
    (("What is my daughter Ella's teacher's name?", "Ella's teacher is Mrs. Grant."), ("What is my daughter Ruby's teacher's name?", "Ruby's teacher is Mr. Chen."),
     "What is my daughter's teacher called?", [["ella"], ["ruby"]], "Ruby.", [["chen"]]),
    (("Where is the blue suitcase?", "The blue suitcase is in the attic."), ("Where is the black suitcase?", "The black suitcase is under the bed."),
     "Where is the suitcase?", [["blue"], ["black"]], "The black one.", [["bed"]]),
    (("What is the password for the work laptop?", "The work laptop password is Orchid#9."), ("What is the password for the home laptop?", "The home laptop password is River#4."),
     "What's the laptop password?", [["work"], ["home"]], "The home laptop.", [["river"]]),
    (("When is Nora's dentist appointment?", "Nora's dentist appointment is on May 2."), ("When is Owen's dentist appointment?", "Owen's dentist appointment is on May 16."),
     "When is the kids' dentist appointment?", [["nora"], ["owen"]], "Owen's.", [["16"]]),
    (("What is the name of my coworker Dana's band?", "Dana's band is called The Lowlights."), ("What is the name of my coworker Phil's band?", "Phil's band is called Static Bloom."),
     "What's my coworker's band called?", [["dana"], ["phil"]], "Phil's.", [["static"]]),
    (("How much is the small pizza?", "The small pizza is 9 dollars."), ("How much is the large pizza?", "The large pizza is 15 dollars."),
     "How much is the pizza?", [["small"], ["large"]], "The large one.", [["15"]]),
]


def facts_in_order():
    """Every stored fact, in the order they are taught: all the older facts first, the updates last."""
    out = [(q, a) for q, a, _, _ in TAUGHT]
    out += [old for old, *_ in UPDATE]
    for f1, f2, *_ in NEARMISS:
        out += [f1, f2]
    for f1, f2, *_ in CLARIFY:
        out += [f1, f2]
    out += [new for _, new, *_ in UPDATE]
    return out


def items():
    """(group, question, keyword groups, extra) for the single-turn questions; follow-ups are built by the scorer."""
    out = [("taught", ask, kw, {}) for _, _, ask, kw in TAUGHT]
    out += [("update", ask, kw, {"wrong": wrong, "kind": kind}) for (_, _, ask, kw, wrong), kind in zip(UPDATE, UPDATE_KIND)]
    for _, _, (q1, k1), (q2, k2) in NEARMISS:
        out += [("nearmiss", q1, k1, {}), ("nearmiss", q2, k2, {})]
    out += [("unknown", q, None, {}) for q in UNKNOWN]
    out += [("clarify", ask, opts, {"followup": fu, "followup_kw": fkw}) for _, _, ask, opts, fu, fkw in CLARIFY]
    return out


TEST_NAMES = {w.lower() for w in """Lena Okafor Jenna Pepper Arlo Frontier Marco Rosa Tucson Theo Hana Delgado Gus Everett Atlas Nadia Rosie
Linda Park Victor Hale Omar Mia Portland Austin Ben Harbor Cedar Zoe Leo Priya Shah Mehta Anna Anne Tom Tim Grace Devon Raj Ravi Carla
Thunder Biscuit Felix Sam Eli Kate Moreno Lin Ella Ruby Grant Chen Nora Owen Dana Phil Lowlights Orchid River NestNet OfficeHub Falcon Hawk
Mazda Volvo Subaru Coldplay maple42""".split()}

assert len(items()) + len(CLARIFY) == 100, len(items())
