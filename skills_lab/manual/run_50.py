"""Manual common-sense rating (Adam, 14 October 2026): 50 questions to a model through run_model's two-answer chat;
writes every MEMORY and NETWORK answer for rating by hand."""
import json, os, sys
ROOT = r"C:\Users\adams\OneDrive\Desktop\Text Bot Model Research\Text-Research-Model"
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.argv = sys.argv[:4]
import run_model as R, m2_qa, torch
R.SETTINGS.update(DEVICE="cpu", THREADS=10); R.say = lambda *a: None; torch.set_num_threads(10)
QS = [
 "What is the capital of France?", "Who painted the Mona Lisa?", "How many days are in a week?", "What is the boiling point of water?",
 "Why is the sky blue?", "What do bees make?", "What is photosynthesis?", "Who was the first president of the United States?",
 "What is the largest ocean on Earth?", "How many legs does a spider have?",
 "How do I boil an egg?", "How can I improve my sleep?", "Give me three tips for saving money.", "How do I make a cup of tea?",
 "What should I pack for a beach trip?", "How do I change a flat tire?", "What are some ways to reduce stress?", "How do I start learning to code?",
 "What is a noun?", "Explain what a computer virus is.", "What does recycle mean?", "Define the word 'generous'.",
 "List five fruits.", "Name three primary colors.", "List four animals that live in the ocean.", "Name some musical instruments.",
 "Write a short poem about the moon.", "Write a sentence describing a rainy day.", "Suggest a name for a pet cat.", "Write a haiku about autumn.",
 "Hello, how are you?", "Thank you for your help!", "What is your name?", "Can you tell me a joke?",
 "If I have three apples and eat one, how many do I have left?", "Which is heavier, a kilogram of feathers or a kilogram of iron?",
 "What comes after Tuesday?", "If it is raining, should I take an umbrella?", "Is ice hot or cold?", "What do you call a baby dog?",
 "What is my mother's name?", "Where did I leave my keys?", "What did I have for breakfast?", "When is my dentist appointment?",
 "Write an email to my boss asking for a day off.", "What should I cook for dinner tonight?", "Summarize the story of Cinderella.",
 "What is the difference between weather and climate?", "Why do we need to drink water?", "Recommend a good book to read.",
]
assert len(QS) == 50
VERIFY = int(sys.argv[3]) if len(sys.argv) > 3 else 0
CHAT_V2 = bool((torch.load(sys.argv[1], map_location="cpu", weights_only=False).get("args") or {}).get("chat_v2"))
if CHAT_V2:   # conversation models: the chat engine (memory, tools, optional verifier); network answer = memory and tools off
    import chat_engine as CE
    eng = CE.ChatEngine(sys.argv[1], native=R.native_encoder(), device="cpu", log=lambda *a: None)
else:
    eng = m2_qa.QAEngine(sys.argv[1], native=R.native_encoder(), device="cpu", log=lambda *a: None); R.strict_exact(eng)
out = []
for i, q in enumerate(QS, 1):
    st = {}
    if CHAT_V2:
        m, mi = eng.reply(q.encode(), 120, verify=VERIFY, history=[])
        nw, ni = eng.reply(q.encode(), 120, use_memory=False, use_tools=False, history=[])
        txt = (R._block("MEMORY", m.decode("utf-8", "replace"), f"tool: {mi['tool']} | memory: {(mi['memory'] or ['none'])[0][:60]} | "
                        f"checked {mi['candidates_checked']} | supports {mi['supports']} | {mi['seconds']}s") + "\n"
               + R._block("NETWORK", nw.decode("utf-8", "replace"), f"memory and tools off | {ni['seconds']}s"))
    else:
        txt = R.qa_turn(eng, q, st)
    out.append({"n": i, "q": q, "turn": txt})
    print(f"### {i}. {q}\n{txt}\n", flush=True)
json.dump(out, open(os.path.join(ROOT, "skills_lab", "manual", sys.argv[2] + ".json"), "w"), indent=1)
