# Multi-turn probe: `models/qa_best_5M_dense.pt`

*2026-10-14 08:52. Earlier turns played live; the final question asked with the conversation and without it. Keyword check on the final answer.*

| group | with history | without |
| --- | ---: | ---: |
| recall | 2/6 | 0/6 |
| reference | 0/5 | 0/5 |
| followup | 1/4 | 0/4 |
| correction | 1/4 | 0/4 |
| return | 1/3 | 0/3 |
| **all** | **5/22** | 0/22 |

History gain on 300 held-out conversations: 4.34% (0.9776 -> 0.9351 bits per byte of the final reply).

**recall**: *My sister is called Ana and she lives in Lisbon.* -> Ana is a small, small, and small, with a large, large, and large, with an estimated 100,000 to 150,500 people.

> **Where does my sister live?** My sister is a small, small, and small, with a large, large, and large, with an estimated 100,000 to 150,500 people. (no; without history: no)

**recall**: *I have a dog named Max. He is three years old.* -> Max is three years old.

> **How old is my dog?** I am a dog. (no; without history: no)

**recall**: *My favourite colour is green.* -> The green is a green color.

> **What is my favourite colour?** The green is a green color. (yes; without history: no)

**recall**: *I work as a nurse at a hospital in Chicago.* -> I'm glad I could help. I'll help you with your experience at a hospital in Chicago. 

I'm sorry to hear that you're goin

> **What is my job?** I'm glad I could help. I'll help you with your experience at a hospital in Chicago. 

I'm sorry to hear that you're going to have a great time to take care of. I can't help but feel a sense of peace and security. I feel like I've been admitted before. 
I'll be able to (no; without history: no)

**recall**: *My car is a red Toyota.* -> The car is a red Toyota.

> **What brand is my car?** I am a car. (no; without history: no)

**recall**: *I was born in 1990 in a small town.* -> The town of Lower cost of living is a small town.

> **What year was I born?** I was born in 1990 in a small town. (yes; without history: no)

**reference**: *Here are three cities: Paris, Tokyo and Cairo.* -> Paris, Paris, and Paris - Paris 
Ciro, Paris (1975)

> **Which country is the second one in?** Paris, Paris, France, France (no; without history: no)

**reference**: *My children are Tom, Sara and Leo.* -> Sara and Leo are both friends.

> **What is the name of the last child I mentioned?** The last child I mentioned was a child named Sarah. (no; without history: no)

**reference**: *I bought apples, bread and milk today.* -> Apples, bread, and milk today.

> **What was the second thing I bought?** The second thing I bought was a second thing that I hadn't ever seen. (no; without history: no)

**reference**: *The meeting is on Monday at 3 pm in room 12.* -> The next meeting will be held on Tuesday, November 3rd at 2PM, in The Conference Room.

> **What room is it in?** The room is in the room, and the room is on the floor. (no; without history: no)

**reference**: *Think about the planets Mars and Venus.* -> Mars and Venus are both planets in the Solar System. Mars is smaller than Earth, with a diameter of 6,779 kilometers com

> **Which of the two is closer to the Sun?** The two are closer to the Sun, the Sun. (no; without history: no)

**followup**: *What is the capital of France?* -> The capital of France is Paris.

> **What river runs through it?** The river runs through it. (no; without history: no)

**followup**: *Who wrote Romeo and Juliet?* -> William Shakespeare

> **Name another play he wrote.** William Shakespeare (no; without history: no)

**followup**: *What is photosynthesis?* -> Photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugar

> **What gas do plants release during it?** Photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugars. It involves the transfer of electrons and protons between molecules as chlorophyll in the presence of light absorbs the light energy used to drive the reaction. The oxygen produ (yes; without history: no)

**followup**: *Tell me about the Eiffel Tower.* -> The Eiffel Tower is a wrought iron lattice tower located in Paris, France. It is located in the heart of the Eiffer Towe

> **In which city is it?** The city is located in the heart of the Eiffer Tower, and is located on the south side of the tower. (no; without history: no)

**correction**: *My meeting is on Tuesday.* -> The next meeting will be held on Tuesday, November 3rd at 2PM, in The Conference Room. / *Sorry, I meant Thursday, not Tuesday.* -> Thursday, not Tuesday.

> **What day is my meeting?** I would like to attend the meeting. (no; without history: no)

**correction**: *I live in Boston.* -> Boston is a city in Boston that is located in Boston. / *Actually, I moved to Denver last week.* -> The Denver River is a city in Boston that is located in Boston.

> **Where do I live now?** I live now in Boston, Massachusetts. (no; without history: no)

**correction**: *Please call me Sam.* -> Sam, I have set a reminder for you to call your parents tomorrow at 7 pm. / *On second thought, call me Alex instead.* -> Alex, I have set a reminder for you to call your parents tomorrow at 7 pm.

> **What should you call me?** I should call my parents tomorrow at 7 pm. (no; without history: no)

**correction**: *My order number is 4417.* -> 1. 1217. / *Correction: the order number is 4471.* -> 1. 1217.

> **What is my order number?** My order number is 4471. (yes; without history: no)

**return**: *My cat is called Luna.* -> Luna is a popular food for a variety of dishes. / *What is the capital of Italy?* -> The capital of Italy is the capital of Rome.

> **What is my cat called?** The capital of Italy is the capital of Rome. (no; without history: no)

**return**: *I am planning a trip to Spain in May.* -> I would plan a trip to Spain in May. I would also plan a tour of the city of Barcelona and take a tour to the city. I wi / *What is a good book to read on a plane?* -> I would plan a trip to Spain in May. I will also plan a tour of it in May, and I would be happy to help you.

> **Which country am I travelling to?** I would plan a trip to Spain in May. I will also plan a tour of it in May, and I would be happy to help you. (yes; without history: no)

**return**: *My brother plays the guitar.* -> He plays the guitar. / *Name a famous painter.* -> He is a famous painter.

> **What instrument does my brother play?** He is a famous painter. (no; without history: no)
