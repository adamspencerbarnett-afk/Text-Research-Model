# Multi-turn probe: `runs/mt_M/qa_mt_M.pt`

*2026-10-15 05:30. Earlier turns played live; the final question asked with the conversation and without it. Keyword check on the final answer.*

| group | with history | without |
| --- | ---: | ---: |
| recall | 2/6 | 0/6 |
| reference | 0/5 | 0/5 |
| followup | 1/4 | 0/4 |
| correction | 1/4 | 0/4 |
| return | 2/3 | 0/3 |
| multipart | 0/5 | 0/5 |
| **all** | **6/27** | 0/27 |

History gain on 300 held-out conversations: 11.25% (0.8547 -> 0.7586 bits per byte of the final reply).

**recall**: *My sister is called Ana and she lives in Lisbon.* -> Ana is a beautiful woman with a beautiful, beautiful woman.

> **Where does my sister live?** Ana is a beautiful woman with a beautiful, beautiful woman. (no; without history: no)

**recall**: *I have a dog named Max. He is three years old.* -> Max is a friendly and loving talking dog who loves to learn new things and meet new people. He is well-mannered, loyal, 

> **How old is my dog?** I'm not sure what I've been looking for. I'd like to learn more about the dog's personality and how it affects the environment. I have a lot of fun and I'll be able to learn about the different types of dogs that I're looking for and how they affect the environment and the environment I' adventure.  (no; without history: no)

**recall**: *My favourite colour is green.* -> Green is a colorful, vibrant color, with a bright, orange color.

> **What is my favourite colour?** Green is a colorful, vibrant color, with a bright, orange color. (yes; without history: no)

**recall**: *I work as a nurse at a hospital in Chicago.* -> I'm glad I could help. I'd be happy to help you with that. I hope you have a great day to visit the hospital and see the

> **What is my job?** I'm glad I could help. I'd be happy to help you with that. I hope you have a great day to visit the hospital and see the hospital you're looking for. Good luck! (no; without history: no)

**recall**: *My car is a red Toyota.* -> The car is a red Toyota.

> **What brand is my car?** I recommend the car to my friends. (no; without history: no)

**recall**: *I was born in 1990 in a small town.* -> I was born in 1990 in a small town. I was a young girl, and I was an old woman. I had a strong sense of community and a 

> **What year was I born?** I was born in 1990 in a small town. I was a young girl, and I was an old woman. I had a strong sense of community and a strong connection with the people around me. I also had a sense of belonging and a sense that I was always looking for. I found myself in a community of people who were not interes (yes; without history: no)

**reference**: *Here are three cities: Paris, Tokyo and Cairo.* -> Paris, Tokyo, Cairo, Caira, Cadasa, Casa.

> **Which country is the second one in?** The second in the country is the second in Paris, Tokyo, Cairo, Cadasa, Caira, Casa. (no; without history: no)

**reference**: *My children are Tom, Sara and Leo.* -> Tom, Sara and Leo are both friends.

> **What is the name of the last child I mentioned?** The first child I mentioned is a child who is a young child. (no; without history: no)

**reference**: *I bought apples, bread and milk today.* -> Apples, bread, milk, and milk.

> **What was the second thing I bought?** The second thing I bought was to buy a new apple. (no; without history: no)

**reference**: *The meeting is on Monday at 3 pm in room 12.* -> 1. The meeting will be held on Tuesday, November 3rd at 3 pm.

> **What room is it in?** The room is open and open, and the room is closed. (no; without history: no)

**reference**: *Think about the planets Mars and Venus.* -> Mars and Venus are two planets in the Solar System. Mars is the largest planet in the solar system, and it is the second

> **Which of the two is closer to the Sun?** The two are closer to the Sun than the Sun. (no; without history: no)

**followup**: *What is the capital of France?* -> The capital of France is Paris.

> **What river runs through it?** The river is a river that flows through the river. (no; without history: no)

**followup**: *Who wrote Romeo and Juliet?* -> William Shakespeare

> **Name another play he wrote.** The other play he wrote is "The Future". (no; without history: no)

**followup**: *What is photosynthesis?* -> photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugar

> **What gas do plants release during it?** The gases in the atmosphere are released during the process of photosynthesis. (no; without history: no)

**followup**: *Tell me about the Eiffel Tower.* -> The Eiffel Tower is a beautiful and beautiful building located in Paris, France. It is located in the heart of Paris, an

> **In which city is it?** The city is the largest city in Paris. (yes; without history: no)

**correction**: *My meeting is on Tuesday.* -> I am not aware of any specific events or events. However, I can provide you with some information on the events and even / *Sorry, I meant Thursday, not Tuesday.* -> I am not aware of any specific events or events. However, I can provide you with some information on the events and even

> **What day is my meeting?** I am not aware of any specific events or events that are happening in Tuesday. (no; without history: no)

**correction**: *I live in Boston.* -> The area code associated with the Boston metropolitan area is 617. / *Actually, I moved to Denver last week.* -> The area code associated with the Boston metropolitan area is 617.

> **Where do I live now?** The area code associated with the Boston metropolitan area is 617. (no; without history: no)

**correction**: *Please call me Sam.* -> Okay, I have set a reminder for you to call your parents tomorrow at 7 pm. / *On second thought, call me Alex instead.* -> Okay, I have set a reminder for you to call your parents tomorrow at 7 pm.

> **What should you call me?** I call you a Alex instead. (yes; without history: no)

**correction**: *My order number is 4417.* -> 1256 / *Correction: the order number is 4471.* -> 1256

> **What is my order number?** 1256 (no; without history: no)

**return**: *My cat is called Luna.* -> Luna is a small cat with a large, rounded head. / *What is the capital of Italy?* -> The capital of Italy is Luna.

> **What is my cat called?** Luna is a small cat with a large, rounded head. (yes; without history: no)

**return**: *I am planning a trip to Spain in May.* -> I am planning a trip to Spain in May. / *What is a good book to read on a plane?* -> A good book to read on a plane is "The Future" by John F. Kennedy.

> **Which country am I travelling to?** A good book to read on a plane is "The Future" by John F. Kennedy. (no; without history: no)

**return**: *My brother plays the guitar.* -> The guitar is a violin. / *Name a famous painter.* -> The famous painter, The The The guitar, is a painter.

> **What instrument does my brother play?** My brother plays the guitar. (yes; without history: no)

**multipart**: 

> **I need two things. First, what is the capital of Canada? Second, what is the boiling point of water in Celsius?** The capital of Canada is the boiling point of water in Celsius. (no; without history: no)

**multipart**: 

> **Here is a note from my courier: 'Order 5521 will be delivered on Friday at noon.' When will it arrive, and what is the order number?** [Company Name] (no; without history: no)

**multipart**: 

> **I have three meetings tomorrow: budget at 9, design at 11 and hiring at 2. Which meeting is at 11, and what time is the hiring meeting?** I have three meetings at 11, and the hiring meeting is at 2. Which meeting is the most important meeting in the world. (no; without history: no)

**multipart**: 

> **My name is Priya and I am learning to cook. I have chicken, rice and spinach at home. What could I make with them, and what is my name?** Priya is a traditional Japanese dish that is made from a variety of ingredients, including vegetables, vegetables, and vegetables. It is a popular dish in Japan and is often served with a variety from the savory and savory flavors of the dish. (no; without history: no)

**multipart**: *I am planning a party for 12 people next Saturday. It will be in my garden, and two of my guests are vegetarian.* -> I'm sure you'll enjoy the delicious and delicious food you're going to eat.

> **How many guests are vegetarian, and where is the party?** The food you are going to eat is vegetarian, and the food you're going to consume is vegetation. (no; without history: no)
