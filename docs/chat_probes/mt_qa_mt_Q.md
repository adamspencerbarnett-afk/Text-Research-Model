# Multi-turn probe: `runs/mt_Q/qa_mt_Q.pt`

*2026-10-15 01:36. Earlier turns played live; the final question asked with the conversation and without it. Keyword check on the final answer.*

| group | with history | without |
| --- | ---: | ---: |
| recall | 4/6 | 0/6 |
| reference | 3/5 | 0/5 |
| followup | 1/4 | 0/4 |
| correction | 2/4 | 0/4 |
| return | 1/3 | 0/3 |
| multipart | 1/5 | 1/5 |
| **all** | **12/27** | 1/27 |

History gain on 300 held-out conversations: 4.96% (0.8909 -> 0.8468 bits per byte of the final reply).

**recall**: *My sister is called Ana and she lives in Lisbon.* -> Ana is called Ana, and she lives in Lisbon.

> **Where does my sister live?** Ana is called Ana, and she lives in Lisbon. (yes; without history: no)

**recall**: *I have a dog named Max. He is three years old.* -> Max is a friendly and loving dog who loves to learn new things and meet new people. Max loves the new things, and he lov

> **How old is my dog?** I have a dog who loves to learn new things and meet new people. Max loves the new things, and he loves exploring new things. He loves playing with friends and family, and loves reading books. Max is well-mannered and has a great memory. He often shares stories from his past, or talks about how he me (no; without history: no)

**recall**: *My favourite colour is green.* -> The green is green, with a bright yellow color.

> **What is my favourite colour?** The green is green, with a bright yellow color. (yes; without history: no)

**recall**: *I work as a nurse at a hospital in Chicago.* -> I'm glad to hear that you're working on the hospital. I've been working on a hospital in Chicago, and I'd be happy to he

> **What is my job?** I'm glad to hear that you're working on the hospital. I've been working on a hospital in Chicago, and I'd be happy to help you. I have been working with a hospital for several years, and it's been a great time to get started. I hope you'll find it interesting to see how the hospital's health and wel (no; without history: no)

**recall**: *My car is a red Toyota.* -> The car is a red Toyota.

> **What brand is my car?** The car is a red Toyota. (yes; without history: no)

**recall**: *I was born in 1990 in a small town.* -> I was born in 1990 in a small town. I was a young girl, and I was always a young woman. I had a great time in my life, a

> **What year was I born?** I was born in 1990 in a small town. I was a young girl, and I was always a young woman. I had a great time in my life, and it was a great place to start. I spent most of my life in the area, and was always on the lookout for a place to live. I also had a lot of fun and a lot to (yes; without history: no)

**reference**: *Here are three cities: Paris, Tokyo and Cairo.* -> Paris, Tokyo - Cairo, Cairo - Egypt

> **Which country is the second one in?** The second one in is the second in the country. (no; without history: no)

**reference**: *My children are Tom, Sara and Leo.* -> Tom, Sara, and Leo.

> **What is the name of the last child I mentioned?** The last child I mentioned is the "Little did I know". (no; without history: no)

**reference**: *I bought apples, bread and milk today.* -> Apples, bread, milk, cheese, and cheese.

> **What was the second thing I bought?** Apples, bread, milk, cheese, and cheese. (yes; without history: no)

**reference**: *The meeting is on Monday at 3 pm in room 12.* -> The next meeting will be held on Monday at 3 pm in room 12.

> **What room is it in?** The next meeting will be held on Monday at 3 pm in room 12. (yes; without history: no)

**reference**: *Think about the planets Mars and Venus.* -> Mars and Venus are planets in the Solar System. Mars is smaller than Earth, with a diameter of 6,779 kilometers compared

> **Which of the two is closer to the Sun?** Mars and Venus are planets in the Solar System. Mars is smaller than Earth, with a diameter of 6,779 kilometers compared to Earth's 12,742 kilometers. Mars also has weaker gravity than Earth and also lacks surface water and an ozone layer to protect (yes; without history: no)

**followup**: *What is the capital of France?* -> The capital of France is Paris.

> **What river runs through it?** The river runs through it. (no; without history: no)

**followup**: *Who wrote Romeo and Juliet?* -> William Shakespeare

> **Name another play he wrote.** William Shakespeare, William Shock, William, and William Shake. (no; without history: no)

**followup**: *What is photosynthesis?* -> photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugar

> **What gas do plants release during it?** The gas is released during the process of photosynthesis. (no; without history: no)

**followup**: *Tell me about the Eiffel Tower.* -> The Eiffel Tower is a tower located in Paris, France. Designed by Gustave Eiffell as the entrance to the 1889 World's Fa

> **In which city is it?** The city is the largest city in Paris, and it is the second largest city of Paris. (yes; without history: no)

**correction**: *My meeting is on Tuesday.* -> Tuesday is on Tuestone, November 1st, and November 2nd. / *Sorry, I meant Thursday, not Tuesday.* -> Thursday is on Tuesday.

> **What day is my meeting?** Thursday is on Tuesday. (yes; without history: no)

**correction**: *I live in Boston.* -> The area code associated with the Boston metropolitan area is 617. / *Actually, I moved to Denver last week.* -> The area code associated with the Boston metropolitan area is 617.

> **Where do I live now?** I moved to Denver last week. (yes; without history: no)

**correction**: *Please call me Sam.* -> I'm sorry to hear that you're going to call my parents tomorrow at 7 pm. / *On second thought, call me Alex instead.* -> I'm sorry to hear that you're going to call my parents tomorrow at 7 pm.

> **What should you call me?** I'm sorry to hear that you're going to call my parents tomorrow at 7 pm. (no; without history: no)

**correction**: *My order number is 4417.* -> 1. 1000 / *Correction: the order number is 4471.* -> 1000

> **What is my order number?** 1000 (no; without history: no)

**return**: *My cat is called Luna.* -> Luna is a small, lush, luxurious, and beautiful country. / *What is the capital of Italy?* -> The capital of Italy is Luna.

> **What is my cat called?** Luna is a small, lush, luxurious, and beautiful country. (yes; without history: no)

**return**: *I am planning a trip to Spain in May.* -> I am planning a trip to Spain in May. I am looking for a beautiful and beautiful destination. I would love to explore th / *What is a good book to read on a plane?* -> A good book to read on a plane is a book that is designed to be a beautiful and beautiful place. It is a great book to e

> **Which country am I travelling to?** A good book to read on a plane is a book that is designed to be a beautiful and beautiful place. It is a great book to explore the country. (no; without history: no)

**return**: *My brother plays the guitar.* -> The guitar is a violin. / *Name a famous painter.* -> The guitar is a violin.

> **What instrument does my brother play?** I do not have a personal instrument. (no; without history: no)

**multipart**: 

> **I need two things. First, what is the capital of Canada? Second, what is the boiling point of water in Celsius?** The capital city of Canada is Ottawa. (no; without history: no)

**multipart**: 

> **Here is a note from my courier: 'Order 5521 will be delivered on Friday at noon.' When will it arrive, and what is the order number?** [Company Name] (no; without history: no)

**multipart**: 

> **I have three meetings tomorrow: budget at 9, design at 11 and hiring at 2. Which meeting is at 11, and what time is the hiring meeting?** The meeting is at 11, and the hiring meeting is scheduled at 9, design at 2. Which meeting is on 10 AM tomorrow with the sales team. (yes; without history: yes)

**multipart**: 

> **My name is Priya and I am learning to cook. I have chicken, rice and spinach at home. What could I make with them, and what is my name?** Priya is a traditional Japanese dish that is made with a variety of vegetables, including tomatoes, onions, and garlic. It is a great addition to the dish and is a good choice for a family and family. It has a rich, creamy flavor and is often served with a side of roasted vegetables. (no; without history: no)

**multipart**: *I am planning a party for 12 people next Saturday. It will be in my garden, and two of my guests are vegetarian.* -> 1. A vegetarian dish for 12 people next Saturday. 
2. A vegan dish for a 10-15 minutes breakfast.

> **How many guests are vegetarian, and where is the party?** The average of the guests is 100,000. (no; without history: no)
