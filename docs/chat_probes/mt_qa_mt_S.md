# Multi-turn probe: `runs/mt_S/qa_mt_S.pt`

*2026-10-15 02:48. Earlier turns played live; the final question asked with the conversation and without it. Keyword check on the final answer.*

| group | with history | without |
| --- | ---: | ---: |
| recall | 4/6 | 0/6 |
| reference | 0/5 | 0/5 |
| followup | 2/4 | 0/4 |
| correction | 1/4 | 0/4 |
| return | 3/3 | 0/3 |
| multipart | 1/5 | 1/5 |
| **all** | **11/27** | 1/27 |

History gain on 300 held-out conversations: 9.41% (0.8439 -> 0.7644 bits per byte of the final reply).

**recall**: *My sister is called Ana and she lives in Lisbon.* -> Anamika is a beautiful and beautiful woman with a beautiful, beautiful woman. She lives in Lisbon, and she lives in the 

> **Where does my sister live?** My sister is a beautiful woman with a beautiful, beautiful woman. She lives in Lisbon, and she lives in the Lisbo region of Lisa. (yes; without history: no)

**recall**: *I have a dog named Max. He is three years old.* -> Max is a friendly and loving dog who loves to learn new things and meet new people. Max loves his loves and loves.

> **How old is my dog?** I have a dog who loves to learn new things and meet new people. Max loves his loves and loves. (no; without history: no)

**recall**: *My favourite colour is green.* -> Green is a colorful, vibrant green, with a bright, bright color.

> **What is my favourite colour?** Green is a colorful, vibrant green, with a bright, bright color. (yes; without history: no)

**recall**: *I work as a nurse at a hospital in Chicago.* -> I'm glad I could help. I'd be happy to help you with that. Let me know if you have any other questions.

> **What is my job?** I'm glad I could help. I'd be happy to help you with that. Let me know if you have any other questions. (no; without history: no)

**recall**: *My car is a red Toyota.* -> The car is a red Toyota.

> **What brand is my car?** The car is a red Toyota. (yes; without history: no)

**recall**: *I was born in 1990 in a small town.* -> I was born in 1990 in a small town. I was a young girl, and I was an old woman. I had a great time in my life, but I was

> **What year was I born?** I was born in 1990 in a small town. I was a young girl, and I was an old woman. I had a great time in my life, but I was always on the lookout for a new family. I loved the way I was, and it was a great place to start. (yes; without history: no)

**reference**: *Here are three cities: Paris, Tokyo and Cairo.* -> Paris, Tokyo - Cairo - Paris, Cairo

> **Which country is the second one in?** The second one in the country is Cairo. (no; without history: no)

**reference**: *My children are Tom, Sara and Leo.* -> Tom

> **What is the name of the last child I mentioned?** Tom (no; without history: no)

**reference**: *I bought apples, bread and milk today.* -> Apples, eggs, milk, and milk are all the same.

> **What was the second thing I bought?** Apples, eggs, milk, and milk are all the same. (no; without history: no)

**reference**: *The meeting is on Monday at 3 pm in room 12.* -> The next meeting will be held on Tuesday, November 3rd at 2PM, in The Conference Room.

> **What room is it in?** The room is open to the public. (no; without history: no)

**reference**: *Think about the planets Mars and Venus.* -> Mars and Venus are planets in the Solar System. Mars is a planet with a diameter of 6,779 kilometers compared to Earth's

> **Which of the two is closer to the Sun?** Mars is a planet with a warmer climate compared to Earth's 12,742 kilometers compared to the Earth'. Mars is also a planet that has a warm climate compared with Earth', meaning it has a colder climate than Earth'). Mars also has a cooler climate, meaning its warmer atmosphere is warmer than Earth. M (no; without history: no)

**followup**: *What is the capital of France?* -> The capital of France is Paris.

> **What river runs through it?** The river runs through it. (no; without history: no)

**followup**: *Who wrote Romeo and Juliet?* -> William Shakespeare

> **Name another play he wrote.** William Shakespeare, a famous play he wrote, is a play he composed for the play. (no; without history: no)

**followup**: *What is photosynthesis?* -> Photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugar

> **What gas do plants release during it?** Photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugars. It involves the transfer of electrons and protons between molecules as chlorophyll in the presence of light absorbs the light energy used to drive the reaction. (yes; without history: no)

**followup**: *Tell me about the Eiffel Tower.* -> The Eiffel Tower is a tower located in Paris, France. Designed by Gustave Eiffell as the entrance to the 1889 World's Fa

> **In which city is it?** The Eiffel Tower is a tower located in Paris, France. Designed by Gustave Eiffell as the entrance to the 1889 World's Fair, the 2063 Fair is the tallest structure in Paris. The tower is composed of eighteen lattice sections connected to one another. Each leg is then (yes; without history: no)

**correction**: *My meeting is on Tuesday.* -> Tuesday is on Tuestone, November 1st, 2019. / *Sorry, I meant Thursday, not Tuesday.* -> Thursday is on Tuesday, and it is not a weekday.

> **What day is my meeting?** Thursday is on Tuesday, and it is not a weekday. (yes; without history: no)

**correction**: *I live in Boston.* -> The area code associated with the Boston metropolitan area is 617. / *Actually, I moved to Denver last week.* -> The area code associated with the Boston metropolitan area is 617.

> **Where do I live now?** I do not have access to the specific area code associated with the Boston metropolitan area. However, I can provide you with some information about the location code associated. (no; without history: no)

**correction**: *Please call me Sam.* -> I'm sorry for you. / *On second thought, call me Alex instead.* -> I'm sorry for you.

> **What should you call me?** I'm sorry for you. (no; without history: no)

**correction**: *My order number is 4417.* -> [Company Name] / *Correction: the order number is 4471.* -> [Company Name]

> **What is my order number?** I don't have the ability to have any order number. However, I can tell you that I can help you with that. (no; without history: no)

**return**: *My cat is called Luna.* -> Luna is a small, lush, green cat with a long, long, short, long tail. Luna has a long tail and a long neck. Lunch is a l / *What is the capital of Italy?* -> The capital of Italy is Luna, the capital of the world.

> **What is my cat called?** Luna is a small, lush, green cat with a long tail and a long neck. Lunch is a long and long tail, with a short tail and long neck that is long and short tail. Lutheran Immortality is a lush and lush cat with long tails and a short neck that are long and longer tails. Lutero is a (yes; without history: no)

**return**: *I am planning a trip to Spain in May.* -> I am not able to travel to Spain. / *What is a good book to read on a plane?* -> A good book to read on a plane is "The Future" by Robert Frost.

> **Which country am I travelling to?** I am not able to travel to Spain. (yes; without history: no)

**return**: *My brother plays the guitar.* -> The guitar is a violin. / *Name a famous painter.* -> The guitar is a violin.

> **What instrument does my brother play?** The guitar is a violin. (yes; without history: no)

**multipart**: 

> **I need two things. First, what is the capital of Canada? Second, what is the boiling point of water in Celsius?** The capital city of Canada is Ottawa. (no; without history: no)

**multipart**: 

> **Here is a note from my courier: 'Order 5521 will be delivered on Friday at noon.' When will it arrive, and what is the order number?** [Company Name] (no; without history: no)

**multipart**: 

> **I have three meetings tomorrow: budget at 9, design at 11 and hiring at 2. Which meeting is at 11, and what time is the hiring meeting?** The meeting is scheduled at 11, and the meeting is at 2. Which meeting is on 10, and at 3. (no; without history: no)

**multipart**: 

> **My name is Priya and I am learning to cook. I have chicken, rice and spinach at home. What could I make with them, and what is my name?** Priya is a traditional Japanese dish that is made with chicken, rice, and vegetables. It is made from a variety of ingredients, including tomatoes, onions, and garlic. It has a rich, creamy flavor and is often served with a variety from the comfort of the dish.

Ingredients:

- 1/2 cup of chicken, c (yes; without history: yes)

**multipart**: *I am planning a party for 12 people next Saturday. It will be in my garden, and two of my guests are vegetarian.* -> I'm sorry to hear that you're not sure if you've been planning a party.

> **How many guests are vegetarian, and where is the party?** I'm sorry to hear that you're not sure if you've been planning a party. (no; without history: no)
