# Multi-turn probe: `runs/mt_L2/qa_mt_L2.pt`

*2026-10-15 09:08. Earlier turns played live; the final question asked with the conversation and without it. Keyword check on the final answer.*

| group | with history | without |
| --- | ---: | ---: |
| recall | 4/6 | 0/6 |
| reference | 0/5 | 0/5 |
| followup | 2/4 | 0/4 |
| correction | 1/4 | 0/4 |
| return | 1/3 | 0/3 |
| multipart | 1/5 | 1/5 |
| **all** | **9/27** | 1/27 |

History gain on 300 held-out conversations: 8.44% (0.8705 -> 0.7971 bits per byte of the final reply).

**recall**: *My sister is called Ana and she lives in Lisbon.* -> Ana is called Ana, and she lives in Lisbon.

> **Where does my sister live?** Ana lives in Lisbon. (yes; without history: no)

**recall**: *I have a dog named Max. He is three years old.* -> Max is a friendly and loving talking dog who loves to learn new things and meet new people. Max is well-mannered, loyal,

> **How old is my dog?** I don't have access to the latest information about my dog. I have a dog named Max. I love to learn new things and meet new people. I'm always up for an adventure! Max loves the way I loves and loves. I always have a good time and have a great time. I also love to love to be a good friend and a good (no; without history: no)

**recall**: *My favourite colour is green.* -> The green is green, green, and green. It is a bright green, with a bright red, orange, and red. It has a bright yellow, 

> **What is my favourite colour?** The green is green, green, and green, with a bright red, orange, and red. It has a bright yellow, green and green. It is a bright green, a bright orange, with red and green green, both red and red green. (yes; without history: no)

**recall**: *I work as a nurse at a hospital in Chicago.* -> I'm glad to hear that you're working on the hospital. I've been working on a hospital in Chicago, and I'd be happy to he

> **What is my job?** I'm glad to hear that you're working on the hospital. I've been working on a hospital in Chicago, and I'd be happy to help you with that. I hope you have a better day. (no; without history: no)

**recall**: *My car is a red Toyota.* -> The car is a red Toyota.

> **What brand is my car?** The car is a red Toyota. (yes; without history: no)

**recall**: *I was born in 1990 in a small town.* -> I was born in 1990 in a small town. I was a young girl, and I was always a little bit of a bigger than mys. I had a lot 

> **What year was I born?** I was born in 1990 in a small town. I was a young girl, and I was always a little bit of a bigger than mys. I had a lot of fun and a lot to do. I always had a great time and I had the same thing. I spent a lot in the woods and was always on the look. I never had a good (yes; without history: no)

**reference**: *Here are three cities: Paris, Tokyo and Cairo.* -> Paris, Tokyo - Cairo, Cairo - Egypt

> **Which country is the second one in?** The second one in is the second in. (no; without history: no)

**reference**: *My children are Tom, Sara and Leo.* -> Tom

> **What is the name of the last child I mentioned?** The last child I mentioned is Tom. (no; without history: no)

**reference**: *I bought apples, bread and milk today.* -> Apples, butter, butter and milk.

> **What was the second thing I bought?** Apples, butter, butter and milk. (no; without history: no)

**reference**: *The meeting is on Monday at 3 pm in room 12.* -> The next meeting will be held on Tuesday, November 3rd at 2PM, in The Conference Room.

> **What room is it in?** The room is located on the right side of the room, and the room is open. (no; without history: no)

**reference**: *Think about the planets Mars and Venus.* -> Mars and Venus are two planets in the Solar System. Mars is smaller than Earth, with a diameter of 6,779 kilometers comp

> **Which of the two is closer to the Sun?** Mars is closer to the Sun than Earth. Mars is smaller than Earth, with a diameter of 6,779 kilometers compared to Earth's 12,742 kilometers. Mars also has more hotter climate than Earth''s. Mars has a much colder climate than earth, meaning objects weigh less on its surface than Earth or also. Mars (no; without history: no)

**followup**: *What is the capital of France?* -> The capital of France is Paris.

> **What river runs through it?** The river runs through it. (no; without history: no)

**followup**: *Who wrote Romeo and Juliet?* -> William Shakespeare

> **Name another play he wrote.** William Shakespeare is a play he wrote. (no; without history: no)

**followup**: *What is photosynthesis?* -> photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugar

> **What gas do plants release during it?** The gas is released during the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugars. It involves the transfer of electrons and protons between molecules as chlorophyll in the presence of light absorbs the light energy used to drive the reaction. The oxy (yes; without history: no)

**followup**: *Tell me about the Eiffel Tower.* -> The Eiffel Tower is a tower located in Paris, France. Designed by Gustave Eiffelle, the tower is the tallest structure i

> **In which city is it?** The city is the largest city in Paris, and it is the tallest city in the world. It is the world's tallest building, with a total of 125,000 square feet. (yes; without history: no)

**correction**: *My meeting is on Tuesday.* -> Tuesday is on Tuestone, November 1st, and November 2nd at 2PM. / *Sorry, I meant Thursday, not Tuesday.* -> Thursday is on Tuesday, not Tuestone, November 2nd at 2PM.

> **What day is my meeting?** Thursday is on Tuesday, not Tuestone, November 2nd at 2PM. (yes; without history: no)

**correction**: *I live in Boston.* -> The Boston metropolitan area is 617. / *Actually, I moved to Denver last week.* -> The Boston metropolitan area is 617.

> **Where do I live now?** I do not have access to real-time information. However, I can tell you that the city is home to many of the most famous and well-known cities in the world. (no; without history: no)

**correction**: *Please call me Sam.* -> Okay, I have set a reminder for you to call your parents tomorrow at 7 pm. / *On second thought, call me Alex instead.* -> I have set a reminder for you to call your parents tomorrow at 7 pm.

> **What should you call me?** I call you a reminder for you to call your parents tomorrow at 7 pm. (no; without history: no)

**correction**: *My order number is 4417.* -> [Company Name] / *Correction: the order number is 4471.* -> [Company Name]

> **What is my order number?** [Company Name] (no; without history: no)

**return**: *My cat is called Luna.* -> Luna is a small, lush, and well-known species of the lush forest. It is a large, well-drained, and has a long, dark, and / *What is the capital of Italy?* -> The capital of Italy is the capital of the Italian city of Rome.

> **What is my cat called?** I don't have a cat called. (no; without history: no)

**return**: *I am planning a trip to Spain in May.* -> I am planning a trip to Spain in May. / *What is a good book to read on a plane?* -> A good book to read on a plane is "The Catcher in the World".

> **Which country am I travelling to?** A good book to read on a plane is "The Catcher in the World". (no; without history: no)

**return**: *My brother plays the guitar.* -> He is a guitar. He is an instrumental guitar. / *Name a famous painter.* -> He is a guitar.

> **What instrument does my brother play?** He is a guitar. (yes; without history: no)

**multipart**: 

> **I need two things. First, what is the capital of Canada? Second, what is the boiling point of water in Celsius?** The capital city of Canada is Ottawa. (no; without history: no)

**multipart**: 

> **Here is a note from my courier: 'Order 5521 will be delivered on Friday at noon.' When will it arrive, and what is the order number?** [Your Name] (no; without history: no)

**multipart**: 

> **I have three meetings tomorrow: budget at 9, design at 11 and hiring at 2. Which meeting is at 11, and what time is the hiring meeting?** The meeting is scheduled at 11, and the meeting is at 2. Which meeting is expected at 9, design at 3. (yes; without history: yes)

**multipart**: 

> **My name is Priya and I am learning to cook. I have chicken, rice and spinach at home. What could I make with them, and what is my name?** I do not have access to the specific recipe or recipe you are interested in. However, you can try using a variety of ingredients such as olive oil, garlic powder, onion powder, garlic, and onion powder. You can also try using fresh herbs like thyme, rosemary, or thyme for a more flavorful and flavor (no; without history: no)

**multipart**: *I am planning a party for 12 people next Saturday. It will be in my garden, and two of my guests are vegetarian.* -> I'm glad to hear that you're planning a party.

> **How many guests are vegetarian, and where is the party?** I'm glad to hear that you're planning a party. (no; without history: no)
