# Multi-turn probe: `runs/mt_L1/qa_mt_L1.pt`

*2026-10-15 07:32. Earlier turns played live; the final question asked with the conversation and without it. Keyword check on the final answer.*

| group | with history | without |
| --- | ---: | ---: |
| recall | 3/6 | 0/6 |
| reference | 0/5 | 0/5 |
| followup | 2/4 | 1/4 |
| correction | 1/4 | 0/4 |
| return | 0/3 | 0/3 |
| multipart | 0/5 | 0/5 |
| **all** | **6/27** | 1/27 |

History gain on 300 held-out conversations: 4.64% (0.8969 -> 0.8553 bits per byte of the final reply).

**recall**: *My sister is called Ana and she lives in Lisbon.* -> Anamika is a beautiful and beautiful woman with a beautiful, beautiful, and beautiful life. She is a great and beautiful

> **Where does my sister live?** I live in a beautiful and beautiful, beautiful, and beautiful woman with a beautiful, lover of nature and beauty. She is a beautiful woman, and she is a lover, and is always happy to be happy and happy. (no; without history: no)

**recall**: *I have a dog named Max. He is three years old.* -> Max is a friendly and loving talking dog who loves to learn new things and meet new people. Max is well-mannered, loyal,

> **How old is my dog?** I have a dog named Max. I love to learn new things and meet new people. Max is well-mannered, loyal, and always up for an adventure! Max is good for learning new things, and loyal is good. Max loves the same thing as his friends and family. (no; without history: no)

**recall**: *My favourite colour is green.* -> The green is green, green, and green.

> **What is my favourite colour?** The green is green, green, and green. (yes; without history: no)

**recall**: *I work as a nurse at a hospital in Chicago.* -> Maggie was always on the lookout for a patient who had been admitted after experiencing a traumatic accident. She was as

> **What is my job?** My job is to provide me with a job that is not a job. I have a job in the hospital, but I have to work with a professional. I want to know what I can do to help me. I also want to provide a job to help you. I am a professional and I want you to know how to do it. I'm a professional, and I'll (no; without history: no)

**recall**: *My car is a red Toyota.* -> The red Toyota is a red Toys that is a colorful and colorful object that is colorful, with a color palette that is blue 

> **What brand is my car?** The red Toyota is a colorful and colorful object that is colorful, with a color palette that is blue and blue. (yes; without history: no)

**recall**: *I was born in 1990 in a small town.* -> I was born in 1990 in a small town. I was a young girl, and I was an old girl. I had a great time in the town, but I was

> **What year was I born?** I was born in 1990 in a small town. I was a young girl, and I was an old girl. I had a great time in the town, but I was always a little bit of a bigger issue. I didn't know what to do. I wasn't sure about the issue. (yes; without history: no)

**reference**: *Here are three cities: Paris, Tokyo and Cairo.* -> Paris, Tokyo - Cairo, Cairo - Egypt

> **Which country is the second one in?** The second one in is the second in. (no; without history: no)

**reference**: *My children are Tom, Sara and Leo.* -> Tom, Sara, and Leo.

> **What is the name of the last child I mentioned?** The last child I mentioned is the "Mother of the Child". (no; without history: no)

**reference**: *I bought apples, bread and milk today.* -> Apples, butter, butter and milk.

> **What was the second thing I bought?** Apples, butter, butter and milk. (no; without history: no)

**reference**: *The meeting is on Monday at 3 pm in room 12.* -> The next meeting will be held on Tuesday, November 3rd at 2PM, in The Conference Room.

> **What room is it in?** The room is located on the right side of the room. (no; without history: no)

**reference**: *Think about the planets Mars and Venus.* -> Mars and Venus are two planets in the Solar System. Mars and Mars are smaller than Earth, with a diameter of 6,779 kilom

> **Which of the two is closer to the Sun?** Mars and Mars are smaller than Earth. Mars is smaller than Mars and is smaller compared to Earth's surface. Mars also has a thicker surface than Earth'. Mars has a thinner surface than on Earth', with a thick, thicker, and thicker. Mars's thicker surfaces are smaller compared with Earth' and are sma (no; without history: no)

**followup**: *What is the capital of France?* -> The capital of France is Paris.

> **What river runs through it?** The river runs through it. (no; without history: no)

**followup**: *Who wrote Romeo and Juliet?* -> William Shakespeare

> **Name another play he wrote.** William Shakespeare. (no; without history: no)

**followup**: *What is photosynthesis?* -> photosynthesis is the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugar

> **What gas do plants release during it?** The gas is released during the process by which plants use light energy to convert carbon dioxide and water into oxygen and sugars. It involves the transfer of electrons and protons between molecules as chlorophyll in the presence of light absorbs the light energy used to drive the reaction. The oxy (yes; without history: yes)

**followup**: *Tell me about the Eiffel Tower.* -> The Eiffel Tower is a tower located in Paris, France. Designed by Gustave Eiffelle as the entrance to the 1889 World's F

> **In which city is it?** The city is the largest city in Paris, and it is the only city in the world. It is the home of the city's most famous city, the "European" city. (yes; without history: no)

**correction**: *My meeting is on Tuesday.* -> Tuesday, November 3rd at 2PM, is a weekly meeting. / *Sorry, I meant Thursday, not Tuesday.* -> Thursday, Thurston, and Thursh.

> **What day is my meeting?** Thursday, Thurston, and Thursh are all weekly meetings. (yes; without history: no)

**correction**: *I live in Boston.* -> The Boston metropolitan area is 617. / *Actually, I moved to Denver last week.* -> The Boston metropolitan area is 617.

> **Where do I live now?** I am not in the United States. (no; without history: no)

**correction**: *Please call me Sam.* -> I have set a reminder for you to call your parents tomorrow at 7 pm. / *On second thought, call me Alex instead.* -> I have set a reminder for you to call your parents tomorrow at 7 pm.

> **What should you call me?** I have set a reminder for you to call your parents tomorrow at 7 pm. (no; without history: no)

**correction**: *My order number is 4417.* -> Dear Customer,

Thank you for your order number 234567. We have received your order and it is being processed. You will  / *Correction: the order number is 4471.* -> Dear Customer,

Thank you for your order number 234567. We have received your order and it is being processed. You will 

> **What is my order number?** Dear Customer,

Thank you for your order number 234567. We have received your order and it is being processed. You will receive an email confirming your shipment shortly.

Thanks for your time. (no; without history: no)

**return**: *My cat is called Luna.* -> Luna is a small, lush, and beautiful country. Luna has a rich, rich history, and is known for its beautiful, beautiful,  / *What is the capital of Italy?* -> The capital of Italy is Luna, and the capital of the city is Luxembourg. It is the capital city of Italy, and is known f

> **What is my cat called?** The capital of Italy is Luxembourg. It is a beautiful, vibrant country, with a rich history and a rich culture. Lux is a rich and vibrant nation, with the city of Luxeve, the city's rich culture, and it is a great place to live in a peaceful and peaceful country, and is known for its beautiful, sun (no; without history: no)

**return**: *I am planning a trip to Spain in May.* -> I am planning a trip to Spain in May. I am sure I will be able to explore the city and explore the country. / *What is a good book to read on a plane?* -> A good book to read on a plane is "The Future of the World" by John F. Kennedy.

> **Which country am I travelling to?** A good book to read on a plane is "The Future of the World" by John F. Kennedy. (no; without history: no)

**return**: *My brother plays the guitar.* -> person / *Name a famous painter.* -> The famous painter is a painter who is known for his paintings and paintings. He is known as a paintbrush.

> **What instrument does my brother play?** My brother plays a violin. (no; without history: no)

**multipart**: 

> **I need two things. First, what is the capital of Canada? Second, what is the boiling point of water in Celsius?** The capital city of Canada is Ottawa. (no; without history: no)

**multipart**: 

> **Here is a note from my courier: 'Order 5521 will be delivered on Friday at noon.' When will it arrive, and what is the order number?** [Company Name] (no; without history: no)

**multipart**: 

> **I have three meetings tomorrow: budget at 9, design at 11 and hiring at 2. Which meeting is at 11, and what time is the hiring meeting?** The meeting is at 11 and hiring at 2. Which meeting is to be held at 3. (no; without history: no)

**multipart**: 

> **My name is Priya and I am learning to cook. I have chicken, rice and spinach at home. What could I make with them, and what is my name?** Priya is a traditional Japanese dish that is made with a variety of ingredients, including:

1. Roasted vegetables - roasted vegetables, such as carrots, cucumbers, and bell peppers
2. Roast beef - a side of roasted beef, such that is a great source of protein and fiber.

2. Lentil soup - (no; without history: no)

**multipart**: *I am planning a party for 12 people next Saturday. It will be in my garden, and two of my guests are vegetarian.* -> The food is vegetarian and vegetarian.

> **How many guests are vegetarian, and where is the party?** The food is vegetarian and vegetarian. (no; without history: no)
