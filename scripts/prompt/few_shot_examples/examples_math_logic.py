# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


only_text = [
    {
        "question": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
        "reference_solution": "We let x be the amount of money she put in the bank the first year.\nThen the second year was 2*x, \nand the third year was 2*2*x=4*x,\nand the fourth year was 2*2*2*x=8*x.\nAnd the total amount she put in the bank, $450, is the sum of all four years, or $450 = x+(2*x)+(4*x)+(8*x).\nThis simplifies to $450=15*x.\nAnd therefore, the amount of money she put in the bank the first year was x=$30.",
        "generated_solution": """Let x be the amount of money Missy put in the bank the first year.She was then doubling the amount of money every year for 4 years.So in the second year she put 2*x amount in her bank.In the third year she put 2*2*x = 4*x amount in her bank.And in the fourth year she put 2*4*x = 8*x amount in her bank.So the total amount she put in the bank is x+(2*x)+(4*x)+(8*x)=15*x.As the problem states, this total amounts to $450.So 450=15*x which implies x=30.Thus, Missy put \\boxed{30} dollars in the bank the first year.""".strip(),
    },
    {
        "question": "Pete has to take a 10-minute walk down to the train station and then board a 1hr 20-minute train to LA. When should he leave if he cannot get to LA later than 0900 hours? (24-hr time)",
        "reference_solution": "There are 60 minutes in an hour so 1 hour 20 minutes = (60+20) minutes = 80 minutes\nHe will spend a total of 80+10 = 90 minutes\n90 minutes is = (60/60) hours and 30 minutes = 1 hour 30 minutes = 0130 in 24-hr time\nHe has to leave 0130 hours earlier than 0900 i.e. 0900-0130 = 0730 hours",
        "generated_solution": """Since Pete needs to take a 10 minutes walk and then a 1 hour 20 minutes train ride, he will spend a total of 1 hour and 30 minutes.This means that he needs to leave 1 hour and 30 minutes earlier than 09:00 hours.Subtracting 1 hour and 30 minutes from 0900 hours we get \\boxed{07:30} hours.""".strip(),
    },
    {
        "question": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
        "reference_solution": "Five times the amount Mark deposited is $88 x 5 = $440.\nSo, Bryan deposited $440 - $40 = $400.",
        "generated_solution": """Five times of what Mark deposited is $88*5=$440.Bryan deposited $440-$40=$400.So Bryan deposited \\boxed{400} dollars in the bank.""".strip(),
    },
    {
        "question": "A department store displays a 20% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
        "reference_solution": "The amount of the discount is $120 x 0.2 = $24.\nThe new price will be $120 \u2013 $24 = $96.",
        "generated_solution": """The lamp was originally priced at $120.A 20% discount amounts to $120*20/100=$24.So the discount reduces the price of the lamp to $120-$24=$96.So the new price of the lamp is \\boxed{96} dollars.""".strip(),
    }
]

# Solutions are deductive
deductive = [
    {
      "question": "Mark deposited $88 in a bank. Bryan deposited $40 less than five times as much as Mark. How much did Bryan deposit in the bank?",
      "reference_solution": "[Deductive Reasoning] Look at the specific facts or conditions provided in the question. Using the general rule and the specific premises, deduce what must logically follow. A valid deductive argument guarantees the truth of the conclusion if the premises are true. Review each possible candidate provided in the problem and determine which one, if any, aligns with your deduced conclusion.",
      "generated_solution": "Use deductive reasoning, we can find five times the amount Mark deposited is $88 x 5 = $440. So, Bryan deposited $440 - $40 = $\\boxed{400}."
    },
    {
      "question": "James opens up a flower shop.  He needs to pay rent of $1200 a week with an additional 20% of rent to pay for utilities and he has 2 employees per shift with the store open 16 hours a day for 5 days a week.  If he pays each employee $12.50 an hour, what are his weekly expenses to run the store?",
      "reference_solution": "[Deductive Reasoning] Look at the specific facts or conditions provided in the question. Using the general rule and the specific premises, deduce what must logically follow. A valid deductive argument guarantees the truth of the conclusion if the premises are true. Review each possible candidate provided in the problem and determine which one, if any, aligns with your deduced conclusion.",
      "generated_solution": "Use deductive reasoning to reason from the given facts, we can find the shop is open 16*5=80 hours a week. So James pays each of his employees 12.5*80=$1000 per week. Since there are two employees, he pays them in total 2 * $1000 = $2000 per week. The utilities cost 1200*.2=$240. So his total cost per week is 1200+2000+240=$\\boxed{3440}."
    },
    {
      "question": "A department store displays a 20\% discount on all fixtures. What will be the new price of a 25 cm high bedside lamp that was worth $120?",
      "reference_solution": "[Deductive Reasoning] Look at the specific facts or conditions provided in the question. Using the general rule and the specific premises, deduce what must logically follow. A valid deductive argument guarantees the truth of the conclusion if the premises are true. Review each possible candidate provided in the problem and determine which one, if any, aligns with your deduced conclusion.",
      "generated_solution": "From the given condition, the amount of the discount is $120 x 0.2 = $24. The new price will be $120 - $24 = $\\boxed{96}."
    },
  ]

# Solutions are inductive
inductive = [
    {
      "question": "Missy had a giant piggy bank in her bedroom.  Every day she would search the house for change to put in her bank.  After 4 years, the bank was opened and it contained $450 in change.  If the second, third, and fourth-year she doubled the amount of money she put in the bank from the amount she had put in the previous year, how much money, in dollars, did she put in the bank the first year?",
      "reference_solution": "[Inductive Reasoning] Induction is the process of making broad generalizations from specific observations. It involves looking at specific examples or scenarios within a text and forming broader, generalized conclusions from them.",
      "generated_solution": "Using inductive reasoning, we look at the patterns from specific observed behaviors about the money saved and generalize to find the amount Missy put in the bank the first year.\n\n1. Observing the Financial Growth:\n   - It's given that the amount deposited each year doubled from the previous year, starting from the second year.\n   - Hence, if $x$ is the amount deposited in the first year, then the second year saw $2x$, the third year $4x$, and by the fourth year $8x$.\n\n2. Summing Total Savings:\n   - Based on the doubling pattern, the total amount saved in 4 years was:\n   - $x + 2x + 4x + 8x = 15x$.\n\n3. Applying the Specific Total Given:\n   - We know the total amount in the piggy bank after four years was $450.\n   - Thus, $15x = 450$, solving for $x$ gives $x = 450 / 15 = 30$. So the answer is \\boxed{30}."
    },
    {
      "question": "I have 5 marbles numbered 1 through 5 in a bag.  Suppose I take out two different marbles at random.  What is the expected value of the sum of the numbers on the marbles?",
      "reference_solution": "[Inductive Reasoning] Induction is the process of making broad generalizations from specific observations. It involves looking at specific examples or scenarios within a text and forming broader, generalized conclusions from them.",
      "generated_solution": "First, let's do some specific examples. If we have 2 marble, then the sum is $3$ and the expected value is $3/2=1.5$. If we have 3 marbles, we can take [(1,2),(1,3),(2,3)] from it and get the sum of 3,4,5. The expected value is $(3+4+5)/3 = 4$. So, by induction, we can make the broader generalization that for any set of integers from 1 to n, if we randomly pick two distinct numbers from the set, the expected value of their sum is around the middle of the range of numbers. This is because each possible pair is equally likely, and the pairs have sums that are symmetrically distributed around the middle of the range of numbers. If we take $n=4$ to verify this conclusion, we can get [(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)] and the sum is $(3+4+5+5+6+7)/6=5$, which meets our conclusion. So for $n=5$, the 10 possible pairs of marbles that we can take out from the bag are [(1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)], and the average sum is around the middle of the sum values, which is $\\boxed{6}$."
    },
    {
      "question": "Melinda will roll two standard six-sided dice and make a two-digit number with the two numbers she rolls. For example, if she rolls a 6 and a 3, she can either form 36 or 63. What is the probability that she will be able to make an integer between 10 and 20, inclusive? Express your answer as a common fraction.",
      "reference_solution": "[Inductive Reasoning] Induction is the process of making broad generalizations from specific observations. It involves looking at specific examples or scenarios within a text and forming broader, generalized conclusions from them.",
      "generated_solution": "Using inductive reasoning, let's begin by looking at specific examples of rolling two dice and then try to generalize to find the probability of forming a number between 10 and 20. A number between 10 and 20 must start with 1, so we know one die must show a 1, and the other die can be any number 0 through 9. Since a die has numbers 1 through 6, we only consider outcomes with one die showing 1 and the other showing a number between 2 and 6 (since there's no 0 on a die, and 1-1 would give us 11, which is not between 10 and 20 if 1 is the first digit). Here're the possible pairs when one die is 1: [(1,1),(1,2),(1,3),(1,4),(1,5),(1,6)]. So, we find that there are 11 possible outcomes that result in a two-digit number between 10 and 20 inclusive. Now, we need to consider the total number of outcomes. Each die has 6 sides, so rolling two dice gives us 6*6 = 36 total possible outcomes. So, the probability of forming a two-digit number between 10 and 20 is \\boxed{\\frac{11}{36}}."
    },
  ]

# Solutions use analogical
analogical = [
  {
    "question": "Yvonne brings a box of chocolates to school. Half have nuts and half do not. The students eat 80% of the ones with nuts and eat half of the ones without nuts. If there are 28 chocolates left, how many chocolates were in the box?",
    "reference_solution": "[Analogical Reasoning] Retrieve several similar cases and draw the conclusion of this problem based on the similarity.",
    "retrieval": "Question: Monica takes a selection of sweets to class. Fifty percent of these sweets contain almonds, while the other fifty percent are almond-free. The pupils consume 80% of the almond sweets and 50% of the almond-free sweets. If there are 35 sweets remaining, how many sweets were there initially in the selection? \nAnswer: From the provided information, we ascertain that 20% of the almond sweets remain uneaten, equating to 10% of the total sweets. Similarly, 50% of the almond-free sweets remain, amounting to 25% of the total sweets. Thus, the 35 remaining sweets correspond to 35% of the initial total. To calculate the original count of sweets, we use the formula $35 / 35% = \\boxed{100}$.",
    "generated_solution": "Based on the example and using analogous reasoning. We can ascertain that 20% of the chocolates with nuts remain uneaten, equating to 10% of the total chocolates. Similarly, 50% of the nut-free chocolates remain, amounting to 25% of the total chocolates. Thus, the 28 remaining chocolates correspond to 35% of the initial total. To calculate the original count of chocolates, we use the formula $28 / 35% = \\boxed{80}$. So, there were originally 80 chocolates in the box."
  },
  {
    "question": "John is 24 years younger than his dad. The sum of their ages is 68 years. How many years old is John?",
    "reference_solution": "[Analogical Reasoning] Retrieve several similar cases and draw the conclusion of this problem based on the similarity.",
    "retrieval": "Question: Lisa is 10 years younger than her mom. The sum of their ages is 70 years. How old is Lisa?\nAnswer: Lisa is 30 years old and her mom is 40 years old.",
    "generated_solution": "These are solved using the same approach as the problem about John and his dad's ages, i.e., setting up two equations based on the information given and then solving for the two variables representing the ages. Therefore, for the given question, John is $\\boxed{22}$ years old."
  },
  {
    "question": "Let $t(x) = sqrt{3x+1}$ and $f(x)=5-t(x)$. What is $t(f(5))$?",
    "reference_solution": "[Analogical Reasoning] Retrieve several similar cases and draw the conclusion of this problem based on the similarity.",
    "retrieval": "Question: Let $g(x) = sqrt{2x + 4}$ and $h(x) = 3 - g(x)$. What is $g(h(3))$?\nAnswer: \nFirst find $h(3)$ by plugging $3$ into $h(x)$ which gives $3 - sqrt{2*3 + 4} = 3 - sqrt{10}$\nThen take this value and plug it into $g(x)$ which gives $sqrt{2*(3-sqrt{10}) + 4} = sqrt{10}$",
    "generated_solution": "In these case we compute the inner function first and then apply the outer function to its result. This is known as function composition and is a common kind of exercise in algebra and calculus. Therefore, for the given question, we first evaluate $f(5) = 5 -t(5) = 5-sqrt{5cdot3+1}=1$. Thus $t(f(5))=t(1)=sqrt{3cdot1 + 1}=\\boxed{2}$."
  }
]

# Solutions use abductive
abductive = [
    {
      "question": "The integer $m$ is between $30$ and $80$ and is a multiple of $6$. When $m$ is divided by $8,$ the remainder is $2$. Similarly, when $m$ is divided by $5,$ the remainder is $2$. What is the value of $m$?",
      "reference_solution": "[Abductive Reasoning] Assume one candidate is correct and check whether it meets the condition in the problem.",
      "generated_solution": "To solve this problem using abductive reasoning, we assume that one possible value of $m$ exists that abides by the constraints and check if this assumption holds.\n\n1. First, filter values of $m$ that are multiples of $6$ between $30$ and $80$. These are $36, 42, 48, 54, 60, 66, 72, 78$.\n\n2. Next, apply the condition that when $m$ is divided by $8$, the remainder is $2$. Checking through our list we can find that only $42, 66$ fit this condition.\n\n3. Apply the third condition, that when divided by $5$, $m$ should leave a remainder of $2$. Testing the applicable values so far and find $42$ get a remainder of $2$ while $66$ get a remainder of $6$ \n\n4. Finally, converge on $m = 42$ as it fits all conditions. So the answer is $\\boxed{42}$."
    },
    {
      "question": "A two-digit integer $AB$ equals $\\frac{1}{9}$ of the three-digit integer $AAB$, where $A$ and $B$ represent distinct digits from 1 to 9. What is the smallest possible value of the three-digit integer $AAB$?",
      "reference_solution": "[Abductive Reasoning] Assume one candidate is correct and check whether it meets the condition in the problem.",
      "generated_solution": "To solve the problem using abductive reasoning, start by translating the given information into mathematical expressions and assume potential values to find the one that satisfies all conditions. \n\n1. Expressing $AB$ and $AAB$ in terms of their digits, we have: \n   - $AB = 10A + B$\n   - $AAB = 110A + B$\n\n2. According to the problem, $AB = \\frac{1}{9} \\times AAB$. So, we have $5A = 2B$.\n\n3. Since $A$ and $B$ are digits (1-9), we can assume that $A = 2, B = 5$ because they are the smallest possible values that satisfy this condition.\n\n4. Then we get $AAB = 225$.\n\n5. Finally, verify if the conditions are mathematically correct with our assumed values: Check if $AB = \\frac{1}{9} \\times AAB$: $25 = \\frac{1}{9} \\times 225$ indeed holds true.\n\nThus, the conditions are satisfied and the smallest possible value of $AAB$ with these digits is $\\boxed{225}$."
    },
    {
      "question": "Question: Find the smallest positive integer $x$ which is greater than $1$ and relatively prime to $120$ (recall that relatively prime means that the GCD of $x$ and $120$ is $1$)",
      "reference_solution": "[Symbolize Reasoning] Use logical operations such as modus ponens (if p then q; p, therefore q), modus tollens (if p then q; not q, therefore not p), disjunctions, and conjunctions, among others, to evaluate the premises and infer conclusions.",
      "generated_solution": "To solve the problem using abductive reasoning, we observe that the prime factorization of $120$ is equal to $2^3 \cdot 3 \cdot 5$.  Therefore the possible candidates are: $2$, $3$, $4$, $5$, $6$, $7$. By checking these candidates one by one, we can find that $\\boxed{7}$ does not share a prime factor with $120$."
    }
  ]


examples_map = {
    "math_text": only_text,
    "math_deductive": deductive,
    "math_inductive": inductive,
    "math_analogical": analogical,
    "math_abductive": abductive,
}
