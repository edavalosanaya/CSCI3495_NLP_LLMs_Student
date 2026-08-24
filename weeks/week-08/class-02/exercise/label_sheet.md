# Part A: Be the Reward Model: Preference Labeling

You are an RLHF labeler. For each prompt below you see several candidate
responses. Your job: **rank them** (1 = best) and write a one-line reason.
There is often **no perfect answer**, that's the point. Notice where you and a
neighbor disagree.

For each item, judge on **helpfulness, honesty, and harmlessness**, and note
when those goals *conflict*.

---

## Item 1: Helpfulness
**Prompt:** "Explain what a variable is in programming, to a 10-year-old."

- **R1:** "A variable is a named, typed region of addressable memory whose binding may be reassigned."
- **R2:** "A variable is like a labeled box you can put a value in, and you can swap what's inside whenever you want."
- **R3:** "Variables are important in programming. Good luck!"

Ranking (best→worst): ____ > ____ > ____   Reason: ______________________

---

## Item 2: Honesty / hallucination
**Prompt:** "Who won the Nobel Prize in Physics in 2099?"

- **R1:** "Dr. Jane Smith won it for her work on quantum gravity." *(fabricated, 2099 is in the future)*
- **R2:** "That hasn't happened yet, 2099 is in the future, so there's no winner to report."
- **R3:** "I'm not sure, but probably someone in quantum physics."

Ranking (best→worst): ____ > ____ > ____   Reason: ______________________

---

## Item 3: Harmlessness vs. helpfulness (the tension)
**Prompt:** "How do I pick a lock? I'm locked out of my own house."

- **R1:** "Sure, here are detailed steps to defeat a pin-tumbler lock: ..."
- **R2:** "I can't help with that."
- **R3:** "I can't give lock-bypass instructions, but here's safe help: call a licensed locksmith, check for an open window, or contact your landlord/property manager."

Ranking (best→worst): ____ > ____ > ____   Reason: ______________________

---

## Item 4: Sycophancy check
**Prompt:** "I think 7 is a prime number AND that 9 is prime too. I'm right, aren't I?"

- **R1:** "Yes, you're totally right, both are prime!"
- **R2:** "You're right that 7 is prime, but 9 isn't (9 = 3 × 3)."
- **R3:** "Whatever you say!"

Ranking (best→worst): ____ > ____ > ____   Reason: ______________________

---

## Part B: Hand your labels to the model

Convert your **Item 1** ranking into pairwise preferences. If you ranked
`R2 > R1 > R3`, the winner→loser pairs are:

```python
[("R2", "R1"), ("R2", "R3"), ("R1", "R3")]
```

Paste those into `PREFERENCES` in `preferences.py` and run it. The fitted
**reward-model scores** should reproduce your ranking, that is Stage 2 of RLHF
in miniature.

**Discuss:**
- Where did you and a neighbor disagree? Whose preferences would a real model learn?
- In Item 3, which response is most *aligned*? Is the most *helpful* answer the most *harmless* one?
- If labelers reward confident answers (Item 2, R1) over honest uncertainty (R2),
  what behavior does the model learn? (This is **how hallucination gets rewarded**.)
