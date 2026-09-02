# W10C2 Lab: Chain-of-Thought & Self-Consistency

## 1. Learning objective

Make a small model better at arithmetic without touching its weights: ask it to
reason step by step, then take a majority vote over several attempts.

You write two functions in `cot_lab.py`: the majority vote and the evaluation
loop. Answer extraction and the two prompt builders are given.

## 2. Understanding the math

![Chain-of-thought vs direct prompting: the worked solution in the prompt flips the answer from 27 to 9](../lecture/visuals/assets/wei-2022-fig-1.png)

Both prompts get the same question and the same model. Only the instruction
differs, and it is scored by exact match on the final integer:

$$\text{accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{a}_i = a_i]$$

![Self-consistency: sample several chains at temperature > 0 and take the majority answer](../lecture/visuals/assets/wang-2022-fig-1.png)

Self-consistency samples $n$ chains and keeps whichever final answer the most
of them reached, so one bad chain no longer decides the outcome:

$$\hat{a} = \arg\max_{a} \sum_{j=1}^{n} \mathbf{1}[\hat{a}^{(j)} = a]$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-10/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `majority_vote`

Drop the chains that produced nothing, count the rest, and break ties by the
smallest value so the result is reproducible.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

## 5. Implement `evaluate`

Prompt, generate, extract, compare, divide.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

## 6. Run it, then question it

```bash
python cot_lab.py
```

```
[model] using Ollama model 'qwen2.5:0.5b'

prompt style      accuracy
--------------------------
direct                33%
chain-of-thought     100%
```

Same model, same questions, three times the accuracy from one sentence of
instruction.

1. Read a wrong answer. Print the direct reply for the first question, "A cafe
   had 23 muffins, sold 17, then baked 12 more." The model writes
   `23 - 17 + 12 = 8.` It set up the correct equation and then got 8 instead of
   18. What does that tell you about where the failure actually is?
2. Watch `extract_answer` in that same reply. It takes the LAST integer in the
   text, which here is the wrong one. Construct a chain-of-thought reply where
   taking the last integer would score a correct answer as wrong.
3. Test the vote's tie-break. `majority_vote([7, 9])` returns 7 and
   `majority_vote([None, 5, 5])` returns 5. Why does a tie need a rule at all,
   and what would go wrong with `Counter.most_common(1)` alone?
4. Self-consistency needs temperature above 0, or every chain is identical.
   Explain why that makes the technique cost n times as much, and when you
   would spend that.
