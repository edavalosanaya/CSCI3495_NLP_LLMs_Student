# W7C2 Walkthrough: Measuring scaling, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `scaling.py` in this folder. Every code block below is taken
from it, and every printed value was produced by running it.

This session's in-class time goes to the compute-budget whiteboard exercise and
the structured debate; the code is a take-home lab or a quick demo. The teaching
value here is less about the four functions (they are short) and more about what
they expose regarding measurement.

---

## Given, `normalize`

```python
def normalize(text: str) -> str:
    return text.strip().strip(string.punctuation + " ").lower()
```

**Two `strip` calls, doing different jobs.** The first removes whitespace; the
second removes any character in `string.punctuation` **or space** from both ends.
The second is needed because a model answers `"Paris."` and the target is
`"paris"`.

**`strip` only touches the ends**, which is deliberate. Stripping punctuation
everywhere would turn "10 minus 3" into "10minus3" and break substring matching
in a different way.

```python
>>> normalize("  Paris.  ")
'paris'
```

---

## Given, `is_correct`

```python
def is_correct(model_output: str, target: str) -> bool:
    return normalize(target) in normalize(model_output)
```

**This is the most consequential line in the file, and it is a compromise.**

What it buys: a model that answers "7 days" when the target is "7", or "The
answer is 4", is graded correct. Small models are bad at obeying format
instructions, and a strict grader would score them near zero for reasons that
have nothing to do with whether they know the answer. Since the whole point is to
compare models, format noise would swamp the signal.

What it costs, and students should be able to name these:

- `is_correct("not 4", "4")` returns **True**. Negation defeats it entirely.
- `is_correct("17", "7")` returns **True**. Substrings of numbers match.
- A model that emits the entire alphabet would match any single-letter target.

**There is no neutral grader.** Lenient overcounts, strict undercounts, and an
LLM judge brings its own biases (W9C2). The honest move is to state which way
your grader errs and by how much, which is exactly what the third stretch goal
asks students to measure.

---

## Step 1, `accuracy`

```python
def accuracy(outputs: list[str], targets: list[str]) -> float:
    if not outputs:
        return 0.0
    correct = sum(is_correct(o, t) for o, t in zip(outputs, targets))
    return correct / len(outputs)
```

`sum` over booleans works because `True` is 1. The empty guard avoids a
`ZeroDivisionError` on an empty suite, which happens when every model is skipped
for not being pulled.

**Note it divides by `len(outputs)`, not by the number of pairs `zip` produced.**
If `outputs` and `targets` have different lengths, `zip` stops at the shorter one
and the accuracy is silently deflated. Not a problem in this exercise, but worth
seeing as the kind of quiet bug evaluation code attracts.

```python
>>> accuracy(["4", "Lyon", "7", "green", "six"], targets)
0.6
```

Three of five: it missed "Lyon" for Paris and "six" for 7. The "six" miss is a
*format* failure (the model said the right number in words), and lenient
substring grading does not rescue it. That distinction is question 2 of the
pre-baked exercise.

---

## Step 2, `scaling_trend`

```python
def scaling_trend(results: dict[str, float]) -> bool:
    vals = list(results.values())
    return all(vals[i] <= vals[i + 1] for i in range(len(vals) - 1))
```

**It relies on dict insertion order**, which Python has guaranteed since 3.7. The
caller is responsible for inserting models smallest-first, and nothing checks
that. Worth flagging as fragile: a student who builds the dict in a different
order gets a confidently wrong answer with no error.

**`<=` and not `<`.** Ties count as "scaling helped". With a 5-item suite, a tie
is the most likely outcome between adjacent model sizes, and demanding strict
improvement would report failure for pure noise. Choosing this *before* seeing
data is the discipline being taught; choosing it after would be exactly the kind
of analysis flexibility that makes published results unreproducible.

```python
>>> scaling_trend({"small": 0.6, "mid": 0.6, "large": 1.0})
True
>>> scaling_trend({"small": 1.0, "large": 0.6})
False
```

---

## Running it

```
small-model accuracy: 0.60
large-model accuracy: 1.00
accuracy non-decreasing with size? True
```

**Be explicit that these outputs are simulated.** `_demo` hard-codes what a small
and a large model "might say". It exercises the scoring core; it is not evidence
of scaling. A student who reports "I measured scaling and got 0.60 vs 1.00" from
this run has measured nothing.

The real measurement is `measure.py`, and it needs two models pulled.

---

## Teaching this session

**The suite was chosen, not found.** Simple arithmetic and strict-format short
answers were picked because a 0.5B model genuinely fails where a 3B model
succeeds, so the gap is visible with five questions. That is a legitimate choice
for a demo, and it is also exactly the kind of choice that makes benchmark
results hard to trust in general. Both halves of that sentence are worth saying.

**The MMLU warning is the most important paragraph in the README.** On a 4-choice
test, chance is 25%. With a handful of items, two models can both land near
chance and a student concludes "scaling does not work". They have discovered a
statistical power problem, not a fact about models. The chance-floor stretch goal
has students produce this failure deliberately, which is a much better lesson
than being warned about it.

**Connecting to the debate.** Schaeffer et al. 2023 argue that many claimed
"emergent abilities" are artifacts of discontinuous metrics: switch from exact
match to a continuous score and the sharp jump becomes a smooth curve. Students
who have just written both a lenient and a strict grader (stretch goal 3) and
watched the numbers move have the concrete version of that argument in hand.

**The honest summary for the class:** this lab measures scaling badly, on
purpose, at a scale you can run on a laptop. Kaplan and Chinchilla measured it
well, with many orders of magnitude of compute and careful loss curves. What
transfers is the shape of the reasoning, not the confidence of the conclusion.
