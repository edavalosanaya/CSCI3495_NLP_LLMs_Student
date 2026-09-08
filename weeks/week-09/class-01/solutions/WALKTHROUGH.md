# W9C1 Walkthrough: Decoding strategies, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The lab is `exercise/lab.ipynb`. Everything below was produced by running it, so
the numbers are what you will actually see. `decoding.py` in this folder is a
separate thing: a from-scratch implementation of the same maths in plain Python
with no model, for the student who wants to see under `top_p`.

---

## Part 1, the model scores words

One forward pass, one score per word in the vocabulary. The prompt is
`"The best thing about living in a small town is"` and distilgpt2's ten
favourite continuations are:

```
that   0.496     to     0.032     its    0.011
you    0.099     it     0.032     your   0.011
the    0.093     not    0.019
how    0.034     being  0.014
```

**TRY IT 1.** Those ten hold **0.841** of the probability between them. That
leaves **0.159** spread across the other **50,247** words. About a sixth of
every draw comes out of a tail you never see on the chart, which is the whole
reason top-k and top-p exist.

---

## Part 2, greedy and beam

Greedy takes the argmax at every step. Beam keeps several partial sequences
alive and scores each as a whole.

**TRY IT 2.** Widening the beam does not monotonically improve anything:

| width | score | what it produces |
|---|---|---|
| 2 | -0.541 | loops on "It's a place where you can live." |
| 4 | -0.681 | the only one that does not loop |
| 8 | -0.475 | loops again, on a different phrase |

A wider beam searches harder for a high-scoring sequence, and **a repeated
phrase is high-scoring**. Beam search is not a fix for repetition; at width 8 it
is worse than at width 4. That is the setup for Part 4.

(The score is a length-normalised mean log probability, which is why the
ordering by score does not match the ordering by width. Do not go deeper than
that in class.)

---

## Part 3, the three knobs

`reshape()` applies temperature, then top-k, then top-p to the same logits:

```
plain softmax     top word 0.496   words with any chance: 39,616
temperature 0.5   top word 0.915   words with any chance:    576
temperature 1.5   top word 0.139   words with any chance: 50,178
top-k 10          top word 0.590   words with any chance:     10
top-p 0.9         top word 0.551   words with any chance:     20
```

Temperature changes the shape and leaves everything on the table. Top-k and
top-p take words off the table.

**TRY IT 3.** Top-p survivors on this distribution:

| p | words kept |
|---|---|
| 0.50 | 2 |
| 0.90 | 20 |
| 0.99 | 1,168 |

Ship 0.9. At 0.5 the nucleus is two words wide, so on a step where the model is
genuinely unsure you have thrown its uncertainty away and the text goes generic.
At 0.99 you have kept most of the junk tail back.

---

## Part 4, after the break

The prompt changes to `"In my opinion, the most important part of learning to
code is"` and greedy derails:

```
... I am not a programmer. I am not a programmer. I am not a programmer. ...

repeated words: 41 out of 62
```

**YOUR TURN 1**, get it under 5. Measured:

| setting | repeats |
|---|---|
| greedy, unchanged | 41 / 62 |
| `do_sample=True, temperature=0.7` | 44 / 65 |
| `do_sample=True, top_k=10` | 27 / 64 |
| `do_sample=True, top_p=0.9` | 12 / 59 |
| `do_sample=True, temperature=1.5` | 2 / 54 |
| `do_sample=False, repetition_penalty=1.2` | 3 / 64 |

Two things worth saying out loud. Sampling at 0.7 is **worse than greedy**, so
"add some randomness" is not automatically the fix. And temperature 1.5 gets the
count down by making the text bad, which is not the same as solving the problem.

**YOUR TURN 2**, do it without sampling:

```python
text = generate(do_sample=False, repetition_penalty=1.2)   #  3 / 64
text = generate(do_sample=False, repetition_penalty=1.5)   #  0 / 42
```

The penalty divides the score of every word already generated, so the argmax
stops being a fixed point. The decoding is still deterministic: run it twice and
you get the same sentence, which sampling can never promise. That is why code
and structured output are generated this way.
