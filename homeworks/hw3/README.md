# HW3: Attention and the Transformer Block

**Out:** Mon Oct 12 · **Due:** Mon Oct 19 · **100 points** · individual work

Everything is in **`hw3.ipynb`**. There are 6 `YOUR TURN` cells,
each worth the points marked in its banner.

## Getting started

Once, from the repository root:

```
uv sync
```

Then open `homeworks/hw3/hw3.ipynb`, select the project's **`.venv`** kernel, and
run the cells from the top.

## What you will do

1. Implement scaled dot-product attention from the equation.
2. Add a causal mask and prove no position sees the future.
3. Assemble a block and explain why the shape has to survive it.

## How to submit

Submit the `.ipynb` with **every cell run and its output visible**. A notebook
submitted with empty outputs cannot be marked, because the output is the
evidence that your code ran.

Submit **`ANSWERS.md`** with it. That is where your written answers go, along
with the AI-use disclosure the course policy requires.

There is no test suite. You are marked on the code working and on the short
written answers, which are worth a large share of the points: several questions
give more marks for the explanation than for the code.

## If you get stuck

The last cell of the notebook lists the marking scheme and the mistakes students
most often make on this assignment. Read it before you start, not after.
