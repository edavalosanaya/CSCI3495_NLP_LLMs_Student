# W8C1 Lab: How a Model Chooses the Next Word

Everything for this session is in **`lab.ipynb`**. We work through it together in
class, cell by cell. The 2 `YOUR TURN` cells are the parts you edit.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-08/class-01/exercise/lab.ipynb` in your editor, select the project's
**`.venv`** kernel, and run the cells from the top with Shift + Enter. There is
nothing else to install.

## What you will do

1. Read the distribution a real GPT outputs for one prompt.
2. Run greedy and beam search and compare their scores.
3. Watch temperature, top-k and top-p reshape that distribution.
4. Break greedy on purpose, then fix it.

Parts 1 to 3 we walk through together, interleaved with the slides. Part 4 is
the work after the break.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | The session: notes, working code, 3 `TRY IT` questions, 2 `YOUR TURN` tasks, and the answers. |
| `images/` | The two lecture figures the notebook shows inline. |

## How this lab works

Everything already runs the moment you open it. Nothing raises, and there is no
test to run and nothing to submit. Every `YOUR TURN` cell says in a comment what
you should see when it is right, in real numbers. A `TRY IT` is a question with
an empty cell under it. Every answer is in the last cell of the notebook.

Read the output and judge whether it looks right. That is the skill.
