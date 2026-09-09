# W10C1 Lab: Byte Pair Encoding, and What It Costs

Everything for this session is in **`lab.ipynb`**. Parts 1 to 3 we work through
together during the lecture; Part 4 is the team bake-off after the break.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-10/class-01/exercise/lab.ipynb` in your editor, select the project's
**`.venv`** kernel, and run the cells from the top with Shift + Enter. There is
nothing else to install.

## What you will do

1. Watch a real tokenizer break your own words apart.
2. Run the merge loop that builds a vocabulary from nothing.
3. See what an instruction-tuned model adds to that vocabulary.
4. With your team: train a tokenizer on one corpus and score it on everyone else's.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | The session: notes, working code, 3 `TRY IT` checkpoints, 3 `YOUR TURN` tasks, and the answers. |
| `data/corpora.py` | The five bake-off corpora, each with a training text and a held-out test text. |

## How this lab works

Everything already runs the moment you open it. Nothing raises, and there is no
test to run and nothing to submit. Every `YOUR TURN` cell says in a comment what
you should see when it is right, in real numbers, and the answers are in the
last cell of the notebook.

Read the output and judge whether it looks right. That is the skill.
