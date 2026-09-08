# W4C2 Lab: Tensors, autograd, and fitting a line

Everything for this session is in **`lab.ipynb`**. Parts 1 to 3 we walk through
together, interleaved with the slides; after the break the notebook is yours.
Each part ends with a `TRY IT`: one question, one empty cell.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-04/class-02/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.
There is nothing else to install.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | Tensors, autograd, and a straight line fitted to real data by gradient descent. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `data/reviews.csv` | 240 movie reviews, with a star rating and a genre. |
| `images/` | The figures from the lecture, so the notebook stands on its own. |

## How this lab works

Everything already runs the moment you open it. Nothing raises, and there is no
test to run and nothing to submit. Every `YOUR TURN` says in a comment what you
should see when it is right, in real numbers, and the answers to everything are
in the last cell of the notebook.

The two tasks each change one thing and redraw the fit. Predict what the change
will do before you run it.
