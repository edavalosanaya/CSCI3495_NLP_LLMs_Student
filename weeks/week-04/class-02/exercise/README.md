# W4C2: Word embeddings, and the arithmetic that made them famous

Everything for this session is in **`lab.ipynb`**. We work through it together in
class, cell by cell. Each part ends with a `TRY IT`: one question, one empty
cell. After the break you plan a strategy on a budget, in teams.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-04/class-02/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | A word as 50 numbers, nearest neighbours, `king - man + woman`, and the same line with `doctor` in it. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `data/glove-50d-20k.npz` | Real GloVe vectors: the 20,000 most frequent words, 50 dimensions each. |
| `images/` | The figures from the lecture, so the notebook stands on its own. |

The vectors are **GloVe 6B 50d** (Pennington, Socher & Manning, EMNLP 2014),
trained on Wikipedia 2014 and Gigaword 5 and released under the Public Domain
Dedication and License. Nobody labelled them, and nobody chose what the 50
numbers mean, which is the whole point of the class.

## How this lab works

Everything already runs the moment you open it. Nothing raises, and there is no
test to run and nothing to submit. Every `YOUR TURN` says in a comment what you
should see when it is right, in real numbers, and the answers to everything are
in the last cell of the notebook.
