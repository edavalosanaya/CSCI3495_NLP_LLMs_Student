# W8C2: BERT in code

Everything for this session is in **`lab.ipynb`**. We work through Parts 1 to 3
together before the break. After the break you fine-tune the model, in pairs.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
uv run python scripts/setup_data.py
```

The second command downloads the model this lab uses. Do it **before class**,
not during it.

Then open `weeks/week-08/class-02/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | The input BERT reads, filling in a `[MASK]`, the `[CLS]` vector as a sentence, then fine-tuning. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `data/headlines.csv` | 26,000 news headlines, labelled sarcastic or not. The lab uses 4,000 of them. |

The model is **DistilBERT** (an encoder, 66M parameters). It is a BERT trained to
be smaller, and it behaves like one.

## What we are not doing

We are not implementing a transformer. We are loading a trained one and using it
the way anybody actually does: read its input, read its output, then move its
weights.

## The number that matters

A frozen DistilBERT plus a logistic regression scores **0.827** on this task,
having learned nothing about sarcasm. Everything after the break is measured
against that.
