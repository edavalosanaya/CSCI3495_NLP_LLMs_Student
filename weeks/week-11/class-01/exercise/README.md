# W11C1 Lab: LoRA on a GPT

Everything for this session is in **`lab.ipynb`**. Freeze the GPT, train two skinny matrices.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-11/class-01/exercise/lab.ipynb` in your editor, select the project's
**`.venv`** kernel, and run the cells from the top with Shift + Enter. There is
nothing else to install, and everything runs on a laptop CPU in under two minutes.

## What you will do

1. Ask a pretrained GPT (distilgpt2) to label a movie review, and see that it cannot.
2. Add LoRA to its attention layers. The `LoRA` class cell is empty: write it as we go.
3. Train a sentiment adapter, save it, train a genre adapter on the same frozen GPT, swap them, and merge one back in.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | The session. |
| `data/sentiment.csv` | 240 movie reviews labeled `positive` or `negative`, with a train/test split. |
| `data/genre.csv` | 260 movie reviews labeled `horror` or `comedy`, with a train/test split. |

## What you should see

| | Result |
|---|---|
| distilgpt2 | 81,912,576 weights, 328 MB |
| LoRA, r = 8 | 147,456 trainable weights, 0.18% |
| before training | 0% on the sentiment test set |
| sentiment adapter | 98% on test, a 0.59 MB file |
| genre adapter | 98% on test |
