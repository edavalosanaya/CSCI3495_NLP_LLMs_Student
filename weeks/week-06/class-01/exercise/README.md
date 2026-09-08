# W6C1: A neural classifier, and a competition to improve it

Everything for this session is in **`lab.ipynb`**. We build the classifier
together before the break, cell by cell, and it will not be very good. After the
break it is yours to improve, in teams, against a public scoreboard.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
uv run python scripts/setup_data.py
```

The second command downloads the 2,000 film reviews this lab is scored on. It
takes a few seconds and only has to happen once, so **do it before class**, not
during it.

Then open `weeks/week-06/class-01/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | Pooling word embeddings, an MLP, training, F1, then the competition. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `data/glove-50d-20k.npz` | Real GloVe vectors, for the `pretrained` setting. |
| `images/` | The figures from the lecture, so the notebook stands on its own. |

The corpus is the **movie review polarity dataset** (Pang & Lee, ACL 2004), 2,000
labelled reviews, fetched through NLTK rather than shipped here.

## How the competition works

Three splits, and the difference matters:

- **train** (1,200) is what the model learns from,
- **validation** (300) is what you tune against, as often as you like,
- **test** (500) is scored **once**, at the end, by everyone at the same time.

Choosing a model by looking at the test set is how you fool yourself into
reporting a number that will not survive contact with real data. That is why the
last cell exists and why you only run it when time is called.
