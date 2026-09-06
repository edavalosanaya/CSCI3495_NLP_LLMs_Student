# W5C2: One LSTM, three jobs

Everything for this session is in **`lab.ipynb`**. We work through it together in
class, cell by cell. Each part ends with a `TRY IT`: one question, one empty
cell. After the break you train a sarcasm detector, in teams.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-05/class-02/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.
Everything you need is in this folder, so there is nothing to download.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | What `nn.LSTM` returns, then the same layer doing three different jobs, then your own sarcasm detector. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `data/headlines.csv` | 26,602 news headlines, half of them satirical. |
| `data/generator.pt` | A trained word-level language model (job 1). |
| `data/tagger.pt` | A trained part-of-speech tagger (job 2). |
| `data/classifier.pt` | A trained sarcasm classifier (job 3). |
| `images/` | The figures from the lecture, so the notebook stands on its own. |

We are **using** LSTMs today, not building one. The three models are already
trained, so every cell in the walkthrough runs in about a second.

## The data

The headlines are the **News Headlines Dataset for Sarcasm Detection** (Misra &
Arora): sarcastic headlines from *The Onion*, genuine ones from *HuffPost*. Real
published writing, so it is occasionally rude; the generator is stopped from
producing the worst of it.

The tagger was trained on the Penn Treebank sample distributed with NLTK.

## How the competition works

You get a working LSTM that is barely better than guessing, and about
twenty-five minutes. One training run takes roughly a second, so try a lot of
things. What you change is the **model**, not a settings file: make it wider,
read the sentence backwards as well as forwards, stack another layer, take a
different vector out of the LSTM.
