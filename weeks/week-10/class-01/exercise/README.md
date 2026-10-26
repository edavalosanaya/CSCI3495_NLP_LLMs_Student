# W10C1 Lab: Teaching a Tiny Assistant Manners

Everything for this session is in **`lab.ipynb`**. SFT imitates. DPO compares.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-10/class-01/exercise/lab.ipynb` in your editor, select the project's
**`.venv`** kernel, and run the cells from the top with Shift + Enter. There is
nothing else to install, and everything runs on a laptop CPU in about two minutes.

## What you will do

1. Ask a pretrained model (distilgpt2) a question, and see that it does not answer.
2. Fine-tune it on demonstrations, half of which are rude. The SFT training cell is
   empty: write it as we go.
3. Train it with DPO on pairs of a polite (chosen) and a rude (rejected) answer.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | The session. |
| `data/sft.csv` | 80 demonstrations: 40 questions, each answered once politely and once rudely. |
| `data/prefs.csv` | 40 preference pairs: the same questions, a polite `chosen` and a rude `rejected` answer. |
| `data/test_prompts.txt` | 10 questions that appear in neither file. Every score is measured on these. |

## What you should see

`rude_rate` is the share of 100 sampled answers (10 per test question) that open with "Ugh".

| After | `rude_rate` |
|---|---|
| nothing (the base model) | 0.0, because it never answers at all |
| SFT | 0.36 |
| DPO | 0.0 |

The facts can still be wrong after DPO ("A duck says gobble."). DPO taught the model manners, not knowledge.
