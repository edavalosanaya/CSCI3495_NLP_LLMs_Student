# W6C1: Taking a transformer apart with your hands

Everything for this session is in **`lab.ipynb`**. We work through it together
before the break. After the break you experiment on two real language models, in
teams.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
uv run python scripts/setup_data.py
```

The second command downloads the two models this lab uses. Do it **before
class**, not during it.

Then open `weeks/week-06/class-01/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | Tokens to vectors, the attention weights drawn as a heatmap, the same word in two sentences, then the experiments. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `images/` | The figure from the lecture, so the notebook stands on its own. |

The models are **DistilBERT** (an encoder, 66M parameters) and **DistilGPT-2** (a
decoder). Neither has been fine-tuned or filtered, which is exactly why their raw
behaviour is worth looking at.

## What we are not doing

We are not implementing attention. We are loading a trained transformer and
taking it apart: reading its attention weights, watching a word's vector change
with its sentence, and finding out what it does and does not know.

## The rule for the second half

A claim about a model comes with the prompt that produced it and the numbers it
returned. "It is biased" is not a finding. "`the doctor finished [MASK] shift`
gives *his* 0.44 and *her* 0.10" is a finding.
