# W5C1: From a crime scene to a search engine

A body in Lauriston Gardens, and six things noted beside it:

> **mud, rain, tobacco, cab, boots, bruise**

Baker Street keeps a casebook of a hundred prior cases, each one a list of what
was observed at it. Somewhere in it is the case that most resembles tonight's.
**No case file notes all six**, and each word on its own returns twenty to fifty
cases, so there is nothing to search for. Finding it is the whole lab.

Everything is in **`lab.ipynb`**. We work through it together, cell by cell. Each
part ends with a `TRY IT`: one question, one empty cell. After the break you get
twenty-five minutes and a race.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-05/class-01/exercise/lab.ipynb` in your editor, select the
project's **`.venv`** kernel, and run the cells from the top with Shift + Enter.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | Why Ctrl+F fails, then tf, idf and cosine similarity by hand, then all 100 case files. Three `TRY IT` checkpoints, two `YOUR TURN` tasks, and the answers. |
| `data/casebook.csv` | 100 case files: an id, a title, and what was observed. Only the observations are searched, never the title. |

## How this lab works

Everything already runs the moment you open it. Nothing raises, there is no test
to run and nothing to submit. Every `YOUR TURN` says in a comment what you should
see when it is right, in real numbers, and the answers to everything are in the
last cell.

The last part is a race. Work in your team, and do not read the answers cell until
your instructor calls time.
