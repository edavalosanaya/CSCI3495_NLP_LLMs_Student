# W5C1: From a crime scene to a search engine

**There is no code in this folder, and `lab.ipynb` is empty.** Making the cells
is the first thing you do. We write the whole thing together: it goes up on
screen, you type it, and by the end you have a search engine. Nothing is
pre-written, so nothing is skipped past.

## The case

A body in Lauriston Gardens, and six things noted beside it:

> **mud, rain, tobacco, cab, boots, bruise**

Baker Street's casebook holds a hundred prior cases, each a list of what was
observed at it. One of them is the same hand. **No case file notes all six**, and
each word alone returns twenty to fifty cases, so there is nothing to search for.
Finding it is the lab.

## Getting started

Once, from the repository root:

```
uv sync
```

Then, from this folder:

```
uv run jupyter lab
```

Open `lab.ipynb`, pick the project's **`.venv`** kernel, and put your hands on the
keyboard.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | Empty. You fill it. |
| `data/casebook.csv` | 100 case files: an id, a title, and what was observed. |

Only the observations are ever searched, never the title, so no answer is
something you could have found by searching for it.

## What you should see as we go

So you can tell you are on track without anyone looking at your screen:

| after | you should see |
|---|---|
| loading the casebook | `(100, 3)` |
| counting each observation | 34, 24, 27, 22, 47, 45 |
| tf on case 1 | every observation `0.167` |
| idf | `boots` and `london` exactly `0.000` |
| tf-idf on case 1 | `creosote` `0.2310`, everything common `0.0000` |
| the hundred, ranked | The Brixton Road affair `0.687` |

## After the break: the second body

**You get no code for this one.** Same job, new scene, written from scratch with
what you typed before the break. Seven or eight lines is the whole answer.

> **A counting-house in Threadneedle Street. Noted at the scene:**
> `ledger clerk tobacco london ink`
>
> **a.** Which case in the book is the same hand?
> **b.** Which of those five observations is doing the least work?

> When it is right: the top three are **0.719**, **0.613** and **0.603**, close
> enough that (a) has to be argued rather than read off. For (b), one of the five
> has an idf of **1.18** because it is noted at 83 of the hundred crimes.

Nothing to submit.
