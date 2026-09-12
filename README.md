# CSCI 3495: Natural Language Processing & Large Language Models

Course materials for CSCI 3495. Everything you need for the semester lives here.

> **Can't `git pull`? Run this instead.**
>
> ```
> python scripts/update_course.py
> ```
>
> It throws away your edits to the course material and puts you on the newest
> version. Your `homeworks/` and `project/` folders are kept, and a dated copy
> of both is saved in `.backups/` every time. Same command on Windows, macOS,
> and Linux. See [Getting updates](#getting-updates).
>
> Don't have that file yet? Get it with
> `git fetch origin && git checkout origin/main -- scripts/update_course.py`

## What's in here

| Folder | What it is |
|---|---|
| `docker/` | The local LLM server (Ollama). Python is managed by uv, not Docker. |
| `weeks/week-NN/class-NN/lecture/` | The slide deck for each session (`slides.pptx`). |
| `weeks/week-NN/class-NN/exercise/` | The in-class lab: `lab.ipynb`, a short README, and any data. |
| `weeks/week-NN/class-NN/solutions/` | The worked answer to that exercise, plus a step-by-step `WALKTHROUGH.md`. |
| `homeworks/hw1` ... `hw6` | The six homework assignments, one notebook each. |
| `project/` | The semester project: spec, rubrics, and the proposal / checkpoint / report templates. |
| `syllabus/` | The syllabus. |
| `schedule/` | The week-by-week schedule with every due date. |

## Setup

Install [uv](https://docs.astral.sh/uv/), then from this folder:

```
uv sync
uv run python scripts/setup_data.py
docker compose -f docker/docker-compose.yml up -d ollama
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:0.5b
uv run python scripts/env_check.py
```

`uv sync` takes about a minute. Everything after it is instant.

## How you work

Open the labs with:

```
uv run jupyter lab
```

Jupyter will offer exactly **one** kernel, the course environment, so there is
nothing to choose. Every lab is a notebook that already runs the moment you open
it: run the cells from the top, then edit the `YOUR TURN` cells.

There is no test to run. Each task says in a comment what you should see when it
is right, and the answers are in the last cell of the notebook.

## About the in-class labs

The labs are **not graded**. They exist to be attempted, and every one carries
its own answers in its last cell, so you are never blocked. Try the task first;
if you are stuck for more than a few minutes, read the answer for *that task*
and keep going.

Homework is different: it is graded, submitted as the `.ipynb` with all cells
run, and its solutions are not here.

## Getting updates

Material is added and corrected through the semester. Before each class:

```bash
git pull
```

That works right up until you have edited a lab that also changed upstream, and
then git refuses to pull and starts talking about conflicts. This happens a lot:
you run cells in a notebook during class, and the same notebook gets corrected
that evening. You do not need to resolve any of it.

### The update script

From the course folder, on any operating system:

```
python scripts/update_course.py
```

On macOS and Linux, use `python3` if `python` is not found. If you have already
run `uv sync`, `uv run python scripts/update_course.py` works too.

What it does, in order:

1. Copies your `homeworks/` and `project/` folders into `.backups/<date-time>/`.
2. If you have made commits of your own, parks them on a branch named
   `my-work-<date-time>` so nothing you committed is lost.
3. Fetches the latest course material and **hard-resets the repo to it**, which
   discards every local change to labs, decks, solutions, and scripts.
4. Copies your `homeworks/` and `project/` folders back over the top.

So: your graded work survives, your notes in the labs do not. If you want to
keep something you wrote in a lab notebook, copy it somewhere outside this
folder first, or rename the file — a lab saved as `lab-my-notes.ipynb` is not a
course file and is left alone.

It works from any stuck state, including a half-finished merge full of
`<<<<<<<` markers. You never have to resolve a conflict by hand.

Nothing is ever deleted without a copy: `.backups/` keeps one dated folder per
run, and it is ignored by git, so it is yours to clean out whenever you like.
To get a file back off a rescue branch:

```bash
git show my-work-<date-time>:path/to/file.ipynb > recovered.ipynb
```

You are welcome to edit anything here for your own use, including the decks —
just expect the update script to reset those edits.

---

*Generated from the course source repository. Do not open pull requests against
this repo; raise anything you find in class or by email.*
