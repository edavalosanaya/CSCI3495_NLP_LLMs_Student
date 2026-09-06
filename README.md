# CSCI 3495: Natural Language Processing & Large Language Models

Course materials for CSCI 3495. Everything you need for the semester lives here.

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

If `git pull` complains that your local changes would be overwritten, you have
edited a file that also changed upstream. Commit or stash your work first:

```bash
git stash        # set your changes aside
git pull
git stash pop    # put them back
```

You are welcome to edit anything here for your own use, including the decks.

---

*Generated from the course source repository. Do not open pull requests against
this repo; raise anything you find in class or by email.*
