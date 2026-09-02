# CSCI 3495: Natural Language Processing & Large Language Models

Course materials for CSCI 3495. Everything you need for the semester lives here.

## What's in here

| Folder | What it is |
|---|---|
| `docker/` | The course environment. Set this up in week 1 and everything else just runs. |
| `weeks/week-NN/class-NN/lecture/` | The slide deck for each session (`slides.pptx`). |
| `weeks/week-NN/class-NN/exercise/` | The in-class exercise: starter code, instructions, and any data. |
| `weeks/week-NN/class-NN/solutions/` | The worked answer to that exercise, plus a step-by-step `WALKTHROUGH.md`. |
| `homeworks/hw1` ... `hw6` | The six homework assignments, with starter code and the tests they must pass. |
| `project/` | The semester project: spec, rubrics, and the proposal / checkpoint / report templates. |
| `syllabus/` | The syllabus. |
| `schedule/` | The week-by-week schedule with every due date. |

## Setup

Follow `docker/README.md`. It takes about fifteen minutes the first time and
nothing after that.

## How you work

You work **inside the container**, not on your own machine, so every command is
the same on macOS, Linux and Windows. Each lab and homework README opens with
one `docker compose ... bash` line that drops you into a shell already sitting
in that folder. After that the commands are short and you type them, not paste
them: `pytest -k step1 -q`, `python text_tools.py`.

## About the in-class exercises

The labs are **not graded**. They exist to be attempted, so the worked solution
and a step-by-step walkthrough ship next to every one of them, in `solutions/`.
Try the step first; if you are stuck for more than a few minutes, read the
walkthrough for *that step* and keep going. Homework is different: it is graded,
and its solutions are not here.

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
