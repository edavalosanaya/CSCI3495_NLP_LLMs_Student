# W11C2 Lab: How to Benchmark an LLM

Everything for this session is in **`lab.ipynb`**. A benchmark is three things: questions, an answer key, and a grader.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then open `weeks/week-11/class-02/exercise/lab.ipynb` in your editor, select the project's
**`.venv`** kernel, put your personal key where the first cell says `sk-trin-...`, and run the
cells from the top with Shift + Enter. The models run on the course server, so you need to be
online.

## What you will do

1. Get an answer back as data with structured outputs.
2. Write the benchmark loop. The `benchmark` cell is empty: write it as we go.
3. Change the method (prompt, tool, agent loop, bigger model) and build a leaderboard.
4. Grade open-ended answers with an LLM judge. The `judge` cell is empty too.
5. Swap the order of two answers and see whether the judge changes its mind.

## What is in this folder

| File | What it is |
|---|---|
| `lab.ipynb` | The session. |
| `data/arrows.csv` | 30 arithmetic word problems with their answers. |
| `data/judge_pairs.csv` | 8 open-ended questions, each with a good and a bad answer. |

## What you should see

| Method (qwen2.5-3b unless noted) | Accuracy |
|---|---|
| Direct answer | 43% |
| Direct answer, qwen2.5-7b | 57% |
| Step by step, then extract | 83% |
| Agent loop | 83% |
| Calculator tool | 100% |

Grading the same short answers, a gemma-3-12b judge passes 100% and a qwen2.5-3b judge passes 0%. Only 2 of 8 verdicts survive swapping the order.
