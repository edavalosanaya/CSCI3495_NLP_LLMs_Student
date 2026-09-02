# W13C2 Walkthrough: the evaluation harness

Step-by-step solutions for `exercise/README.md`. Code is copied verbatim from
`eval_suite.py` next to this file. The five strategies were given to you; everything
here is about measuring them honestly.

---

## Given, `is_correct`

```python
def is_correct(predicted: Optional[float], gold: float, tol: float = 1e-4) -> bool:
    if predicted is None:
        return False
    return abs(predicted - gold) <= max(tol, abs(gold) * tol)
```

**Common mistake:** `predicted == gold`. The strategies parse a float out of
prose, so a correct run can return `18.000000000000004`. You would spend the
lab debugging an agent that was right all along.

---

## Given, `evaluate_one`

```python
def evaluate_one(problem: Problem, name: str, fn: Callable, llm, **kw) -> Result:
    """Run one strategy on one problem. Reflexion gets the evaluator to react to."""
    if name.startswith("Reflexion"):
        def feedback(ans):
            if is_correct(ans, problem.answer):
                return True, "Correct."
            return False, (f"Wrong: you answered {ans}. Recheck each arithmetic "
                           "step with the calculator.")
        kw = {**kw, "feedback_fn": feedback}
    run = fn(problem.question, llm, **kw)
    return Result(problem.pid, name, run.answer,
                  is_correct(run.answer, problem.answer),
                  run.calls, run.steps, run.attempts, run.trace)
```

**The idea.** The `feedback` closure captures `problem.answer`, so Reflexion
gets told *what it got wrong* without ever seeing the gold answer itself. That
separation is the whole point: the agent reads the message, the harness holds
the truth.

**Match on the prefix, not the exact name.** There are two Reflexion entries,
`Reflexion+CoT` and `Reflexion+ReAct`, because Reflexion is a **wrapper** rather
than a rival: it takes a base strategy and adds attempt-feedback-retry around
it. `name == "Reflexion"` silently leaves both unevaluated, they degrade to
their base, and your leaderboard shows a lift of exactly zero with nothing
obviously broken. That is the nastiest bug in this lab because it looks like a
finding.

**Common mistake:** the opposite, giving *every* strategy the feedback function.
Then Naive and CoT retry too, and you are comparing five things that are all
secretly Reflexion.

---

## Step 1, `run_matrix`

```python
def run_matrix(problems: list[Problem], strategies: dict, llm,
    out: dict[str, list[Result]] = {}
    for name, fn in strategies.items():
        rows = []
        for p in problems:
            rows.append(evaluate_one(p, name, fn, llm))
            if progress:
                print(f"  {name:10s} {p.pid} {'ok' if rows[-1].correct else '. '}",
                      flush=True)
        out[name] = rows
    return out
```

---

## Given, `success_rate` and `avg_calls`

```python
def success_rate(results: list[Result]) -> float:
    return sum(r.correct for r in results) / len(results) if results else 0.0
```

```python
def avg_calls(results: list[Result]) -> float:
    return sum(r.calls for r in results) / len(results) if results else 0.0
```

**Why the empty guard matters:** `--n 0` or a filtered-to-nothing subset should
give you `0.0`, not a `ZeroDivisionError` in the middle of a 15-minute run.

---

## Step 2, `paired_wins`

```python
def paired_wins(a: list[Result], b: list[Result]) -> tuple[int, int, int]:
    by_b = {r.pid: r for r in b}
    a_only = b_only = same = 0
    for ra in a:
        rb = by_b.get(ra.pid)
        if rb is None:
            continue
        if ra.correct and not rb.correct:
            a_only += 1
        elif rb.correct and not ra.correct:
            b_only += 1
        else:
            same += 1
    return a_only, b_only, same
```

**The idea.** Pair by `pid`, never by index: the two lists are only in the same
order until someone adds a filter or runs the strategies concurrently.

This is also the function that measures the **Reflexion lift**. Run it on
`(Reflexion+X, X)` and the three numbers mean exactly: how many problems the
retry *recovered*, how many it *broke*, and how many it left alone. Two success
rates cannot distinguish "recovered 3, broke 0" from "recovered 5, broke 2",
and those are very different agents.

Real output from this lab's own run:

```
Reflexion vs ReAct   Reflexion only: 2   ReAct only: 0   same: 18
```

30% vs 20% on 20 problems is two problems, which is well inside the noise. "Won
2, lost 0" is the statement you can actually defend.

---

## Step 3, `leaderboard`

```python
def leaderboard(matrix: dict[str, list[Result]]) -> list[tuple[str, float, float]]:
    rows = [(name, success_rate(rs), avg_calls(rs)) for name, rs in matrix.items()]
    return sorted(rows, key=lambda r: (-r[1], r[2]))
```

**Why cost is in the sort key.** With accuracy alone, Reflexion (30%, 4.1 calls)
outranks CoT-like cheap strategies whenever it scrapes a point ahead. The
measured run makes the trap concrete: CoT wins on both axes at once, but a
leaderboard blind to cost would have crowned whatever retried the most.

---

## Running it

```
1  Reflexion+CoT     65%  13/20  1.5 calls
2  CoT               45%   9/20  1.0
3  Reflexion+ReAct   40%   8/20  9.3
4  ReAct             35%   7/20  5.7
5  Naive             10%   2/20  1.0

Reflexion+CoT     vs CoT     recovered 4, broke 0
Reflexion+ReAct   vs ReAct   recovered 2, broke 1
```

Three things to make sure students actually see:

1. **Reflexion lifted both baselines**, which is the point of running it as a
   wrapper rather than a rival. It is a layer, and the layer works.
2. **The two lifts are very different.** +20 points and nothing broken on CoT,
   for half a call; a net of one problem on ReAct, for 3.6 extra calls. A retry
   can only rescue problems the base was close on.
3. **CoT still beats ReAct** at a fifth of the cost. The calculator fixes
   arithmetic the model was mostly getting right anyway, and the tool-calling
   format costs it reasoning it does well. Compare with W12C2, where the tools
   were transformative because the model genuinely could not know the date.

If a student's numbers differ by a problem or two, that is worth a minute: at
temperature 0 the run should be reproducible, and small drifts usually mean the
Ollama server is batching differently, not that anything in the harness changed.
