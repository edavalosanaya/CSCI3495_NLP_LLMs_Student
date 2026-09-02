# W13C2 Lab: Agent Evaluation Suite

## 1. Learning objective

Stop arguing about which agent design is better and measure it: run several
strategies over the same problems, pair the results, and rank them by success
AND by what they cost.

You write three functions in `eval_suite.py`: the run matrix, the paired
comparison, and the leaderboard. Scoring, the strategies and the problem set
are given.

## 2. Understanding the math

![Anatomy of an agent eval: task suite, run the agent, score each, leaderboard](../lecture/visuals/agent-eval.png)

Success rate is the obvious number, over $n$ problems:

$$\text{success} = \frac{1}{n}\sum_{i=1}^{n} \mathbf{1}[\text{solved}_i]$$

but it is not enough on its own. Two strategies can tie on success while one
spends five times the model calls, so the leaderboard ranks by success first
and then by cost:

$$\text{rank} = \big(-\text{success},\; \text{calls per task}\big) \quad \text{ascending}$$

Comparing two strategies means pairing them BY PROBLEM, not comparing two
aggregates. Only pairing tells you whether a wrapper recovered problems its
baseline missed or broke ones it had already solved:

$$\text{recovered} = |\{i : \text{wrapper}_i \land \lnot\text{base}_i\}|, \qquad \text{broke} = |\{i : \lnot\text{wrapper}_i \land \text{base}_i\}|$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-13/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `run_matrix`

Run every strategy over every problem and collect the results, so each cell is
comparable with every other.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 16 deselected
```

## 5. Implement `paired_wins`

Match runs BY PROBLEM ID and count recovered, broke and unchanged.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 16 deselected
```

## 6. Implement `leaderboard`

Rank by success rate, then by calls per task as the tie-break.

```bash
pytest -k step3 -q
```

```
..                                                                       [100%]
2 passed, 15 deselected
```

## 7. Run it, then question it

The full suite is 20 problems and takes over an hour on CPU. Use a slice in
class and run the whole thing once at the end:

```bash
python ../solutions/run_bench.py --n 4
```

```
model: qwen2.5:1.5b   problems: 4   strategies: Naive, CoT, Reflexion+CoT, ReAct, Reflexion+ReAct

rank strategy       success   solved  calls/task
1    Reflexion+CoT     100%      4/4         1.2
2    CoT               75%      3/4         1.0
3    Reflexion+ReAct    75%      3/4         6.8
4    ReAct             25%      1/4         5.0
5    Naive              0%       0/4         1.0

Reflexion lift, each variant against the baseline it wraps:
  Reflexion+CoT    100%  vs  CoT       75%   recovered:  1   broke:  0   unchanged:  3
  Reflexion+ReAct   75%  vs  ReAct     25%   recovered:  2   broke:  0   unchanged:  2

Baselines, paired by problem:
  CoT       vs Naive      CoT only:  3   Naive only:  0   same:  1
  ReAct     vs CoT        ReAct only:  1   CoT only:  3   same:  0
```

1. Compare rows 2 and 3. They tie at 75% success, and one spends 1.0 calls per
   task while the other spends 6.8. On success rate alone they are equal. Which
   would you deploy, and what does that say about single-number leaderboards?
2. ReAct scores 25%, worse than plain CoT at 75%, while costing five times as
   much. Giving the agent tools made it worse. Look at the paired row:
   `ReAct only: 1, CoT only: 3`. What kind of problem did tools actually help
   on, and what did they break?
3. Reflexion never broke anything: `broke: 0` in both pairs. Is that a property
   of Reflexion, or of this problem set being small and this run being lucky?
   Say what evidence would settle it.
4. Four problems is far too few to rank five strategies. Run `--n 8` and see
   which of the rankings above survive. Which conclusion would you be willing
   to write down after 4 problems, and which needs 20?
