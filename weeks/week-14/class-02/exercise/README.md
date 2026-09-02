# W14C2 Lab: Project Work & Reproducibility Check

## 1. Learning objective

Push your project forward with an instructor checkpoint, and run a
reproducibility check over the code you are about to hand in.

There is nothing to implement. `repro_check.py` is written; you point it at
your own files and act on what it says.

## 2. Understanding the math

![Chained 90 percent reliable steps decay: 0.9, 0.81, 0.73, 0.66, 0.59](../lecture/visuals/why-expensive.png)

Reliability multiplies across chained LLM steps. With per-step success $p$ over
$n$ steps, a pipeline that feels fine at each step is not fine end to end:

$$
P(\text{whole task succeeds}) = p^{n},
\qquad \text{e.g. } 0.9^{5} \approx 0.59
$$

![Decision checklist: prompt, then augmented LLM, then workflow, and only last an agent](../lecture/visuals/decision-checklist.png)

That is the argument for the checklist: reach for a plain prompt first, then an
augmented LLM, then a fixed workflow, and only last an agent that decides its
own control flow.

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-14/class-02/exercise course bash
```

## 4. Check a file that is reproducible

```bash
python repro_check.py ../../../week-04/class-01/solutions/mlp_classifier.py
```

```
Reproducibility check: ../../../week-04/class-01/solutions/mlp_classifier.py
  [OK]   parses cleanly.
  [OK]   found seeding: torch.manual_seed
  Note: run your real entry point in Docker too; this is only a static hint.
```

## 5. Check one that is not

```bash
python repro_check.py ../../../week-02/class-01/solutions/ngram_lm.py
```

```
Reproducibility check: ../../../week-02/class-01/solutions/ngram_lm.py
  [OK]   parses cleanly.
  [WARN] no RNG seeding found, set seeds for reproducible results.
  Note: run your real entry point in Docker too; this is only a static hint.
```

## 6. Check your own project

```bash
python repro_check.py /workspace/<path to your project file>
```

Run it over every file you intend to submit. Fix each `[WARN]` before the
checkpoint, or be ready to say why it does not apply.

## 7. The checkpoint conversation

Bring answers to these. They are the same questions the final rubric asks.

1. Run the numbers on your own pipeline. How many LLM calls are chained end to
   end, and what is $p^{n}$ if each is 90% reliable? If that number is
   uncomfortable, which step would you delete first?
2. Walk down the decision checklist for your project. What is the simplest
   design that would still meet your goal, and what specifically does your
   current design buy over it?
3. `repro_check.py` is a static hint: it greps for seeding and never runs your
   code. Name a way your project could be irreproducible that this tool cannot
   possibly detect.
4. State the one result you will show in the final presentation, and the
   command a reader would run to reproduce it.
