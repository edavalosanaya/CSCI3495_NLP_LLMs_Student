# W10C1 Lab: Prompt Engineering

## 1. Learning objective

Treat prompting as an experiment: build few-shot prompts, run them against a
real local model at temperature 0, and measure which wording actually helps.

You write two functions in `prompt_lab.py`: the few-shot prompt builder and
the experiment loop. Parsing, accuracy and the model backends are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-10/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `build_fewshot_prompt`

![Anatomy of a few-shot prompt: instruction, demonstrations, query, output cue](../lecture/visuals/prompt-anatomy.png)

A few-shot prompt is four parts: an instruction, some demonstrations, the
query, and an output cue the model completes. The cue is the part people
forget, and without it the model has no signal about what shape to answer in.

Instruction, blank line, each demonstration as a review/sentiment pair, then
the query and a bare `Sentiment:` cue.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 5 deselected
```

## 4. Implement `run_experiment`

![The experiment loop: prompt variant, model at temperature 0, parse, score](../lecture/visuals/experiment-loop.png)

Everything runs at temperature 0, so a difference between two variants is a
difference in the prompt, not in the sampling:

$$\text{accuracy} = \frac{1}{N} \sum_{i=1}^{N} \mathbf{1}[\hat{y}_i = y_i]$$

Build a prompt per item, generate, parse, and score.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 5 deselected
```

## 5. Run it, then break it

```bash
python prompt_lab.py
```

```
[model] using Ollama model 'qwen2.5:0.5b'

variant                   words  accuracy
-----------------------------------------
zero-shot (baseline)         17      75%
few-shot (baseline)          30      88%
few-shot + constraint        33     100%
zero-shot + constraint       20     100%
golf: 'One word only.'       11     100%
golf too far                  8      12%
```

Eleven words score 100% and eight score 12%. Everything below is deterministic
at temperature 0, so a change in the number is a change in the prompt.

1. Reverse the two demonstrations. Pass `list(reversed(DEMOS))` to
   `run_experiment`. Accuracy moves from 88% to 100% with identical content in
   a different order. What does that do to your confidence in an 88% vs 100%
   comparison anywhere else in the table?
2. Remove the demonstrations entirely, passing `[]`. Few-shot drops to 75%,
   which is exactly the zero-shot baseline. Then try demos that are all
   `positive`: still 88%. Which is doing more work here, the demonstrations'
   content or their format?
3. Read the two golf rows. Going from 11 words to 8 costs 88 points of
   accuracy. Find the deleted words in `build_bare_prompt` versus `GOLFED` and
   say which one was load-bearing.
4. `zero-shot + constraint` reaches 100% in 20 words while `few-shot +
   constraint` needs 33 for the same score. On this dataset the demonstrations
   are dead weight. Construct a task where you would expect the opposite.
