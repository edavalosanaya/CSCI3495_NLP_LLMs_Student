# `gsm8k_mini.jsonl`, a 20-problem slice of GSM8K

**Source:** GSM8K (Grade School Math 8K), Cobbe et al., 2021, [arXiv:2110.14168](https://arxiv.org/abs/2110.14168),
<https://github.com/openai/grade-school-math>. **MIT License, Copyright (c) 2021 OpenAI.**

This is a real benchmark, not a toy one. GSM8K is the dataset the
chain-of-thought paper (Wei et al., 2022) reported its headline gains on, which
is exactly why it is the right yardstick here: you already know from Week 10
that CoT should beat a naive prompt on it, so if your harness says otherwise,
suspect the harness.

## What was changed

Nothing about the problems themselves. The slice was produced by:

1. taking the official **test** split (1319 problems),
2. shuffling with a fixed seed (3495) so the sample is reproducible and not
   cherry-picked,
3. keeping only **arithmetic-heavy** problems: ones whose reference solution
   multiplies or divides numbers big enough that doing it in your head is a
   real risk (see `arithmetic_heavy()` in `make_slice.py`). About a third of
   the test split qualifies, so this is a slice, not a hand-picked set,
4. keeping the first 20 of those whose question is under 320 characters, so a
   problem fits on a slide and a run finishes on a laptop CPU,
5. extracting the gold answer from the `#### <number>` line into a numeric
   `answer` field, and dropping the worked solution.

### Why filter for hard arithmetic?

This is a benchmark-design decision, and it is worth understanding rather than
just accepting. On `12 * 6` a 1.5B model does not need a calculator, so an easy
suite scores a tool-using agent and a mental-arithmetic agent the same and
tells you nothing about which to ship. **A benchmark that cannot separate the
systems you are comparing is not a neutral benchmark, it is a broken one.**

The honest way to state the result is therefore conditional: on GSM8K problems
with awkward arithmetic, tools and reflection earn their cost. Run
`make_slice.py` without the filter and rerun the suite if you want to watch
that conclusion change, which is itself the most useful thing in this folder.

Regenerate or resize it with `python make_slice.py --n 20`.

## Format

One JSON object per line:

```json
{"id": "G01", "question": "Julie operates the cash register exactly twice as fast as ...", "answer": 1050.0}
```

The worked solutions are deliberately **not** included: the agents must produce
the number themselves, and the evaluator compares numbers, not prose.

## Citing it

```bibtex
@article{cobbe2021gsm8k,
  title={Training Verifiers to Solve Math Word Problems},
  author={Cobbe, Karl and Kosaraju, Vineet and Bavarian, Mohammad and Chen, Mark
          and Jun, Heewoo and Kaiser, Lukasz and Plappert, Matthias and Tworek, Jerry
          and Hilton, Jacob and Nakano, Reiichiro and Hesse, Christopher and Schulman, John},
  journal={arXiv preprint arXiv:2110.14168},
  year={2021}
}
```
