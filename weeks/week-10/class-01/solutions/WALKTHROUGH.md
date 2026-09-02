# W10C1 Walkthrough: Prompt experiments, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `prompt_lab.py` in this folder. Every printed value was
produced by running it against `qwen2.5:0.5b` through Docker.

---

## Orientation

`parse_label` and `StubModel` ship written. One detail in `parse_label` matters
all lab:

```python
    if pos and (not neg or pos.start() < neg.start()):
        return "positive"
```

**First signal wins.** "It is good, not bad" parses as positive. That is a
deliberate simplification and a real source of grading error: a model that hedges
("this is not good") gets graded on the wrong word. When a student's accuracy
looks impossible, the parser is a suspect alongside the prompt.

---

## Steps 1 and 2, `build_fewshot_prompt` and `run_experiment`

These are short and mechanical: exact-match accuracy, a template assembler, and a
loop that ties them together. Two things are worth insisting on.

**The trailing output cue.** `build_fewshot_prompt` must end with `Sentiment:`
and nothing after it. The test asserts `p.rstrip().endswith("Sentiment:")`. This
is not cosmetic: the cue is what makes the completion a label rather than a
continuation of the review.

**Fixed decoding in `run_experiment`.** `OllamaModel.generate` pins
`temperature: 0, seed: 0, num_predict: 8`. That is what makes a score difference
attributable to the prompt. Students who change decoding mid-experiment are
comparing two things at once, which is the exact failure the lab is teaching them
to avoid.

---

## Running it

The baselines, measured (not estimated), on `qwen2.5:0.5b` at temperature 0:

```
variant                   words  accuracy
zero-shot (baseline)         17       88%
few-shot (baseline)          30       50%   <- MORE demos, WORSE score
few-shot + constraint        33      100%
zero-shot + constraint       20      100%
golf: 'One word only.'       11      100%
golf too far                  8       12%
```

**The baseline few-shot prompt scores worse than no demonstrations at all**, and
students will assume they have a bug. They do not. Here is the diagnosis, and it
is the whole point of the session:

The failure is **format, not sentiment**. Asked to "classify the movie review's
sentiment", the model begins a sentence: "The sentiment of this movie review
is..." With `num_predict=8` that gets truncated before the label arrives, so
`parse_label` returns `unknown`. In the few-shot layout it does something worse:
it latches onto "Positive" and returns it for all eight items, scoring exactly
50% on a balanced set.

**The fix is one clause pinning the output shape.** Adding "Answer with one word:
positive or negative." takes few-shot from 50% to **100%**. Adding demonstrations
does not help; constraining the output does.

**Then it golfs remarkably far.** The shortest prompt that still holds 100% is
`"One word only."` at **11 words** including the review and cue. At 8 words (the
cues alone, no instruction) it collapses to **12%**, because the model stops
producing labels at all and starts refusing.

**How to run the Arena.** Do not give away the 11-word answer. The productive
path is: students notice the 50%, print the raw replies, see "Positive" eight
times, realize the model is not reading the query, and add an output constraint.
That sequence is the lesson. Point them at the raw output, not at the fix.

The cliff at 8 words is the other half: golf has a floor, and finding it by
falling off it is more memorable than being told.

**Connecting to W7C1.** Students fixed temperature at 0 here because they built
the decoding knobs themselves last week and know what varies otherwise. Worth
naming the callback.
