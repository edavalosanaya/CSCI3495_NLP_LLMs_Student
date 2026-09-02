# W10C2 Walkthrough: Chain-of-thought, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `cot_lab.py` in this folder. Every printed value was
produced by running it against `qwen2.5:0.5b`.

---

## Orientation

The two prompt builders differ by one clause:

```python
def direct_prompt(question: str) -> str:
    return f"Answer with only the final number.\n\nQ: {question}\nA:"
```

That is the entire intervention. No weights change, no extra data, no examples.
One instruction.

**`StubModel` deserves a word of honesty.** It computes the right answer from a
lookup table and reveals it only when the prompt says "step by step", returning
`correct + 1` otherwise. It *simulates* the CoT effect rather than exhibiting it,
so the offline pipeline is testable without a network. It is a fixture, not
evidence. Make sure students know the real result comes from the Ollama run.

---

## Given, `extract_answer`

`re.findall(r"-?\d+", text)` then `[-1]`.

**The last integer, not the first**, and this is the whole reason the function
exists. A CoT reply is full of intermediate numbers:

> "23 minus 17 is 6, then 6 plus 12 is 18. The answer is 18."

Taking the first match grades the model on `23`. Taking the last grades it on its
conclusion. This is a genuine design decision in every CoT evaluation harness,
and it is fragile: a model that appends "(that took 3 steps)" would be graded on
`3`. Production harnesses use a stricter answer format for exactly this reason,
which is why `cot_prompt` asks for "The answer is N." at the end.

---

## Step 1, `majority_vote`

Most common value, ties to the smallest, ignoring `None`.

**This is self-consistency** (Wang et al. 2022), and the intuition is worth
stating: sample several chains at temperature above 0, and take the answer they
agree on. It works because **wrong reasoning goes wrong in many different ways
while correct reasoning converges**. Three chains that each make a different
arithmetic slip produce three different wrong answers; three correct chains
produce the same right one. The vote exploits that asymmetry.

The tie-break to the smallest value is arbitrary but must be deterministic, the
same discipline as everywhere else in the course.

---

## Step 2, `evaluate`

Build a prompt per item, query, extract, compare, average.

**It grades the final number and never the prose.** That is a deliberate and
slightly uncomfortable choice: a chain full of nonsense that happens to end on
the right number scores the same as a flawless derivation. It is also the honest
one, because grading reasoning automatically is an unsolved problem. The lecture's
"plausible is not correct" slide is this point, and students should be able to
say why the harness cannot check it.

---

## Running it

```
[model] using Ollama model 'qwen2.5:0.5b'

prompt style      accuracy
--------------------------
direct                33%
chain-of-thought     100%
```

**2 of 6 to 6 of 6, from one clause in the prompt.** No training, no examples, no
change to the model at all. This is the most striking single result in the
prompting weeks and it lands best if students have already committed to a
prediction on the whiteboard.

**Then immediately qualify it**, or students over-generalize:

1. **Six problems.** This is a demo, not a measurement. One item is 17
   percentage points.
2. **Arithmetic word problems are CoT's best case.** They decompose into
   sequential steps with intermediate values worth writing down. The lecture's
   "when NOT to use CoT" slide covers the cases where it adds latency and noise
   for nothing.
3. **Wei et al. found the benefit is scale-dependent.** Their Fig. 4 shows CoT
   helping only at each family's largest models, and *hurting* small ones. That
   this 0.5B model benefits at all is partly because these problems are easy
   enough for it once the steps are separated.

**The mechanism, in one sentence for the board:** a direct prompt asks the model
to produce the answer in a single forward pass through a fixed number of layers;
a CoT prompt lets it write intermediate results into the context and read them
back, effectively buying more computation per problem. That framing is what makes
W10's "scale inference, not just parameters" slide and the reasoning-model
material click.

**Running the activity.** The whiteboard step matters more than the code here.
Problem B (the nests: 4 x 3 = 12 eggs, 2 hatch, so 10 unhatched) is the one to
spend time on, because both 12 and 2 appear in the problem as tempting one-step
answers. Students who predicted "direct will answer 12" and then watch it do
exactly that have learned something no amount of explanation delivers.
