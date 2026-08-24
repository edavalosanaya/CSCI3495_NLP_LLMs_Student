# W12C2 Lab: Give the model hands, one tool at a time

You are going to watch a language model stop guessing.

Ask a small model `log(3^2 * 16 - 10)` and it answers **4**. The real answer is
**4.8978**. Ask it today's date and it says **2023-11-04**, because that is
roughly when its training data ended. Neither is a bug you can prompt away: the
arithmetic was never done and the date is not in the weights.

By the end of this lab the same model answers both correctly, not because it
got smarter but because you gave it a calculator and a clock. Then you will
ask it something no single tool can answer, *"how much hotter is it today than
yesterday in San Antonio?"*, and watch it chain three calls to get there.

![The ReAct loop: Thought, Action, Observation, repeated](../../class-01/lecture/visuals/react-loop.png)

![Which guard stops which failure mode](../lecture/visuals/robust-table.png)

## Before you code: the shape of the thing

An agent here is a **loop around a text model**, and nothing more exotic:

1. Show the model the task and the list of tools it may call.
2. Read one `Action: tool[input]` line out of its reply.
3. Run that tool yourself, in Python.
4. Paste the result back as `Observation:` and go round again.

The model never runs anything. It only ever emits text; **your loop decides
what actually happens**. Every tool has the same tiny contract, one string in
and one string out:

```
calc[log(3**2 * 16 - 10)]   ->  4.897839799950911
today[]                     ->  2026-08-14
weather[san antonio, today] ->  101.0
search[reflexion]           ->  Reflexion (Shinn et al., 2023) has an agent ...
```

Tools **never raise**. An error comes back as an ordinary Observation the
model can read and recover from, which is why a bad tool call costs a step
instead of crashing the run.

## Which model this lab uses

This lab needs a model that can chain calls, so it defaults to
**`qwen2.5:1.5b`** rather than the course's usual 0.5b:

```bash
ollama pull qwen2.5:1.5b
```

Both are far under the course's 3B ceiling and both run on a laptop CPU. Step 8
has you rerun everything on `qwen2.5:0.5b` to find where the floor is, which is
one of the more interesting things in this lab.

**Everything except the demo runs without a model at all.** The agent takes its
model as a plain callable, so the tests inject a scripted fake. You can do the
entire lab offline and only need Ollama for `run_demo.py`.

## How this lab works

Each step tells you **what to write**, then how to check it.

```bash
alias lab='docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-12/class-02/exercise course'
```

> **Read this before you trust a green test.** Until you fill in the TODOs, the
> suite falls back to the reference solution so the course sweep stays green on
> a fresh checkout. That means a passing test is not by itself evidence that
> *your* code works. The real check for each step is the **hand check** printed
> under it, which runs your file directly.

Check **one step**:

```bash
lab python -m pytest test_agent.py -k step1 -q
```

Check **everything**:

```bash
lab python -m pytest test_agent.py -q
```

Stuck for more than a few minutes? Open the walkthrough released after class at the
matching step.

---

### Step 0, Orientation

Nothing to write. Look at what you have been given.

Open `tools.py`. The calculator's dangerous half is already written for you:
`_eval_node` walks a Python **AST** and evaluates only a whitelist of node
types. Read it and notice what is missing: there is no branch for `ast.Name`,
so an expression can never mention a variable, an attribute or a builtin.

**Why it matters:** the obvious way to write `calc` is `eval(expr)`, and the
model's output is untrusted text. `eval` on untrusted text is a remote code
execution bug. The AST walk is how you get arithmetic without getting `eval`.

Run the starter to see it fail loudly:

```bash
lab python -c "import tools; print(tools.calculator('2+2'))"
```

```
NotImplementedError
```

---

### Step 1, The calculator

**Write:** `calculator(expr)` in `tools.py`. Strip the expression, rewrite `^`
as `**`, parse it with `ast.parse(expr, mode="eval")`, hand it to `_eval_node`,
and turn every failure into an `Error: ...` string. Render a whole float as an
int so `2 + 2` reads `4`, not `4.0`.

**Done when:**

```
lab python -m pytest test_agent.py -k step1 -q
3 passed, 20 deselected
```

**Check it by hand:**

```bash
lab python -c "import tools; print(tools.calculator('log(3^2 * 16 - 10)'))"
```

```
4.897839799950911
```

**Why the `^` rewrite matters.** In Python `^` is bitwise XOR, not a power. If
you skip the rewrite, `log(3^2 * 16 - 10)` parses as `log(3 XOR 22)` = `log(21)`
and returns **3.0445**: a plausible number, silently wrong. A tool that fails
loudly costs the agent one step; a tool that lies costs you the whole answer.

---

### Step 2, The clock

**Write:** `today(_arg)`. One line: return `datetime.date.today().isoformat()`.
It ignores its argument, because the model will sometimes call `today[]` and
sometimes `today[now]`.

**Done when:** `-k step2` gives `1 passed, 22 deselected`.

**Why it matters:** this is the cheapest possible proof that a tool beats
memory. No prompt can tell the model what day it is; a two-line function can.

---

### Step 3, The weather service

**Write:** `weather(arg)`, where `arg` looks like `"san antonio, yesterday"`.
Split on the first comma. Normalise the city (strip whitespace and quotes,
lowercase it, turn `_` into a space). Reject an unknown city with a message
that **lists the known ones**. Use the provided `_day_offset(day)` for the date
and reject an offset outside the series.

**Done when:** `-k step3` gives `4 passed, 19 deselected`.

**Check it by hand:**

```bash
lab python -c "import tools; print(tools.weather('San_Antonio, yesterday'))"
```

```
94.0
```

**Why the sloppiness tolerance matters.** A small model will write
`San_Antonio`, `"yesterday"`, and `SAN ANTONIO` on three consecutive turns. A
tool that accepts only one exact spelling turns every one of those into a
retry, and the agent burns its whole budget on punctuation. Be forgiving about
the **input** and exact about the **output**. Note also that the temperatures
are indexed by *offset from today*, not by calendar date, so this lab gives the
same numbers whatever day you run it.

---

### Step 4, Local search

**Write:** `search(query)`. Score every `CORPUS` entry by how many words it
shares with the query and return the best entry's text.

**Done when:** `-k step4` gives `1 passed, 22 deselected`.

**Why it matters:** this stands in for retrieval (Week 11) as a *tool*. Same
contract as the others, so the agent needs no special case for it. That is the
payoff of a uniform tool interface.

---

### Step 5, The registry

**Write:** fill in `TOOLS` at the bottom of `tools.py` with all four tools.

**Done when:** `-k step5` gives `3 passed, 20 deselected`.

**Why it matters:** read `build_prompt` in `agent.py`. The prompt's tool list is
generated **from this dict**. A tool you forget to register is a tool the model
is never told about and can never call, no matter how well you wrote it. This
is the single most common reason a student's agent "ignores" a working tool.

---

### Step 6, Parsing the model's action

**Write:** `parse_action(text)` in `agent.py`. `_ACTION_OPEN` finds the tool
name and which bracket it opened with. Walk forward from there counting depth,
and return `(tool, input)` when depth hits zero. Return `None` if it never
closes.

**Done when:** `-k step6` gives `3 passed, 20 deselected`.

**Why you cannot just use a regex.** The obvious pattern `\[(.*?)\]` is
non-greedy and stops at the first `]`. Given `calc[log(3**2 * 16 - 10)]` a
naive pattern truncates the expression and you get a syntax error from a
perfectly good action. Accept `tool(input)` as well: small models drift into
parentheses, and there is nothing to be gained by failing on that.

---

### Step 7, The grounding check

**Write:** `is_grounded(answer, observations)`. Return True only if every number
in the answer also appears in some observation.

**Done when:** `-k step7` gives `2 passed, 21 deselected`.

**Why it matters, and this is the interesting one.** Once tools exist, the most
common failure is not a broken call. It is the model calling **one** tool, then
**inventing** the second number instead of looking it up, and reporting a
confident wrong answer. That is a real trace from this lab:

```
weather[san antonio, today] -> 101.0
calc[101.0 - 98.0]          -> 3.0        <- 98.0 came from nowhere
finish[3.0 degrees hotter]                <- the true answer is 7.0
```

The loop cannot tell a real lookup from an invented number. This check can.

---

### Step 8, Run the whole thing

```bash
lab python -m pytest test_agent.py -q
```

```
.......................                                                  [100%]
23 passed
```

Now the demo, which needs Ollama:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps \
  -e OLLAMA_HOST=http://host.docker.internal:11434 \
  course python weeks/week-12/class-02/solutions/run_demo.py
```

Real output, `qwen2.5:1.5b`:

```
WITHOUT TOOLS (the model answers from memory)
  What is log(3^2 * 16 - 10)?
    4
  What is today's date?
    2023-11-04

WITH TOOLS (the model may call calc / today / weather / search)
  What is log(3^2 * 16 - 10)?
    calc[log(3^2 * 16 - 10)] -> 4.897839799950911
    finish[4.897839799950911]
    stopped: finished
  What is today's date?
    today[] -> 2026-08-14
    finish[2026-08-14]
    stopped: finished

A QUESTION THAT NEEDS THREE TOOL CALLS
  How much hotter is it today than yesterday in San Antonio?
    weather[san antonio, today] -> 101.0
    weather[san antonio, yesterday] -> 94.0
    calc[101.0 - 94.0] -> 7
    finish[7 degrees hotter]
    stopped: finished
```

Three calls, no arithmetic done by the model, and an answer every number of
which came from a tool.

---

### Step 9, Find the floor

Run the same demo on the smaller model:

```bash
COURSE_MODEL=qwen2.5:0.5b   # add -e COURSE_MODEL=... to the docker command
```

Measured behaviour of `qwen2.5:0.5b` on this exact lab:

| Task | What it does |
|---|---|
| `calc` | calls the tool correctly, gets **4.897839799950911**, then never says `finish` |
| `today[]` | skips the tool and answers "Today's date" |
| `weather` | correct lookup, **94.0**, then repeats the action until the guard fires |
| three-call chain | makes all three correct calls, then loops and never finishes |

Notice what is and is not broken. It picks the **right tool** and reads the
**right number** almost every time. What it cannot do is *stop*, and it cannot
hold a three-step plan together. That is why the loop has a fallback: when the
budget or the repeat-guard trips without a `finish`, the agent answers with the
last good Observation rather than throwing the work away.

**Write down your own answer to this:** if the 0.5b gets every individual call
right, why does making the tool list longer make it *worse*? (Look at what
`build_prompt` produces for one tool versus four, and think back to the
chain-of-thought scaling result in Week 10.)

## In-class activity: break your own agent

Full period, individual, then compare in pairs at the end.

1. **Add a fifth tool** of your own. Anything deterministic and offline: a unit
   converter, a dictionary, a fake stock ticker. Register it and ask a question
   that needs it.
2. **Write a question that makes your agent fail**, and classify the failure:
   wrong tool, malformed action, invented number, or never finishes.
3. **Fix that failure with a guard, not with a bigger model.** Then check your
   guard did not break the passing cases: `pytest -q` must still be green.

## Stretch goals

- Make `is_grounded` report *which* number was invented, and feed that back as
  an Observation so the model can correct itself instead of just being rejected.
- Add a per-tool call budget (at most 2 `weather` calls per task) and see
  whether it fixes the 0.5b's looping.
- The fallback answers with the last good Observation. Find a task where that is
  the *wrong* thing to do, and decide what the loop should do instead.
