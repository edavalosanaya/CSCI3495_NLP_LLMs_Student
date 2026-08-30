# HW6: Building an LLM Agent with Tool Use (ReAct)

**Out:** Week 12, Class 1 · **Due:** Week 13, Class 2 · **100 points** · **Weight:** 2.5% of the course grade
**Estimated time:** 6-8 hours

## Learning goals
By completing this homework you will be able to:
1. Implement **safe tools** an agent can call, a calculator (no `eval`!) and a
   local knowledge-base search.
2. Parse a model's output in the **ReAct format** (Thought / Action / Action
   Input / Final Answer) into structured steps.
3. **Dispatch** tool calls and feed observations back to the model.
4. Build the **ReAct loop**: reason → act → observe → repeat → answer, with a
   step budget.
5. Test the whole agent with a **mock LLM**, then run it for real on **Ollama**.

## Background: ReAct (Yao et al., 2022)
Reading: **ReAct: Synergizing Reasoning and Acting in Language Models**, Yao et
al., 2022, [arXiv:2210.03629](https://arxiv.org/abs/2210.03629). (Week 12.)

A plain LLM that only "thinks" can reason but can't look anything up or compute
reliably; a model that only "acts" can call tools but lacks a plan. **ReAct**
interleaves the two: the model emits a **Thought** (reasoning), then an
**Action** (a tool call), receives an **Observation** (the tool's result), and
loops, grounding its reasoning in real tool output and reducing hallucination.
This thought-action-observation trace is exactly what powers modern tool-using
agents.

In this homework the agent has two safe, local tools (a `calculator` and a
`search` over a fixed corpus, **no network**). You drive the loop with a
**mock** LLM in tests (canned Thought/Action turns) so everything is
deterministic, then plug in a real local model via Ollama.

## Files

```
hw6/
  agent.py             # <- YOU implement the TODOs here
  tests/test_agent.py  # the tests each step below refers to
  ANSWERS.md           # <- YOU write the short answers here
  README.md            # this handout
```

## How this homework works

This handout is a sequence of steps. Each step is one function, and **each step
ends with a test you can run**, so you always know whether you are done before
you move on. Work them in order: later steps import earlier ones.

From the repository root, inside the course image:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw6/tests -q
```

`hw` is a shortcut for the long docker command. Set it up once per
terminal session, using the line for **your** shell:

```
# macOS / Linux (bash, zsh)
alias hw='docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw6/tests -q'

# Windows, PowerShell
function hw { docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw6/tests -q @args }

# Windows, Command Prompt
doskey hw=docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw6/tests -q $*
```

Then:

```bash
hw -k step3      # check ONLY step 3
hw               # run every step
```

If you already work inside the container (`... run --rm --no-deps course bash`),
drop the docker prefix and just use `python -m pytest homeworks/hw6/tests -q`.

**Before you write anything, every test skips.** That is expected: the suite
detects the unfinished starter and skips rather than drowning you in failures.
The moment step 1 is implemented the tests start running for real.

**Total when you are finished: `17 passed, 1 skipped`.**

### Step 0, Orientation (0 pts)

Nothing to write yet.

Read `agent.py` top to bottom. `CORPUS` and the `TOOLS` registry are already there,
and `ollama_llm()` is written for you. The agent you are building is the ReAct loop
from the reading: think, act, observe, repeat. Every step except the last is testable
with a scripted fake LLM, so you do not need a model running. Then:

```bash
hw
```

You should get `18 skipped`. One test stays skipped even when you are done: it only
runs against a live Ollama, which is why the finished total is `17 passed, 1 skipped`.
To run it too, start `ollama serve`, `ollama pull qwen2.5:0.5b`, and set
`HW6_LIVE_OLLAMA=1`.

### Step 1, `calculator` (25 pts)

**Write** a **safe** arithmetic evaluator. Parse the expression with `ast.parse(..., mode='eval')` and walk the tree, allowing only numbers and the arithmetic operators. Anything else, and any arithmetic error, returns a string starting with `Error:`. **Do not use `eval` or `exec`.**

**Done when** `hw -k step1` prints `6 passed, 12 deselected`.

**Check it by hand**

```python
>>> from agent import calculator
>>> calculator("2 + 3 * 4")
'14'
>>> calculator("1/0")
'Error: division by zero'
>>> calculator("__import__('os')")
'Error: invalid expression (unsupported expression)'
```

**Why it matters.** This is the highest-scoring step because it is the security one. `eval` on a string the model produced is arbitrary code execution triggered by whatever text the model read, which in an agent includes text an attacker controls. An AST whitelist is the fix, and returning an error string rather than raising is what keeps the loop alive.

### Step 2, `search` (10 pts)

**Write** keyword search over `CORPUS`: return the matching entry, or a clear no-results message. Never raise.

**Done when** `hw -k step2` prints `1 passed, 17 deselected`.

**Check it by hand**

```python
>>> from agent import search
>>> search("python")
'Python is a programming language created by Guido van Rossum in 1991.'
>>> search("capital")
'No results found.'
```

**Why it matters.** 'No results found.' is an *answer*, not a failure. The model reads it as an observation and can try a different query; an exception would end the run.

### Step 3, `parse_step` (20 pts)

**Write** the parser: pull `Thought:`, `Action:`, `Action Input:` and `Final Answer:` out of the model's text. Matching is case-insensitive and values are stripped. **A `Final Answer` wins over an `Action`** if both appear. Text with no labels at all returns a `Step` with everything `None`.

**Done when** `hw -k step3` prints `4 passed, 14 deselected`.

**Check it by hand**

```python
>>> from agent import parse_step
>>> s = parse_step("Thought: I need math.\nAction: calculator\nAction Input: 2+2")
>>> s.action, s.action_input, s.final_answer
('calculator', '2+2', None)
>>> parse_step("Thought: done.\nFinal Answer: 4").final_answer
'4'
>>> parse_step("I think it is 42 but I am not sure.").action is None
True
```

**Why it matters.** Small models format badly. The last check is the one that matters in practice: a model that ignores your format must leave the loop able to continue, not crash it. The final-answer precedence rule stops an agent that has already answered from firing one more tool call.

### Step 4, `run_tool` (10 pts)

**Write** the dispatcher: look the step's action up in `TOOLS` and call it with `action_input`. An unknown tool returns an error **string** naming the tool, not an exception.

**Done when** `hw -k step4` prints `2 passed, 16 deselected`.

**Check it by hand**

```python
>>> from agent import run_tool, Step
>>> run_tool(Step(thought=None, action="calculator", action_input="2+2", final_answer=None))
'4'
>>> run_tool(Step(thought=None, action="teleport", action_input="home", final_answer=None))
'Error: unknown tool "teleport". Available: calculator, search'
```

**Why it matters.** Listing the available tools in the error is what lets the model correct itself on the next turn. An agent that says only 'error' teaches the model nothing and usually repeats the same mistake until the step budget runs out.

### Step 5, `build_react_prompt` (5 pts)

**Write** the prompt builder: state the question, list the available tools, show the required `Thought / Action / Action Input` format, and append the history so far so the model can see its own previous observations.

**Done when** `hw -k step5` prints `1 passed, 17 deselected`.

**Check it by hand**

```python
>>> p = build_react_prompt("what is 2+2?", [])
>>> "what is 2+2?" in p and "calculator" in p
True
```

**Why it matters.** The history is the agent's only memory. Everything the model knows about what it already tried has to be in this string, because the model itself is stateless between calls.

### Step 6, `react_loop` (15 pts)

**Write** the loop: build the prompt, call `llm(prompt)`, parse the step, and either return the final answer or run the tool, append the observation to the history, and go again. Stop at `max_steps` and return a dict with the answer and the transcript.

**Done when** `hw -k step6` prints `3 passed, 1 skipped, 14 deselected`.

**Check it by hand**

```python
>>> scripted = iter([
...     "Thought: math.\nAction: calculator\nAction Input: 2+2",
...     "Thought: done.\nFinal Answer: 4",
... ])
>>> out = react_loop("what is 2+2?", llm=lambda p: next(scripted))
>>> out["answer"]
'4'
>>> len(out["history"]) >= 2      # the observation was fed back in
True
```

**Why it matters.** Feeding the observation back is what makes this an agent rather than one long prompt: the model's next decision is conditioned on what actually happened, not on what it predicted would happen. `max_steps` is the guard that keeps a confused model from looping forever, and the test checks it fires.

### Step 7, Run the whole thing (0 pts)

```bash
hw
```

Every step green means `17 passed, 1 skipped`. If a step you finished earlier has gone red,
you broke it with a later change; fix that before you submit.

## Written reflection (15 pts)

Worth 15 points: 10 for the answers, 5 for an honest AI-use note.

Answer in `ANSWERS.md`, 2-4 sentences each:

- **Q1.** Why does the ReAct loop pass each tool's **Observation** back into the next
  prompt? What breaks if you don't?
- **Q2.** Why must the `calculator` avoid Python's `eval`? Give one concrete thing a
  malicious or buggy `eval`-based tool could do.
- **Q3.** The loop has a `max_steps` budget. Give one failure mode this guards against,
  and one downside of setting it too low.

## What to submit

- `agent.py` with every TODO filled in and `hw` green (`17 passed, 1 skipped`).
- `ANSWERS.md` with Q1-Q3 answered.
- The `AI-USE:` note described below.

Partial credit follows the tests: each step is worth the points listed above, and a
step whose tests pass earns them. Code that does not import earns at most the written
points, so submit something that runs even if it is incomplete.

## AI-use disclosure (required)

Per the syllabus, you may use LLM tools as coding assistants, but you must
**disclose** it (which tool, for what), be able to **explain every line** you
submit, and write the reflection in your own words. Put a short `AI-USE:` note
in your file header. Undisclosed AI use is an academic-integrity violation.
