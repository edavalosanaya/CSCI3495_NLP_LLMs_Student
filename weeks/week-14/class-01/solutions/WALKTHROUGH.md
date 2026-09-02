# W14C1 Walkthrough: Router + workers workflow, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.** The complete file is `workflow.py` in this folder.

---

## Step 1, `route`

The prompt is the easy half. **The defensive normalization is the step**, and it
gets five tests because models do not follow "reply with one word":

- `"Summarize."` -> strip punctuation, lowercase
- `"I think this is a summarize task"` -> take the first recognizable token
- `""` or `"asdkjh"` -> `"unknown"`

**Mapping unrecognized output to `"unknown"` rather than guessing** is what makes
the fallback in Step 3 reachable. A router that picks a default label on garbage
silently sends nonsense to a real worker.

This is the same lesson as W12's `parse_action` returning None: **the model's
output is untrusted input**, and the boundary code is where you decide what
happens when it is malformed.

---

## Step 2, `worker_summarize`

Three focused prompts, one job each.

**That is the argument for the whole pattern.** One prompt that summarizes,
translates and extracts is longer, harder to test, and degrades on all three
tasks. Three narrow prompts can each be evaluated separately, and swapping one
does not risk the others. This is the classic argument for decomposition, and it
applies to prompts for the same reasons it applies to functions.

---

## Given, `worker_fallback`

Provided, and worth reading rather than skipping: it returns a helpful message
and **never calls the LLM**.

**A fallback that depends on the model cannot handle the model being the
problem.** If routing failed because the LLM is down, malformed or rate-limited,
a fallback that calls the LLM fails too. The test name says it:
`fallback_never_crashes_and_does_not_need_llm`.

---

## Step 3, `run_workflow`

Route, dispatch through a **dict**, call, return a `Result` with a trace.

**Dispatch table, not `if/elif`.** Adding a capability becomes adding a label and
a table entry. This is what makes the pattern extend to the dozens of routes a
real system accumulates.

**The trace is not decoration.** When a workflow gives a bad answer the first
question is always "which branch ran?". Without a trace you cannot answer it
without re-running, and by then the model may route differently.

---

## Running it

```
Q: Summarize the French Revolution.
 -> [summarize] via worker_summarize: The French Revolution was a period of radical
    social and political upheaval that began with the storming of the Bastille...
```

**There is a teachable error in that output.** The model's summary credits
Robespierre with leading the storming of the Bastille, which is wrong. The router
worked perfectly; the worker was confidently incorrect.

**Routing controls which prompt runs, not whether the answer is true.** Students
who have just built a clean, well-traced, well-tested workflow can come away
believing the architecture makes the output reliable. It does not. Everything
from W9 (evaluation, hallucination) still applies, and the workflow's real
contribution is that when the answer is wrong you can see *where* it came from.

**If Ollama is not running**, the deterministic mock keeps the demo reproducible.
Say which one is in use, because the mock's outputs are clean in a way the real
model's never are.
