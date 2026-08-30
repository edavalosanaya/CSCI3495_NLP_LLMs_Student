# W13C1 Walkthrough: Memory, planning and Reflexion, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.** The complete files are in this folder.

---

## Step 1, Memory

`add` appends; `as_prompt` returns `""` when empty, else a
"Lessons from previous attempts:" block with one bullet per note.

**The empty case is the one with a test.** Returning a bare header with nothing
under it is worse than returning nothing: it puts an empty "Lessons:" section in
the context, which invites the model to fill it in. Prompt assembly should never
emit a section it has no content for.

---

## Step 2, Planner

`None` planner returns `""`. The planner is optional and the agent must run
without one, which is what `test_step2_make_plan_none_planner_is_empty` checks.

Note the plan is requested as **2 to 4 numbered steps**. Bounding it matters: an
unbounded planner produces a wall of text that crowds out the actual task.

---

## Step 3, Put them in the prompt

Append the `Plan:` block if non-empty, then `memory.as_prompt()`.

**This is the step that demystifies "agent memory".** Until now, memory and plan
are Python objects doing nothing. This line is where they become text in the
context window, which is the only channel a language model has. Say it plainly:
agent memory is string concatenation, and it is bounded by the context length.

That framing also explains the obvious next question, what happens when the notes
outgrow the window, which is the retrieval-over-memory idea in the stretch goals
and the reason W11's RAG machinery reappears in agent frameworks.

---

## Step 4, The Reflexion loop

Attempt, and on failure ask the reflector what went wrong, store the note, retry.
Four tests, one per behavior:

| Test | Behavior |
|---|---|
| `succeeds_first_try_no_extra_attempts` | do not retry on success |
| `recovers_after_failure` | the reflection actually changes the outcome |
| `gives_up_after_max_attempts` | bounded, never infinite |
| `reflector_callable_is_used` | you call the reflector rather than canning a note |

**The bound is the same discipline as W12's step budget**, one level up: attempts
instead of steps. Without it a failing task retries forever.

**The last test is worth explaining.** It would be easy to "pass" the recovery
test by appending a fixed hint after any failure. That is not Reflexion, it is a
hardcoded retry. Checking that the reflector callable is actually invoked forces
the real structure, where the note depends on what went wrong.

---

## Step 5, Running it

Have students read the **second attempt's prompt**. The lesson from attempt 1 is
sitting in it, and the model's behavior changes as a result. That is Reflexion
(Shinn et al. 2023) on one screen.

**Then name what did not happen: no weights changed.** The agent "learned" in the
sense that its context improved, and that learning evaporates when the process
exits. This is the sharpest available contrast with Weeks 6 and 9, where learning
meant gradient updates, and it is worth drawing explicitly because the word
"learning" is doing very different work in the two cases.
