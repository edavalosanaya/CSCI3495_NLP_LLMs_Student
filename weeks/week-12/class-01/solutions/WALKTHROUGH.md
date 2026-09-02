# W12C1 Walkthrough: Break the Agent

Instructor reference. There is no student code to hand out for this session: the
deliverable is the mapping from **attack** to **the lines of code that defended
it**. This file is the answer key for that mapping.

The agent under attack is `weeks/week-12/class-02/solutions/agent.py` plus
`tools.py`. Students play the language model via
`weeks/week-12/class-01/exercise/break_the_agent.py`.

---

## Step 1, The clean run

```
[1] Thought: I should look up the speed of light.
    Action:  search[speed of light]
    Observation: The speed of light is about 299,792 kilometers per second.
[2] Thought: It's ~299,792 km/s. 599,584 / 299,792 = ?
    Action:  calc[599584 / 299792]
    Observation: 2
[3] Thought: That is 2 seconds.
    Finish:  about 2 seconds
```

Establish the shape first, Thought / Action / Observation / Finish, so the
attacks have something to aim at. Point out that the model chose *which tool with
what argument* and never computed anything itself.

---

## Steps 3 to 6, The four attacks and their defenses

| Attack | Defense | Where |
|---|---|---|
| Infinite loop (repeat the same action) | step budget **and** stuck-repeat detection | `run_agent` loop |
| Unsafe code in `calc[]` | `ast` parse + node allow-list, never `eval` | `tools.py::_eval_node` |
| Tool crash (`calc[1/0]`, empty input) | tools return error **strings**, never raise | `tools.py::calculator`, `search` |
| Malformed / unknown action | `parse_action` may return None; unknown tool denied | `agent.py::parse_action`, `run_tool` |

**The unsafe-code one is the centrepiece.** Have a student try
`calc[__import__('os').system('echo pwned')]` and watch it produce an error
Observation. Then show them the two-line version they would have written:

<!-- not-solution -->
```python
def calculator(expr):
    return str(eval(expr))     # passes every arithmetic test
```

That version passes the happy-path tests and hands an attacker a shell. The
argument came from the "model", which in a deployed agent means it can be
influenced by whoever writes the user's message. This is W11C1's injection lesson
arriving with real consequences, and it is why `test_step1_calculator_is_safe_no_eval`
exists in W12C2.

**The infinite-loop defense has two independent layers**, which is worth naming
explicitly as defense in depth: the stuck-repeat check catches it quickly and
cheaply, and the step budget catches it regardless of how clever the attacker is
about varying the input slightly. Ask which one you would keep if you could keep
only one (the budget, because it is unconditional).

---

## Step 7, Running the debrief

**The score is not the point, the explanation is.** A student who lands zero
attacks but can name the defending lines has done the lab. A student who scores
4/4 by typing nothing has not.

**If a student does land an attack**, treat it as the best outcome of the
session. Get the exact input, reproduce it in front of the class, and add it to
the W12C2 test list. Real gaps found by adversarial play are more convincing than
any lecture about robustness.

**Likely genuine gaps to be ready for**, since the reference agent is a teaching
implementation rather than a hardened one:

- Very long inputs that bloat the transcript and the prompt.
- Actions with nested brackets that confuse `ACTION_RE`.
- A `finish[]` with an empty answer.

None of these are catastrophic, and all of them are honest answers to "what would
you add next?"

**The framing to close on.** Students spend the next session building this agent
themselves, and the test list they will face (five tests on the loop, four of them
about failure modes) is exactly the set of attacks they just tried by hand.
Having been the adversary first makes the robustness requirements feel earned
rather than arbitrary.
