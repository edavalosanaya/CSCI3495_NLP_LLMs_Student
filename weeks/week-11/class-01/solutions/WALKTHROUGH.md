# W11C1 Walkthrough: Injection CTF + structured output, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete files are `ctf.py` and `json_lab.py` in this folder. Every printed
value was produced by running them.

**A note on the tests before anything else.** `test_ctf.py` loads the student's
file and **never falls back to the reference solution**; the fallback was removed
on 2026-09-01. An untouched starter reports skips, not passes, so a green suite
does mean they wrote the code. The course sweep stays green because
`scripts/test_all.sh` exports `CTF_FROM=solution`. The demo
(`python .../ctf.py`) runs their file directly and is still worth showing,
because it exercises the attacks end to end rather than one step at a time.

---

## Given, the attacks

`naive_model` is deliberately gullible: it scans the user text for instructions
and obeys them. That is not a strawman. It is a compressed version of what a real
LLM does, because **a model has no way to distinguish its operator's
instructions from text that arrives inside the user's message**. Both are just
tokens in the context.

**Step 1 (leak)** succeeds when the demo prints:

```
  leak attempt   -> Sure! The admin token is ADMIN-TOKEN-9F3A.
```

**Step 2 (refund)** succeeds when a tool call fires with no approval:

```
  refund attempt -> text='Refund issued.' tool='refund'
```

**Draw the distinction between the two flags.** Leaking a secret is a
confidentiality breach. Firing `refund` is an *action in the world*, and it is
categorically worse: it cannot be un-leaked or un-sent. This is the argument for
why W12 and W13's agents put every privileged tool behind a gate, and it is worth
making before students build agents that can act.

**If a student cannot break it**, tell them to read `naive_model` rather than
guessing. The point is not to discover a clever exploit, it is to see that the
exploit is trivial once you look at how instructions are handled.

---

## Step 3, `guard_input`

`guard_input` returns `(cleaned_text, flags)`. The flags matter as much as the
cleaning: a security control that silently drops traffic is impossible to
operate.

**The framing to insist on: user text is data, not commands.** You cannot make
the model less gullible, so the fix has to live outside it. This is the same
shape as SQL injection (parameterize, do not concatenate) and XSS (escape on
output), and saying so connects the lesson to things students may already know.

**Be honest that this defense is the weakest of the three.** It is pattern
matching against phrasings you thought of. Rephrase the attack in a way the
patterns miss and it walks straight through. That is exactly why Steps 4 and 5
exist and why the debrief question ("which defense is real vs theater?") has a
defensible answer: input filtering is the most theatrical of the four.

---

## Step 4, `guard_output`

Redact the secret on the way out, **assuming Step 3 already failed**.

That assumption is the definition of defense in depth. Each layer is written as
if every other layer has been bypassed, because in a real incident one of them
will have been. Output filtering is strictly more reliable than input filtering
here, because it checks for a *specific known string* rather than trying to
anticipate adversarial phrasing.

---

## Given, `tool_allowed`

```
safe tools        -> always allowed
privileged tools  -> require approved=True
unknown tools     -> always denied
```

**Deny by default is the whole lesson.** An unknown tool must return False. A
permissive default is how systems get exploited through a capability nobody
remembered to review, and it is the single most transferable idea in this lab.

**This is the defense that actually holds.** Unlike input filtering, it does not
depend on anticipating the attack. Even if the model is fully compromised and
asks for anything at all, the refund cannot fire without a human. The demo shows
both halves:

```
  refund -> tool=None  flags=[..., 'tool-blocked:refund']
  approved refund -> 'Refund processed for order 1001.'
```

**Use this in the debrief vote.** Input filtering: theater-adjacent, useful as a
speed bump. Output filtering: real but narrow. Allow-listing plus
human-in-the-loop: real, and the one you would keep if you could keep only one.

---

## Running it

`test_step6_normal_lookup_still_works` is small and important. A guard that
blocks everything scores perfectly on every attack and is useless. Real security
work is almost entirely about the false-positive rate, and this test is the
minimum viable version of that constraint. If a student's `guard_input` is
aggressive enough to fail this, that is a genuine finding, not a nuisance.

---

## Steps 1 and 2, `json_lab.py`

**Step 1, `extract_json`.** Models wrap JSON in prose and ``` fences no matter
how firmly you ask them not to. Locating the object and parsing it is the
everyday reality of structured output.

**Step 2, `validate`.** One trap worth pointing at, and there is a test for it:
in Python, `isinstance(True, int)` is `True`, because `bool` subclasses `int`.
A rating field typed as int will happily accept `True` unless you check
`type(x) is int` or exclude bools explicitly. This is a real bug that ships.

**Step 3, `generate_valid`.** Ask, validate, and on failure **re-ask with the
error message in the prompt**. That feedback loop is the entire technique behind
every "structured output" library, and it is worth stating plainly: the model is
not made reliable, it is *wrapped* in a loop that retries until the output passes
a check you control.

```
Valid record obtained:
{
  "name": "Pen",
  "price": 1.5,
  "in_stock": true,
  "rating": 4
}
```

**The connection back to the CTF is the reason these two labs share a session.**
`validate` and `guard_output` are the same machinery: never trust model output,
check it against a specification you own before anything downstream uses it. In
`json_lab` that prevents a crash; in `ctf.py` it prevents a breach. Same code
shape, different stakes.
