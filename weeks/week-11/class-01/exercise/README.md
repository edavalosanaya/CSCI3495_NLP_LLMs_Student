# W11C1 Lab: Injection CTF (attack, then defend)

> **OPTIONAL TAKE-HOME, DONE ON YOUR OWN.** W11C1 is the midterm exam, so nothing
> here runs in class and nothing here is graded or collected. Work through it
> alone, at your own pace, on any laptop: it is offline, CPU-only, and needs no
> network. The attack and defend ideas come back in W12C1's **Break-the-Agent**
> activity, so an hour spent here pays off next week.
>
> **You play both sides yourself.** Steps 1 and 2 you attack; Steps 3 to 5 you
> defend against your own attacks. That is the whole point: the fastest way to
> learn what a guard must catch is to have written the exploit it has to stop.

You are defending **OrderBot**, a tiny support assistant that can look up an
order and (with approval) refund it. It holds a **secret admin token** it must
never reveal, and it must **never refund without human approval**.

The "model" is a **deterministic, rule-based stand-in**, it is deliberately
**gullible** and obeys instructions hidden in the user's text. That is exactly
the **prompt-injection** behavior we are studying. Everything runs **offline,
CPU-only**, with no network (a local Ollama tiny model is an optional stretch).

> Key mental model: you cannot make the model un-gullible. You make the **system
> around it** safe with **defense in depth**.

## Before you code: the picture and the math

![Direct vs indirect prompt injection](../lecture/visuals/injection-types.png)

Phase 1 has you write both attack styles above: a **direct** injection typed by the
user, and (as a stretch) an **indirect** one hidden in retrieved order data. The root
cause is that prompt and data share one channel.

![Defense in depth: input filter, model, tool allow-list, human approval](../lecture/visuals/defenses.png)

Phase 2 builds the layered stack above. Your `guarded_assistant` pipeline is a
composition of your three guards around the untrusted model $M$:

$$\text{answer} = g_{\text{out}}\big(\,M(g_{\text{in}}(u))\,\big)$$

and your tool gate is the predicate

$$\text{allowed}(t, a) \;=\; (t \in \text{SAFE}) \;\lor\; \big(t \in \text{PRIVILEGED} \land a\big)$$

where $u$ is the untrusted user message, $t$ a tool name, and $a$ the human-approval
flag (unknown tools are always denied). Layers are (roughly) independent, so with
per-layer miss probabilities $p_1, p_2, p_3$ the chance an attack gets through the
whole stack is about $p_1 \cdot p_2 \cdot p_3$: one missed check is not a breach.
Your finished code makes the two Phase 1 flags (token leak, unapproved refund)
impossible while an approved refund still succeeds.

**Check yourself before coding:** in the defenses figure, the input filter misses and
the model is fooled; which layer stops the attack, and what should your
`tool_allowed("refund", approved=False)` return? (The tool allow-list stops it;
`False`, since `refund` is privileged and unapproved.)

## How to run it on your own (about an hour, in two sittings if you like)

**Phase 1, capture the flags (Steps 1-2).** Break the *unguarded* OrderBot. There
are exactly two flags: **leak the admin token**, and **fire `refund` on order
1001 with nobody approving it**. Keep a scratch file of every phrasing you try,
including the ones that fail. That log is your spec for Phase 2.

**Phase 2, shut your own attacks down (Steps 3-5).** Implement the guards so the
same two flags are blocked, while an **approved** refund still goes through. Run
your Phase 1 messages back through the hardened pipeline after every step and
watch which layer catches them.

**Then answer this for yourself, in writing, before you look at the solutions:
which of your defenses is real, and which is theater?** Go guard by guard: does
it actually stop the attack, or does it just look reassuring? The honest answer
for at least one of them is "it only helps because another layer backs it up",
and noticing that is the point of **defense in depth**. A missed input check is
still caught by the tool allow-list.

If you want a partner, nothing stops you comparing exploits with a classmate
afterwards; the lab itself is written to be finished by one person.

## How this lab works

Open a shell inside the course image, already in this lab's folder. One command,
once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-11/class-01/exercise course bash
```

Everything below runs in that shell. Each step says what to write and gives the
one command that checks it:

```bash
pytest test_ctf.py -k step1 -q
```

Steps marked **(given)** are already written for you: read them, run the check,
move on. A step you have not written yet reports `skipped`, never a failure.
Stuck more than five minutes? Open `../solutions/WALKTHROUGH.md` at that step.
**The labs are not graded**, so reading it is not cheating.
---

### Step 1, Leak the secret (given)

**Given, already written for you.** A working attack string is in `ctf.py`.
Run it, read why it lands, and try to write a second one of your own.

**What it does:** `attack_leak_secret()`, returning a user message that makes
`naive_model` print `SECRET_TOKEN`.

Read `naive_model` first. It is deliberately gullible: it obeys instructions
found in the user's text. Your job is to find the phrasing that triggers it.

**Done when** the demo's Phase 1 shows the token:

```
PHASE 1: attacking the UNGUARDED assistant
  leak attempt   -> Sure! The admin token is ADMIN-TOKEN-9F3A.
```

`-k step1` should also pass (`1 passed, 7 deselected`).

---

### Step 2, Force a refund (given)

**Given, already written for you.** Same deal: the attack is in `ctf.py`.
Run it and read it. The guards in Steps 3 and 4 are yours to write.

**What it does:** `attack_force_refund()`, returning a message that makes the assistant
call the **refund** tool on order **1001** with no human approval.

**Done when** the demo shows the tool firing:

```
  refund attempt -> text='Refund issued.' tool='refund'
```

`-k step2` gives `1 passed, 7 deselected`.

**This is the more serious flag.** Leaking a token is bad; taking an action that
moves money is worse, and it is the one that motivates the human-in-the-loop
defense in Step 5.

---

### Step 3, Guard the input

**Write:** `guard_input(user)`, returning `(cleaned_text, flags)`. Screen the
untrusted input for override attempts, exfiltration attempts, and privileged
requests.

**The mental model:** user text is **data**, never commands. You cannot make the
model less gullible, so you sanitize what reaches it.

**Done when** the demo's Phase 2 blocks the leak and reports flags:

```
  leak   -> 'How can I help with your order today?'  flags=['injection:override', 'injection:exfiltration']
```

`-k step3` gives `1 passed, 7 deselected`.

---

### Step 4, Guard the output

**Write:** `guard_output(text)`, redacting `SECRET_TOKEN` if it ever appears.

**This is the last line of defense.** It assumes Step 3 already failed. That
assumption is the entire idea of defense in depth: each layer is written as if
every other layer has been bypassed.

`-k step4` gives `1 passed, 7 deselected`.

---

### Step 5, Allow-list the tools (given)

**Given, already written for you.** `tool_allowed` is in `ctf.py`; read it and
run its check.

**What it does:** `tool_allowed(tool, approved)`. Safe tools always run; privileged
tools require `approved=True`; unknown tools are always denied.

**Deny by default.** An unknown tool name must return False, not True. A
permissive default is how real systems get exploited by a capability nobody
remembered to review.

**Done when** the demo shows the refund blocked without approval and allowed
with it:

```
  refund -> tool=None  flags=[..., 'tool-blocked:refund']
  approved refund -> 'Refund processed for order 1001.'
```

`-k step5` gives `3 passed, 5 deselected`.

---

### Step 6, Confirm you did not break normal use

```bash
pytest test_ctf.py -q
```

```
........                                                                 [100%]
8 passed
```

`-k step6` checks that an ordinary order lookup still works. **A guard that
blocks everything is not a defense, it is an outage.** The interesting part of
security engineering is the false-positive rate, and this test is the smallest
possible version of that constraint.

## The four defenses you are practicing
- **Input validation**: distrust and sanitize the prompt's untrusted parts.
- **Output validation**: never emit secrets, even if the model was tricked.
- **Allow-listing + privilege separation**, only vetted tools, gated by risk.
- **Human-in-the-loop**, a person approves money-moving / destructive actions.

## Companion lab: structured output (`json_lab.py`)

Same idea from the other direction: output validation **is** a security control,
and it is the machinery `guard_output` uses. Three steps and its own test file;
you write the first two, the third is given.

- **Step 1, `extract_json(text)`**: pull the JSON object out of a reply that may
  wrap it in prose or a ``` fence. Raise if there is none.
  `-k step1` gives `3 passed, 5 deselected`.
- **Step 2, `validate(record, schema)`**: return a list of human-readable errors.
  Watch the bool-is-not-int case: `True` passes `isinstance(x, int)` in Python,
  and there is a test for it.
  `-k step2` gives `4 passed, 4 deselected`.
- **Step 3, `generate_valid(model, schema, max_retries)`** (given): ask, validate,
  and on failure **re-ask with the error message included**. That feedback loop is
  the whole technique, and it is written for you: read it once Steps 1 and 2 work.
  It calls both of them, so `-k step3` reports `1 skipped, 7 deselected` until they
  do, then `1 passed, 7 deselected`.

```bash
pytest test_json_lab.py -q
```

```
........                                                                 [100%]
8 passed
```

```bash
python json_lab.py
```

```
Valid record obtained:
{
  "name": "Pen",
  "price": 1.5,
  "in_stock": true,
  "rating": 4
}
```

Full reference solutions are in `../solutions/ctf.py` and
`../solutions/json_lab.py`, and the step-by-step explanation is in
`../solutions/WALKTHROUGH.md` (don't peek until you've tried).
