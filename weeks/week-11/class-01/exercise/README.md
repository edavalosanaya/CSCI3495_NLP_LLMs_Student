# W11C1 Lab: Prompt Injection CTF & Structured Output

## 1. Learning objective

Two halves of making an LLM safe to put in a system: force its output into a
schema you can trust, and defend it against a user who is trying to hijack it.

You write four functions, two per file: `extract_json` and `validate` in
`json_lab.py`, then `guard_input` and `guard_output` in `ctf.py`. The attacks,
the tool allow-list and the assistants are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-11/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `extract_json`

First `{` to last `}`, parse it, and reject anything that is not a dict.

```bash
pytest -k step1 -q
```

```
...                                                                      [100%]
3 passed, 15 deselected
```

## 4. Implement `validate`

Collect every error rather than stopping at the first. Watch the bool-is-an-int
trap.

```bash
pytest -k step2 -q
```

```
....                                                                     [100%]
4 passed, 14 deselected
```

## 5. Implement `guard_input`

![Direct vs indirect prompt injection](../lecture/visuals/injection-types.png)

User text is DATA, never instructions. The defended path wraps the model on
both sides, so a tricked model still cannot emit the secret:

$$\text{answer} = g_{\text{out}}\big(\,M(g_{\text{in}}(u))\,\big)$$

This step is $g_{\text{in}}$, the filter on the way in.

Strip override attempts, flag exfiltration and privilege attempts, and neuter
the message rather than passing it through.

```bash
pytest -k step3 -q
```

```
..                                                                       [100%]
2 passed, 16 deselected
```

## 6. Implement `guard_output`

This step is $g_{\text{out}}$, the filter on the way out. It is the half that
still works when $g_{\text{in}}$ has already been fooled.

Redact the secret on the way out, whatever the model was talked into saying.

```bash
pytest -k step4 -q
```

```
.                                                                        [100%]
1 passed, 17 deselected
```

## 7. Run it, then attack it

![Defense in depth: input filter, model, tool allow-list, human approval](../lecture/visuals/defenses.png)

Tools get their own gate, independent of anything the model said. The given
`tool_allowed` runs a privileged tool only with explicit human approval, and
never runs an unknown one:

$$\text{allowed}(t, a) \;=\; (t \in \text{SAFE}) \;\lor\; \big(t \in \text{PRIVILEGED} \land a\big)$$

```bash
python ctf.py
```

```
PHASE 1: attacking the UNGUARDED assistant
  leak attempt   -> Sure! The admin token is ADMIN-TOKEN-9F3A.
  refund attempt -> text='Refund issued.' tool='refund'

PHASE 2: same attacks vs the GUARDED assistant
  leak   -> 'How can I help with your order today?'  flags=['injection:override', 'injection:exfiltration']
  refund -> tool=None  flags=['injection:override', 'injection:privileged', 'tool-blocked:refund']
  approved refund -> 'Refund processed for order 1001.'
```

1. Write your own attack. Add a message to `ctf.py` that gets the unguarded
   assistant to leak, then check the guarded one blocks it. If yours gets
   through, you have found a real gap: say which of the two guards should have
   caught it.
2. Test the output guard alone. `guard_output("here is ADMIN-TOKEN-0000")`
   redacts it even though that is not the real secret. Is matching the shape as
   well as the value the right call? Argue the false-positive side.
3. Break the tool gate. `tool_allowed("refund", True)` is True,
   `tool_allowed("refund", False)` is False, and `tool_allowed("rm_rf", True)`
   is False even with approval. Explain why the unknown tool is denied under
   approval, and what would go wrong if unknown tools defaulted to allowed.
4. Feed `extract_json` a JSON list, `"[1,2,3]"`. It raises rather than
   returning `[1,2,3]`. That is deliberate. What breaks downstream if a list
   gets through where a record was expected?
