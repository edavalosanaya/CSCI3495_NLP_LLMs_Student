# W12C2 Walkthrough: giving the model hands

Step-by-step solutions for `exercise/README.md`. Code here is copied verbatim
from `tools.py` and `agent.py` next to this file. Try each step yourself
first; the point of the lab is the failures you meet on the way.

---

## Step 1, The calculator

```python
def calculator(expr: str) -> str:
    expr = expr.strip().replace("^", "**")
    if not expr:
        return "Error: empty expression"
    try:
        result = _eval_node(ast.parse(expr, mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception:  # noqa: BLE001, any parse/type error becomes an Observation
        return f"Error: could not evaluate '{expr}'"
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)
```

**The idea.** All the real work is in `_eval_node`, which was given to you: it
walks the AST and evaluates only whitelisted node types. Your job is the
wrapper, and the wrapper's job is that **nothing escapes as an exception**.

Expected output:

```
>>> calculator("log(3^2 * 16 - 10)")
'4.897839799950911'
>>> calculator("3^2")
'9'
>>> calculator("1/0")
'Error: division by zero'
>>> calculator("__import__('os')")
"Error: could not evaluate '__import__('os')'"
```

**Common mistakes**

- *Forgetting the `^` rewrite.* This is the one that hurts, because it does not
  raise. `log(3^2 * 16 - 10)` becomes `log(3 XOR 22)` = `log(21)` = **3.0445**.
  You get a number, it looks fine, and it is wrong.
- *Letting the exception propagate.* One bad expression then kills the whole
  agent run instead of costing it one step.
- *Returning `4.0` where a human expects `4`.* Cosmetic, but the tests pin it,
  and observations get pasted straight back into the prompt.

---

## Step 2, The clock

```python
def today(_arg: str = "") -> str:
    return _dt.date.today().isoformat()
```

**The idea.** The argument is accepted and ignored on purpose: the model writes
`today[]` on one turn and `today[now]` on the next, and neither should be an
error.

**Common mistake:** giving the parameter no default. The loop sometimes calls it
with an empty string and sometimes with nothing at all.

---

## Step 3, The weather service

```python
def weather(arg: str) -> str:
    city, _, day = arg.partition(",")
    city = city.strip().strip('"\'').lower().replace("_", " ")
    if not city:
        return "Error: usage is weather[city, day]"
    if city not in _SERIES:
        return f"Error: no weather for '{city}'. Known: {', '.join(sorted(_SERIES))}."
    off = _day_offset(day)
    if off is None:
        return f"Error: could not read the date '{day.strip()}'. Use today, yesterday, or YYYY-MM-DD."
    if not 0 <= off < len(_SERIES[city]):
        return f"Error: no reading that far back; I have the last {len(_SERIES[city])} days."
    return str(_SERIES[city][off])
```

**The idea.** `partition(",")` rather than `split(",")` so a stray second comma
cannot explode into three values. Then normalise hard: strip whitespace, strip
quotes, lowercase, underscores to spaces.

Expected output:

```
>>> weather("san antonio, today")
'101.0'
>>> weather("San_Antonio, yesterday")
'94.0'
>>> weather('austin, "yesterday"')
'96.0'
>>> weather("paris, today")
"Error: no weather for 'paris'. Known: austin, boston, san antonio, seattle."
```

**Common mistakes**

- *Not stripping quotes.* The model really does emit `weather[San_Antonio,
  "yesterday"]`. Observed, not hypothetical.
- *An error message that does not say what IS allowed.* `Error: bad city` gives
  the model nothing to recover with; listing the known cities lets it retry
  correctly on the very next turn.

---

## Step 4, Local search

```python
def search(query: str) -> str:
    query = query.strip()
    if not query:
        return "Error: empty query"
    q = _tokens(query)
    best_key, best_score = None, 0
    for key, text in CORPUS.items():
        score = len(q & _tokens(key + " " + text))
        if score > best_score:
            best_key, best_score = key, score
    if best_key is None:
        return f"No results found for '{query}'."
    return CORPUS[best_key]
```

**The idea.** Score by set overlap against the key *and* the body, so both
`search[reflexion]` and `search[what does the reflexion paper do]` land on the
same entry. `best_score` starts at 0, so zero overlap correctly returns nothing
rather than an arbitrary first entry.

---

## Step 5, The registry

```python
TOOLS = {
    "calc": calculator,
    "today": today,
    "weather": weather,
    "search": search,
}
```

**Why this is a whole step.** `build_prompt(tools)` generates the tool list in
the prompt from this dict. Register nothing and the model is told about no
tools and will answer from memory, exactly as it did before you started. This
is the most common "my agent ignores my tool" bug, and it is always this.

---

## Step 6, Parsing the action

```python
def parse_action(text: str) -> Optional[tuple[str, str]]:
    m = _ACTION_OPEN.search(text)
    if not m:
        return None
    tool, opener = m.group(1), m.group(2)
    closer = "]" if opener == "[" else ")"
    depth, out, i = 1, [], m.end()
    while i < len(text):
        ch = text[i]
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return tool, "".join(out).strip()
        out.append(ch)
        i += 1
    return None   # never closed: treat as malformed
```

**The idea.** Depth counting, not a regex. Only the delimiter that was opened
with is counted, so the parens inside `calc[log(3**2 * 16 - 10)]` are just
characters and the whole expression survives.

Expected output:

```
>>> parse_action("Action: calc[log(3**2 * 16 - 10)]")
('calc', 'log(3**2 * 16 - 10)')
>>> parse_action("Action: finish(7 degrees)")
('finish', '7 degrees')
>>> parse_action("Action: calc[1 + 2") is None
True
```

**Common mistakes**

- *`re.search(r"\[(.*?)\]")`.* Non-greedy, so it stops at the first `]` and you
  silently truncate every expression containing brackets.
- *Greedy `\[(.*)\]` instead.* Now it runs to the last `]` anywhere in the
  reply and swallows the model's next sentence.
- *Rejecting `tool(input)`.* Small models drift into parentheses constantly.
  Accepting both costs one line and saves a step every few turns.

---

## Step 7, The grounding check

```python
def is_grounded(answer: str, observations: list[str]) -> bool:
    seen = {n for obs in observations for n in _NUM_RE.findall(obs)}
    return all(n in seen for n in _NUM_RE.findall(answer))
```

**The idea.** Every number in the final answer must have appeared in some
observation. An answer with no numbers is trivially grounded, which is what you
want: this check is about fabricated *quantities*, not about prose.

Expected output:

```
>>> is_grounded("7 degrees hotter", ["101.0", "94.0", "7"])
True
>>> is_grounded("98.0", ["101.0", "94.0"])
False
```

**Why it earns its place.** From a real run of this lab, the model looked up
today's temperature, then invented yesterday's instead of calling the tool
again:

```
weather[san antonio, today] -> 101.0
calc[101.0 - 98.0]          -> 3.0
finish[3.0 degrees hotter]
```

Every step is well-formed. No tool errored. The loop has no way to notice, and
the answer is wrong by 4 degrees. `is_grounded` catches it because `98.0` never
came out of a tool.

**Common mistake:** comparing floats numerically after parsing. Keep it as
string matching against what the observation actually said; `7` and `7.0` are
different strings and the tests pin the behaviour you get from the real tools.

---

## Step 8, Full run

```
23 passed
```

The demo (needs Ollama, `qwen2.5:1.5b`) is in `run_demo.py`; its verified output
is printed in the README. The headline: `log(3^2 * 16 - 10)` goes from **4** to
**4.897839799950911**, the date goes from **2023-11-04** to the real one, and
the three-call chain answers **7 degrees hotter** with every number traceable to
a tool.

---

## Step 9, Where the floor is

Rerunning on `qwen2.5:0.5b` is worth doing once, because the failure is
specific rather than general. Measured on this lab, the 0.5b:

- picks the **right tool** and gets the **right number** for calc, weather and
  search;
- almost never emits `finish`, so the loop's fallback is what produces its
  answer;
- makes all three correct calls in the chain task and then loops instead of
  finishing.

The answer to the README's closing question: `build_prompt` emits one line per
registered tool plus the rules and the exemplar, so going from one tool to four
roughly doubles the prompt. A 0.5B model has to hold the task, the tool list,
the format rules and the running transcript in the same small attention budget,
and the instruction that suffers first is the one it needs last, `finish`. This
is the same shape as the Week 10 result: the ability to follow multi-step
procedural instructions appears with scale, and prompting harder does not
substitute for it. Which is also why the answer is a **guard in the loop**, not
a longer prompt.
