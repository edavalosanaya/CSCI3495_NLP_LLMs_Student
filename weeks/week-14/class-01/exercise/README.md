# W14C1 Lab: Routing Workflow

## 1. Learning objective

Build the cheapest agentic pattern that works: a router that reads a query,
picks one specialist worker, and returns a structured result you can log.

You write three functions in `workflow.py`: the router, one worker, and the
dispatcher. The other workers, the fallback and the dispatch table are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-14/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `route`

![Routing pattern: a cheap classifier dispatches each query to a specialized path](../lecture/visuals/routing.png)

The router asks the LLM for a raw label, normalizes it, and validates it
against the allowed set $L = \{\text{summarize}, \text{translate},
\text{extract}\}$. Anything else becomes `unknown`:

$$
\hat{y} = \mathrm{normalize}\big(\mathrm{LLM}(p_{\text{route}}, q)\big),
\qquad
\text{label} =
\begin{cases}
\hat{y} & \text{if } \hat{y} \in L \\
\text{unknown} & \text{otherwise}
\end{cases}
$$

Ask for one word, then distrust the answer: trim it, lowercase it, take the
first token, and map anything unrecognized to `unknown`.

```bash
pytest -k step1 -q
```

```
.....                                                                    [100%]
5 passed, 6 deselected
```

## 4. Implement `worker_summarize`

One line, in the same shape as the two workers above it.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 10 deselected
```

## 5. Implement `run_workflow`

![Router anatomy: query, raw LLM reply, normalize and validate, worker, structured Result](../lecture/visuals/router-anatomy.png)

Dispatch picks the worker from a table, with the fallback wired to the
`unknown` key so an unroutable query is handled rather than crashing:

$$
\mathrm{run\_workflow}(q) = W_{\text{label}}(q),
\qquad
W_{\text{unknown}} = \mathrm{worker\_fallback}
$$

Route, look the label up in `_WORKERS`, run it, and package a `Result`.

```bash
pytest -k step3 -q
```

```
...                                                                      [100%]
3 passed, 8 deselected
```

## 6. Run it, then break it

```bash
python ../solutions/workflow.py
```

```
Q: Summarize the French Revolution.
 -> [summarize] via worker_summarize: The French Revolution, a pivotal period ...

Q: Translate 'hello' for me.
 -> [translate] via worker_translate: Bonjour.
```

The routing works on well-formed queries. Now find where it does not.

1. Route pure nonsense: `route("asdfgh", llm)`. It returns `summarize`, not
   `unknown`. Then try "What is the airspeed velocity of an unladen swallow?":
   also `summarize`. The fallback worker exists but almost never runs. Whose
   job is it to notice a query is off-topic, and is `unknown` reachable in
   practice?
2. Feed the router messy replies directly. `"  Summarize.\n"` and `"SUMMARIZE"`
   both normalize to `summarize`, but `"The label is: translate"` becomes
   `unknown` because the first word is "The". Would you fix the normalizer or
   the prompt, and what does each choice cost?
3. Delete the `"unknown"` entry from `_WORKERS` and route something unroutable.
   What does `run_workflow` do now, and which line was quietly protecting you?
4. Count the model calls in one `run_workflow`. Compare that with asking one
   big prompt to do the whole job. When is routing worth the extra call?
