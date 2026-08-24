# HW6: Building an LLM Agent with Tool Use (ReAct)

**Out:** Week 12, Class 1 · **Due:** Week 13, Class 2 · **Weight:** 5% of course grade
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

## Tasks
Edit **`homeworks/hw6/agent.py`** and implement each `# TODO`. Do **not** edit
the tests.

1. **Tools** (Task 1): `calculator`, a **safe** arithmetic evaluator (use
   `ast` or a hand-written parser; **no `eval`/`exec`**; return `"Error..."` on
   bad input or divide-by-zero). `search`, best keyword match over `CORPUS`.
2. **`parse_step`** (Task 2): parse model text into a `Step`
   (thought/action/action_input or final_answer). A `Final Answer:` line wins
   over any action; labels are case-insensitive and trimmed.
3. **`run_tool`** (Task 3): dispatch a `Step` to the right tool via the `TOOLS`
   registry; unknown tool → `"Error: unknown tool ..."`.
4. **`build_react_prompt` + `react_loop`** (Task 4): assemble the prompt from
   the question + transcript, then loop up to `max_steps`: call the `llm`, parse,
   run tools, append the observation, and stop on a final answer. Return
   `{'answer', 'steps', 'history'}`.

### Short written questions (put answers in `ANSWERS.md`)
- **Q1.** Why does the ReAct loop pass each tool's **Observation** back into the
  next prompt? What breaks if you don't?
- **Q2.** Why must the `calculator` avoid Python's `eval`? Give one concrete
  thing a malicious or buggy `eval`-based tool could do.
- **Q3.** The loop has a `max_steps` budget. Give one failure mode this guards
  against, and one downside of setting it too low.

## Deliverables
- Completed `homeworks/hw6/agent.py` (all TODOs; tests pass).
- `homeworks/hw6/ANSWERS.md` with Q1-Q3 and your **AI-use disclosure**.

## How to run & test
From the repo root, inside the course Docker image:
```bash
docker compose -f docker/docker-compose.yml run --rm course \
    python -m pytest homeworks/hw6/tests -q
```
All tests must pass (they use a mock LLM, no Ollama needed). One live test
calls real Ollama; it is **skipped by default** and runs only when you opt in:
```bash
# (with `ollama serve` running and `ollama pull qwen2.5:0.5b` done)
HW6_LIVE_OLLAMA=1 python -m pytest homeworks/hw6/tests -q
```
Smaller models won't always follow the ReAct format perfectly, observing where
the agent goes off-script is part of the learning.

## Grading rubric (100 pts)
| Item | Pts |
|------|----:|
| `calculator` correct **and** safe (no eval; handles errors) | 25 |
| `search` keyword match over the corpus | 10 |
| `parse_step` (final-answer precedence, case-insensitive, trimmed) | 20 |
| `run_tool` dispatch incl. unknown-tool error | 10 |
| `react_loop` (tool use, observation feedback, max_steps, return shape) | 20 |
| Written Q1-Q3 | 10 |
| AI-use disclosure present & honest | 5 |

## AI-use disclosure reminder
You may use LLM coding assistants, **but** you must (a) disclose any AI
assistance in `ANSWERS.md`, (b) be able to explain every line you submit, and
(c) never present AI-generated prose as your own. See the syllabus AI-use policy.
