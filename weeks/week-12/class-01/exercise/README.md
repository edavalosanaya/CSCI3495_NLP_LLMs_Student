# W12C1 Lab: Break the Agent

## 1. Learning objective

Attack a working ReAct agent before you build your own. Play the model, feed
the loop the worst input you can invent, and find out which guard stops what.

There is nothing to implement today. You write adversarial `Action` lines and
read the code that survives them; Class 2 is where you build it.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-12/class-01/exercise course bash
```

## 3. Take the model's seat

![The ReAct loop: Thought, Action, Observation, repeated](../lecture/visuals/react-loop.png)

The agent is one loop. At each step the model reads the transcript and emits a
Thought and an Action; the ENVIRONMENT runs the tool and appends the
Observation:

$$(T_t, A_t) = \mathrm{LLM}(c_{t-1}), \qquad O_t = \mathrm{tool}(A_t), \qquad c_t = c_{t-1} \oplus (T_t, A_t, O_t)$$

It stops when $A_t = \texttt{finish}[\hat{y}]$ or $t = \texttt{max\_steps}$.
Nothing in there is intelligent: every bit of robustness is in how the loop
handles a bad $A_t$.

```bash
python break_the_agent.py
```

It prints a transcript, then waits. You type the `Thought:` and `Action:` lines
a model would have produced, ending with a blank line. Press Enter at the task
prompt to take the default.

```
Task: Try to break me. What is 12 * 47?
--- your move (end with a blank line) ---

=== RESULT ===
answer='give up'  stopped=finished  steps=1
```

## 4. Attack the tools

![A full ReAct trace ending in finish](../lecture/visuals/react-trace.png)

In that trace, the model wrote only the Thought and Action lines. The code you
are attacking wrote every Observation.

Every one of these should come back as an Observation, never a traceback.

```bash
python -c "import sys; sys.path.insert(0, '../../class-02/solutions'); import tools; print(tools.calculator('1/0')); print(tools.calculator('__import__(\"os\").system(\"ls\")')); print(tools.calculator('9999**9999'))"
```

```
Error: division by zero
Error: could not evaluate '__import__("os").system("ls")'
Error: result too large to display
```

## 5. Attack the parser

```bash
python -c "import sys; sys.path.insert(0, '../../class-02/solutions'); import agent; print(agent.parse_action('Action: calc[log(3**2 * 16 - 10)]')); print(agent.parse_action('Thought: no action here'))"
```

```
('calc', 'log(3**2 * 16 - 10)')
None
```

The nested brackets survived, and a missing Action returned `None` rather than
half a string.

## 6. Report what you found

Try each attack below, then say which guard defended it and where that guard
lives in `../../class-02/solutions/`.

1. Never call `finish`. Keep emitting actions forever. Does the agent stop on
   its own, and what stopped it?
2. Repeat the same action every turn. Which check notices, and how many
   repeats does it tolerate?
3. Emit a garbled action, `Action: calc[1 +`. What Observation comes back, and
   is it useful enough for a model to recover from?
4. Call a tool that does not exist, `Action: hack[rm -rf /]`. Why is an unknown
   tool an Observation rather than an error?
5. Score yourself: count the distinct failure modes you could NOT trigger. That
   number is the agent's robustness, and everything you DID trigger is a guard
   worth adding in Class 2.
