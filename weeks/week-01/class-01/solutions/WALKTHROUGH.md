# W1C1 Walkthrough: Setup and your first local LLM

Instructor reference and student rescue hatch. There is no code to hand out for
this session: the "solution" is a working environment plus understanding what
each command did. **Read only the step you are stuck on.**

Every output below was produced by actually running the commands in the course
container.

---

## Step 1, Build the image and pull the model

**What the three commands actually do.**

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d ollama
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:0.5b
```

1. **`build`** constructs the course image from `docker/Dockerfile`: Python 3.12
   plus everything in `docker/requirements.txt`. This is the step that makes
   "works on my machine" a non-issue for the rest of the semester, and it is why
   every command in the course is prefixed with `docker compose ... run`.
2. **`up -d ollama`** starts the model server as a **separate long-lived
   container** (`-d` is detached). It keeps running between your commands, which
   is why you only do this once.
3. **`exec ollama ollama pull`** runs the `ollama` CLI *inside* that running
   container to download the weights into a named volume, so the 397 MB survives
   container restarts.

**Why two containers.** The `course` container is disposable and is recreated on
every `run --rm`. The `ollama` container holds the model in memory and persists.
Students who try to `pull` inside the `course` container will find the download
vanishes; that is the design working, not a bug.

**What you should see:**

```
NAME            ID              SIZE      MODIFIED
qwen2.5:0.5b    a8b0c5157701    397 MB    ...
```

**Common failures.**

- `ollama list` reports nothing: the pull went to the wrong container, or was
  interrupted. Re-run the pull.
- Build fails on a network timeout: usually campus wifi. Re-running resumes from
  the last cached layer.
- Disk pressure: the image plus model is a few GB. `docker system df` shows what
  is being used.

---

## Step 2, Verify the environment

```bash
docker compose -f docker/docker-compose.yml run --rm course python scripts/env_check.py
```

```
Python 3.12.13
Core libraries:
  [ok]  numpy                  2.5.1
  [ok]  scipy                  1.18.0
  [ok]  pandas                 3.0.3
  [ok]  sklearn                1.9.0
  [ok]  matplotlib             3.10.7
  [ok]  nltk                   3.9.2
  [ok]  regex                  2025.11.3
  [ok]  torch                  2.12.1+cpu
  [ok]  transformers           5.13.0
  [ok]  datasets               5.0.0
  [ok]  sentence_transformers  5.6.0
  [ok]  peft                   0.19.1
  [ok]  faiss                  1.14.3
  [ok]  ollama                 ?
Ollama:
  [ok]  Ollama reachable at http://ollama:11434; models: ['qwen2.5:0.5b']

ENV CHECK OK.
```

**Two details worth pointing out to a stuck student.**

- `torch 2.12.1+cpu`. The `+cpu` suffix is deliberate: the whole course is
  CPU-only by design, so nobody needs a GPU and nobody's results depend on
  having one.
- `ollama ?`. The version is unknown because that package does not expose
  `__version__`; the `[ok]` means the import succeeded. Not a failure.

**`http://ollama:11434`** is a Docker-internal hostname. The `course` container
reaches the `ollama` container by service name over the compose network. This is
why `--no-deps` breaks this step: it skips starting the dependency, and there is
no `ollama` host to resolve.

---

## Step 3, Run your first LLM prompt

```
Model:       qwen2.5:0.5b
Temperature: 0.7
Prompt:      In one short sentence, what is Natural Language Processing?
------------------------------------------------------------
Natural Language Processing (NLP) is the study of how computers can understand
and interpret human language, enabling machines to communicate with humans.
------------------------------------------------------------
```

**The script degrades on purpose.** If Ollama is unreachable it prints the two
commands that fix it rather than a traceback. That pattern (fail with an
actionable message, never a stack trace at a student) recurs in every exercise in
the course, and it is worth naming as a design choice rather than letting it pass
unnoticed.

**The teaching moment for the whole session.** The model is 397 MB of numbers on
the student's own disk, doing arithmetic on their own CPU, with the network
irrelevant. Half the mystique of "AI" evaporates the first time someone watches
this run offline. It is worth saying explicitly: there is no server, no account,
no API key, and nothing about this prompt left the room.

---

## Step 4, Change the prompt

No right answer; the goal is that they touch the file and re-run it.

**Push them to break it.** A 0.5B model fails fast and legibly, which makes it a
better teaching instrument here than a frontier model would be. Reliable ways to
find an edge:

- Multi-digit arithmetic (`what is 3847 times 219?`).
- Anything local and specific ("what is the mascot of Trinity University?").
- Anything dated ("what year is it?").
- Multi-step reasoning ("if I have 3 boxes of 4 pens and give away 5...").

These failures are not a detour, they are the raw material for the scavenger hunt
and the first data point in the reliability thread that runs to Week 9 (evaluation
and hallucination) and Week 15 (ethics).

---

## Step 5, Change the temperature

**What you should see** (identical prompt, three runs):

```
T=0.0: Natural Language Processing (NLP) is the study of how computers can understand
       and interpret human language, enabling machines to communicate with humans.
T=0.0: Natural Language Processing (NLP) is the study of how computers can understand
       and interpret human language, enabling machines to communicate with humans.
T=1.2: Natural Language Processing (NLP) involves the development of systems and
       algorithms that can understand, interpret, generate human language naturally
```

**The explanation, in the order students usually need it.**

1. The model does not output words. It outputs a **score for every token in its
   vocabulary**, and something else has to choose one.
2. Temperature rescales those scores before they become probabilities:
   $P(w) = e^{z_w / T} / \sum_v e^{z_v / T}$.
3. Dividing by a **small** $T$ exaggerates the differences between scores, so the
   top token's probability approaches 1 and the choice stops being random. At
   $T = 0$ implementations simply take the argmax.
4. Dividing by a **large** $T$ compresses the differences, so tokens the model
   thought were mediocre get a real chance. Output gets more varied, and past
   about 1.5 it degrades into nonsense.

**A caveat to be honest about.** $T = 0$ giving byte-identical output is a
property of this setup, not a mathematical guarantee: batching, hardware
floating-point differences, and server-side changes can all perturb it. It held
across the runs above, and it is a reasonable expectation, but "deterministic"
here means "argmax decoding", not "provably reproducible forever".

**Where this goes.** Week 7 covers decoding properly, greedy vs beam vs top-k vs
nucleus, and students implement the sampling themselves. All that is needed today
is the intuition that the knob exists and controls a randomness/quality trade-off.

---

## Step 6, Reflect

**What a good reflection looks like.** Specific, and tied to something they
actually observed:

> I expected a model this small to be useless, but it defined NLP fine. It also
> told me confidently that 3847 x 219 was a number that was not even close, and
> it did not hedge at all. The confidence seems independent of whether it is
> right.

**What to push back on.** "It was surprisingly good" or "it was worse than
ChatGPT" without an example. Ask what they ran.

That observation, that fluency and accuracy are separate axes and the model
signals no difference between them, is the single most useful thing a student can
carry out of Week 1. It is the thesis of W9C2 (hallucination) and it is why the
course spends Week 9 on evaluation at all.

---

## Running the session

- **Budget the build.** The image build is the long pole. If students have not
  built before class, expect 10-20 minutes of the period to go to it, on campus
  wifi possibly more. Consider assigning Step 1 as pre-work.
- **The likely failure modes, in order:** Docker Desktop not running at all;
  `--no-deps` copied from a later exercise's command; pulling the model into the
  wrong container; and on Apple silicon, a first-run architecture warning that is
  safe to ignore.
- **Pair the stuck with the working.** The scavenger hunt is already a pair
  activity, so pairing a student whose environment is broken with one whose works
  keeps them in the lesson while you debug.
