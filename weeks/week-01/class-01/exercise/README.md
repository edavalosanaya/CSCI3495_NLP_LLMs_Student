# W1C1 Lab: Environment Setup & Your First Local LLM

## 1. Learning objective

Get the course environment running and send your first prompt to a language
model on your own laptop. No API key, no cloud account.

There is nothing to implement. Every step ends with a command and the exact
output you should see. If yours differs, stop and fix it there: a broken
environment in Week 1 costs you the whole semester.

## 2. Understanding the math

![A local LLM completing a prompt one token at a time](../lecture/visuals/autocomplete.gif)

The model does one thing, repeatedly: given the words so far, pick the next
token.

$$P(w_t \mid w_1, w_2, \ldots, w_{t-1})$$

The `temperature` knob in step 6 rescales the raw scores $z_w$ before sampling:

$$P(w) = \frac{e^{z_w / T}}{\sum_{v} e^{z_v / T}}$$

Small $T$ exaggerates the gaps, so the most likely token wins every time and
sampling stops being random. Large $T$ flattens them, so unlikely tokens get a
real chance.

## 3. Getting started

From the repository root on your own machine. The build downloads the whole
scientific Python stack, so it takes a while the first time:

```bash
docker compose -f docker/docker-compose.yml build
docker compose -f docker/docker-compose.yml up -d ollama
docker compose -f docker/docker-compose.yml exec ollama ollama pull qwen2.5:0.5b
```

## 4. Confirm the model is there

```bash
docker compose -f docker/docker-compose.yml exec ollama ollama list
```

```
NAME            ID              SIZE      MODIFIED
qwen2.5:0.5b    a8b0c5157701    397 MB    ...
```

397 MB is the entire model. The thing about to write English sentences for you
is one file smaller than a podcast episode.

## 5. Verify the environment

```bash
docker compose -f docker/docker-compose.yml run --rm course python scripts/env_check.py
```

```
  [ok]  torch                  2.12.1+cpu
  [ok]  transformers           5.13.0
  ...
  [ok]  Ollama reachable at http://ollama:11434; models: ['qwen2.5:0.5b']

ENV CHECK OK.
```

Every library must report `[ok]` and the last line must read `ENV CHECK OK`.

## 6. Run your first prompt

Open a shell inside the image, in this lab's folder:

```bash
docker compose -f docker/docker-compose.yml run --rm -w /workspace/weeks/week-01/class-01/exercise course bash
```

```bash
python hello_nlp.py
```

```
Model:       qwen2.5:0.5b
Temperature: 0.7
Prompt:      In one short sentence, what is Natural Language Processing?
------------------------------------------------------------
Natural Language Processing (NLP) is the study of how computers can understand,
interpret, and generate human language, enabling the processing and interaction
between humans and computers.
------------------------------------------------------------
```

## 7. Change it, then break it

The three edit points are marked `STEP 4`, `STEP 5` and `STEP 6` in
`hello_nlp.py`.

1. Ask it something you actually want to know (STEP 4). Then ask something you
   already know the answer to, precisely: a date, a population, a definition
   from your major. Is it right? Is it confidently wrong?
2. Set `temperature = 0.0` and run twice, then `1.2` and run twice (STEP 5).
   The two runs at 0.0 are identical; the two at 1.2 are not. Explain why using
   the formula in section 2 before reading anyone else's answer.
3. Push the temperature to `2.0`. At what point does the output stop being
   English, and what does that tell you about what temperature is actually
   doing to the distribution?
4. Write your reflection as a comment at the bottom of the file (STEP 6): what
   surprised you about a 0.5B model on your own laptop? "It was cool" is not a
   reflection. "It answered a definition question fine but confidently invented
   a wrong date" is.
