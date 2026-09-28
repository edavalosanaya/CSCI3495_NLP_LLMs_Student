# W6C1: Prompting an LLM

Open **`lab.ipynb`**, select the project's **`.venv`** kernel, and run the cells
from the top. It talks to the course LLM server at `spark.cs.trinity.edu`, so you
need to be on the campus network.

## What is in the notebook

| Part | What it does |
|---|---|
| Connect | Sends one prompt to `qwen2.5:0.5b` through the OpenAI client. |
| Zero-shot | Asks for a sentiment label with no examples. |
| Few-shot | The same question, after two worked examples. |
| Persona | A system message decides who the model is. |
| **Exercise** | Two agents, each with its own persona, talk to each other for 5 turns. You write this one. |
