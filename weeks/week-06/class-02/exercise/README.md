# W6C2: Chain-of-thought prompting

Open **`lab.ipynb`**, select the project's **`.venv`** kernel, and run the cells
from the top. It talks to the course LLM server at `spark.cs.trinity.edu`, so you
need to be on the campus network. Paste your own key into the connect cell.

## What is in the notebook

| Part | What it does |
|---|---|
| Connect | Sends one prompt to `qwen2.5-3b` through the OpenAI client. |
| Zero-shot | Asks for the sum of five 3-digit numbers, answer only. |
| Few-shot | The same question, after two worked examples that show only the answer. |
| Chain-of-thought | The same examples, but each one shows its reasoning. |
| **Exercise** | Two agents talk to each other to solve a problem. You write this one. |
