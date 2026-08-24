# HW5: Prompt Engineering & Retrieval-Augmented Generation (RAG)

**Out:** Week 10, Class 1 · **Due:** Week 11, Class 1 · **Weight:** 5% of course grade
**Estimated time:** 5-7 hours

## Learning goals
By completing this homework you will be able to:
1. Build a **vector-space retriever** (TF-IDF + cosine similarity) over a corpus.
2. **Chunk** long documents into overlapping passages suitable for retrieval.
3. Assemble a **grounded prompt** that forces the model to answer only from
   retrieved context, the core trick that reduces hallucination in RAG.
4. Add **chain-of-thought** prompting and reason about when it helps.
5. Wire retrieval and generation into an end-to-end **RAG pipeline**, calling a
   local **Ollama** model for the final answer.

## Background: RAG (Lewis 2020) + CoT (Wei 2022)
Readings:
- **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks**, Lewis
  et al., 2020, [arXiv:2005.11401](https://arxiv.org/abs/2005.11401). (Week 11.)
- **Chain-of-Thought Prompting Elicits Reasoning in LLMs**, Wei et al., 2022,
  [arXiv:2201.11903](https://arxiv.org/abs/2201.11903). (Week 10.)

A parametric LLM stores knowledge only in its weights: it can be out of date,
can't cite sources, and hallucinates confidently. **RAG** augments generation
with a **non-parametric memory**, a retriever fetches relevant passages from an
external corpus, and the generator is conditioned on those passages. This lets
you update knowledge by editing the corpus (not retraining) and lets the model
ground its answer in citable text.

In this homework the retriever is a classic **TF-IDF** vector-space model
(simpler than Lewis et al.'s dense retriever, but the same retrieve-then-read
shape) and the generator is a small local model via Ollama. You'll also add
**chain-of-thought**: asking the model to "think step by step" before answering,
which Wei et al. showed substantially improves multi-step reasoning.

## Tasks
Edit **`homeworks/hw5/rag.py`** and implement each `# TODO`. Do **not** edit the
tests. Everything except the final generation step is a pure, offline function.

1. **Retriever** (Task 1): `tokenize`, `cosine`, and a `TfidfIndex` with
   smoothed idf `ln((1+N)/(1+df)) + 1`, per-document tf-idf vectors, and a
   `search(query, k)` that returns the top-`k` `(doc_index, cosine)` pairs.
2. **Chunking** (Task 2): `chunk_text` splits text into overlapping word
   windows (`max_words`, `overlap`), covering the whole document.
3. **Prompt assembly** (Task 3): `build_prompt` produces a grounded prompt with
   a "answer only from context / else say I don't know" instruction, numbered
   `[1] [2] ...` passages, the question, and an optional step-by-step (CoT) cue.
4. **End-to-end** (Task 4): `rag_answer` retrieves, builds the prompt, and calls
   `generate_fn` (defaults to the real Ollama `generate`), returning the
   retrieved hits, passages, prompt, and answer.

### Short written questions (put answers in `ANSWERS.md`)
- **Q1.** Give two concrete advantages of RAG over a purely parametric LLM for a
  question-answering product.
- **Q2.** Why does the grounding instruction ("answer only from context, else
  say *I don't know*") matter? What failure mode does it target?
- **Q3.** TF-IDF retrieval matches on exact word overlap. Give one query where it
  fails but a *dense* (embedding) retriever would likely succeed, and explain why.

## Deliverables
- Completed `homeworks/hw5/rag.py` (all TODOs; offline tests pass).
- `homeworks/hw5/ANSWERS.md` with Q1-Q3 and your **AI-use disclosure**.

## How to run & test
From the repo root, inside the course Docker image:
```bash
docker compose -f docker/docker-compose.yml run --rm course \
    python -m pytest homeworks/hw5/tests -q
```
All offline tests must pass. One live test calls real Ollama; it is **skipped by
default** and only runs when you opt in with a running Ollama server:
```bash
# (with `ollama serve` running and `ollama pull qwen2.5:0.5b` done)
HW5_LIVE_OLLAMA=1 python -m pytest homeworks/hw5/tests -q
```
Try your full pipeline interactively to see grounded generation in action.

## Grading rubric (100 pts)
| Item | Pts |
|------|----:|
| `TfidfIndex` (idf, vectors, cosine `search`) correct | 35 |
| `chunk_text` overlap/coverage correct | 15 |
| `build_prompt` grounded + numbered + CoT option | 25 |
| `rag_answer` pipeline wires retrieve→prompt→generate | 10 |
| Written Q1-Q3 | 10 |
| AI-use disclosure present & honest | 5 |

## AI-use disclosure reminder
You may use LLM coding assistants, **but** you must (a) disclose any AI
assistance in `ANSWERS.md`, (b) be able to explain every line you submit, and
(c) never present AI-generated prose as your own. See the syllabus AI-use policy.
