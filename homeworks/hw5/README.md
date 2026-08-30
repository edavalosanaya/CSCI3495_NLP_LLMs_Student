# HW5: Prompt Engineering & Retrieval-Augmented Generation (RAG)

**Out:** Week 10, Class 1 · **Due:** Week 11, Class 1 · **100 points** · **Weight:** 2.5% of the course grade
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

## Files

```
hw5/
  rag.py             # <- YOU implement the TODOs here
  tests/test_rag.py  # the tests each step below refers to
  ANSWERS.md         # <- YOU write the short answers here
  README.md          # this handout
```

## How this homework works

This handout is a sequence of steps. Each step is one function, and **each step
ends with a test you can run**, so you always know whether you are done before
you move on. Work them in order: later steps import earlier ones.

From the repository root, inside the course image:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw5/tests -q
```

`hw` is a shortcut for the long docker command. Set it up once per
terminal session, using the line for **your** shell:

```
# macOS / Linux (bash, zsh)
alias hw='docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw5/tests -q'

# Windows, PowerShell
function hw { docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw5/tests -q @args }

# Windows, Command Prompt
doskey hw=docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw5/tests -q $*
```

Then:

```bash
hw -k step3      # check ONLY step 3
hw               # run every step
```

If you already work inside the container (`... run --rm --no-deps course bash`),
drop the docker prefix and just use `python -m pytest homeworks/hw5/tests -q`.

**Before you write anything, every test skips.** That is expected: the suite
detects the unfinished starter and skips rather than drowning you in failures.
The moment step 1 is implemented the tests start running for real.

**Total when you are finished: `12 passed, 1 skipped`.**

### Step 0, Orientation (0 pts)

Nothing to write yet.

Read `rag.py` top to bottom. The retrieval half is pure Python: no embeddings model,
no vector database, just TF-IDF over dictionaries, so you can see every number.
`generate()` is already written and talks to Ollama; every step except the last one
is testable without a model running. Then:

```bash
hw
```

You should get `13 skipped`. One test stays skipped even when you are finished: it
only runs with a live Ollama, which is why the finished total is
`12 passed, 1 skipped`. To run it too, start `ollama serve`, `ollama pull
qwen2.5:0.5b`, and set `HW5_LIVE_OLLAMA=1`.

### Step 1, `tokenize` and `cosine` (10 pts)

**Write** `tokenize(text)`, lowercasing and keeping word characters, and `cosine(a, b)` over two `{term: weight}` dictionaries. Return `0.0` when either vector is empty rather than dividing by zero.

**Done when** `hw -k step1` prints `2 passed, 11 deselected`.

**Check it by hand**

```python
>>> from rag import tokenize, cosine
>>> tokenize("BPE splits rare words!")
['bpe', 'splits', 'rare', 'words']
>>> round(cosine({"a": 1.0, "b": 1.0}, {"a": 1.0}), 4)
0.7071
```

**Why it matters.** Sparse dictionaries instead of dense arrays keep the vocabulary implicit, which is how real TF-IDF search engines are built. The 0.7071 is the same cos(45 degrees) you computed in HW2, on a different representation.

### Step 2, `TfidfIndex.build` (15 pts)

**Write** `build(docs)`: store the documents, compute an IDF per term, and build one TF-IDF vector per document. Use a smoothed IDF so a term appearing in every document does not blow up.

**Done when** `hw -k step2` prints `1 passed, 12 deselected`.

**Check it by hand**

```python
>>> D = ["byte pair encoding merges frequent pairs",
...      "attention lets every token look at every other token",
...      "perplexity is the inverse geometric mean"]
>>> idx = TfidfIndex().build(D)
>>> len(idx.idf)                       # distinct terms across the three documents
19
>>> round(idx.idf["attention"], 4)     # appears in one of three documents
1.6931
```

**Why it matters.** A term in one of three documents scores higher than one in all three, which is the entire idea of IDF: rarity is informativeness. Check that `every`, which appears twice in document 2, does not get a higher IDF than `attention`.

### Step 3, `_vectorize_query` and `search` (10 pts)

**Write** the query vectorizer and `search(query, k)`: score every document by cosine against the query vector and return the top `k` as `(index, score)` pairs, highest first. Drop zero-scoring documents rather than padding the list.

**Done when** `hw -k step3` prints `3 passed, 10 deselected`.

**Check it by hand**

```python
>>> D = ["byte pair encoding merges frequent pairs",
...      "attention lets every token look at every other token",
...      "perplexity is the inverse geometric mean"]
>>> idx = TfidfIndex().build(D)
>>> [(i, round(s, 4)) for i, s in idx.search("attention", k=2)]
[(1, 0.2774)]
```

**Why it matters.** Only one document mentions attention, so asking for two results correctly returns one. A retriever that pads its answer to `k` with irrelevant text is how you poison your own prompt in step 5.

### Step 4, `chunk_text` (15 pts)

**Write** `chunk_text(text, max_words, overlap)`: split into chunks of at most `max_words` words where consecutive chunks share `overlap` words. Validate that `overlap < max_words` and handle empty input.

**Done when** `hw -k step4` prints `2 passed, 11 deselected`.

**Check it by hand**

```python
>>> from rag import chunk_text
>>> chunk_text(" ".join(str(i) for i in range(12)), max_words=5, overlap=2)
['0 1 2 3 4', '3 4 5 6 7', '6 7 8 9 10', '9 10 11']
```

**Why it matters.** Read the overlap: chunk 2 starts at `3`, repeating the last two words of chunk 1. Without it, a sentence that straddles a boundary is split across two chunks and neither one retrieves. With `overlap >= max_words` the loop never advances, which is why the validation is part of the step.

### Step 5, `build_prompt` (25 pts)

**Write** `build_prompt(query, passages, cot)`. The prompt must instruct the model to answer **only** from the context and to say `I don't know` otherwise, number the passages `[1]`, `[2]`, ... so they can be cited, include the question, and add a step-by-step reasoning cue when `cot=True`. It must not crash when `passages` is empty.

**Done when** `hw -k step5` prints `3 passed, 10 deselected`.

**Check it by hand**

```python
>>> p = build_prompt("what is BPE?", ["byte pair encoding merges pairs"])
>>> "[1]" in p and "what is BPE?" in p
True
>>> "I don't know" in p
True
>>> build_prompt("what is BPE?", []).strip() != ""    # empty retrieval is survivable
True
```

**Why it matters.** This is the highest-scoring step because it is where grounding actually happens. The retrieval can be perfect and the model will still answer from memory unless the prompt tells it not to, and the `[1]` numbering is what makes a citation checkable rather than decorative.

### Step 6, `rag_answer` (10 pts)

**Write** the pipeline: retrieve the top `k` passages for the query, build the prompt from them, call `generate_fn(prompt, model=model)`, and return the answer. Take `generate_fn` as a parameter so the test can pass a fake.

**Done when** `hw -k step6` prints `1 passed, 1 skipped, 11 deselected`.

**Check it by hand**

```python
>>> calls = []
>>> def fake(prompt, model=None):
...     calls.append(prompt)
...     return "stub answer"
>>> rag_answer("what is BPE?", idx, generate_fn=fake)
'stub answer'
>>> "[1]" in calls[0]        # the prompt it built really contained the passages
True
```

**Why it matters.** Injecting `generate_fn` is what makes this whole pipeline testable without a GPU, a network, or a model download. It is also how you would swap Ollama for a hosted API later without touching the retrieval code.

### Step 7, Run the whole thing (0 pts)

```bash
hw
```

Every step green means `12 passed, 1 skipped`. If a step you finished earlier has gone red,
you broke it with a later change; fix that before you submit.

## Written reflection (15 pts)

Worth 15 points: 10 for the answers, 5 for an honest AI-use note.

Answer in `ANSWERS.md`, 2-4 sentences each:

- **Q1.** Give two concrete advantages of RAG over a purely parametric LLM for a
  question-answering product.
- **Q2.** Why does the grounding instruction ("answer only from context, else say
  *I don't know*") matter? What failure mode does it target?
- **Q3.** TF-IDF retrieval matches on exact word overlap. Give one query where it fails
  but a dense (embedding) retriever would likely succeed, and explain why.

## What to submit

- `rag.py` with every TODO filled in and `hw` green (`12 passed, 1 skipped`).
- `ANSWERS.md` with Q1-Q3 answered.
- The `AI-USE:` note described below.

Partial credit follows the tests: each step is worth the points listed above, and a
step whose tests pass earns them. Code that does not import earns at most the written
points, so submit something that runs even if it is incomplete.

## AI-use disclosure (required)

Per the syllabus, you may use LLM tools as coding assistants, but you must
**disclose** it (which tool, for what), be able to **explain every line** you
submit, and write the reflection in your own words. Put a short `AI-USE:` note
in your file header. Undisclosed AI use is an academic-integrity violation.
