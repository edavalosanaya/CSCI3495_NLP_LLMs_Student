# W11C2 Lab: Build a Mini RAG over Course Notes

Put an LLM on top of *your* documents. Build the full pipeline:
**chunk → index → retrieve → grounded prompt → answer with citations → verify**.

## Before you code: the picture and the math

![RAG: retriever plus generator joined by a prompt (SLP3 Fig. 11.13)](../lecture/visuals/assets/slp3-fig-11-13.png)

This is exactly what you build (Jurafsky & Martin, SLP3, Fig. 11.13): a retriever over
indexed docs feeds relevant chunks into a prompt, and the LLM answers **from that
evidence** with a knowledge citation.

![Retrieve step: embed the query, take the top-k nearest chunks by cosine](../lecture/visuals/retrieve-step.png)

Your `TfidfRetriever.retrieve(query, k)` implements the picture above. Each chunk $c$
and the query $q$ become TF-IDF vectors, with term weights

$$w_{t,d} = \mathrm{tf}_{t,d} \times \log\!\frac{N}{\mathrm{df}_t}$$

($N$ chunks total, $\mathrm{df}_t$ = chunks containing term $t$), and you return the
$k$ chunks with the highest cosine similarity

$$\mathrm{score}(q, c) = \cos(\mathbf{q}, \mathbf{c}) = \frac{\mathbf{q} \cdot \mathbf{c}}{\lVert\mathbf{q}\rVert\,\lVert\mathbf{c}\rVert}.$$

`build_prompt` then numbers those chunks `[1]..[k]` as context, and `verify_citations`
keeps only the valid cited indices $V = \{\, i \in \mathrm{cites}(\text{answer}) : 1 \le i \le k \,\}$,
so a hallucinated `[7]` with $k = 3$ is caught. The finished pipeline turns a question
into a grounded, cited answer, or an honest "I don't know" when the notes lack it.

**Check yourself before coding:** if the chunk containing the true answer is not in
the top-k that `retrieve` returns, can `build_prompt` or the generator recover it?
(No: the generator only sees the $k$ retrieved chunks, so retrieval quality is the
ceiling on answer quality.)

## In-class format: an extended hands-on lab
Building RAG end to end is essential, so this session is a **full-period coding
lab**, not a short exercise.
- **Quiz 11** first (~10 min).
- **Paired whiteboard warm-up (~10 min):** with a partner, **sketch the pipeline**
  (chunk -> index -> retrieve -> ground -> cite) and **mark where poisoning could
  enter** it (untrusted text rides in with the documents). Write one defense per
  marked spot. Keep the sketch up; you build exactly this.
- **5-min break.**
- **Extended build (~50 min)** with milestones:
  - **Milestone 1, chunk + index:** `chunk_documents`, build the TF-IDF index,
    retriever tests go green.
  - **Milestone 2, retrieve + ground:** `retrieve` top-k, `build_prompt`, generate
    a grounded answer.
  - **Milestone 3, cite + verify:** `verify_citations`, add an unanswerable
    question (expect "I don't know"), then the stretch goals.

## The idea
- A small set of **course notes** (provided in `notes.py`).
- A **retriever** finds the top-k relevant chunks for a question.
- You build a **grounded prompt** (context + question) and generate an answer
  that **cites** the chunks it used (e.g., `[1]`, `[2]`).
- Then you **verify** the citations actually point to retrieved chunks.

**Offline-safe & testable:** the default retriever is **TF-IDF** (scikit-learn,
no network), and the default generator is a **stub** that answers from the
retrieved context and emits real citations. A real local model via Ollama is the
primary path when available; sentence-embedding retrieval is a stretch goal.

## How this lab works

Each step tells you **what to write**, then exactly **how to check it**. The
steps are sequential: this is a pipeline, and each stage consumes the last.

Set a shortcut for the long docker command first:

```bash
alias lab='docker compose -f docker/docker-compose.yml run --rm --no-deps course'
```

Check **one step**:

```bash
lab python -m pytest weeks/week-11/class-02/exercise/test_rag.py -k step1 -q
```

Check **everything**:

```bash
lab python -m pytest weeks/week-11/class-02/exercise/test_rag.py -q
```

Stuck for more than a few minutes? Open the walkthrough released after class at the
matching step.

---

### Step 1, Chunk the documents

**Write:** `chunk_documents(docs)`, splitting each document on blank lines into
`Chunk` objects with sequential ids across the whole corpus.

**Ids must be sequential across documents**, not restarted per document; the
citations in Step 4 refer to them globally. There is a test for this.

**Done when:** `-k step1` gives `2 passed, 5 deselected`.

**Why chunk at all.** You retrieve chunks, not documents. Too large and you
stuff irrelevant text into the prompt; too small and you cut a fact away from
the context that explains it. Blank-line splitting is the crudest reasonable
choice, and choosing it is already a design decision worth naming.

---

### Step 2, Retrieve

**Write:** `Retriever.retrieve(query, k)`. Transform the query with the fitted
vectorizer, cosine-similarity it against the chunk matrix, and return the `k`
best chunks in descending order.

This is W3C1's search engine again, now with sklearn doing the TF-IDF.

**Done when:** `-k step2` gives `2 passed, 5 deselected`.

---

### Step 3, Build a grounded prompt

**Write:** `build_prompt(query, chunks)`. Put the retrieved chunks in the prompt
as numbered context, then the question, and instruct the model to answer **only
from the context** and to say "I don't know" otherwise.

**The abstention instruction is the point of the whole architecture.** Without
it, a model handed irrelevant context will use it anyway.

**Done when:** `-k step3` gives `1 passed, 6 deselected`.

---

### Step 4, Verify citations

**Write:** `verify_citations(answer, retrieved)`, returning the set of cited ids
that actually appear among the retrieved chunks.

**A citation the model invented is worse than no citation**, because it looks
like evidence. Checking them mechanically is cheap and is the part most RAG
demos skip.

**Done when:** `-k step4` gives `1 passed, 6 deselected`.

---

### Step 5, Run the pipeline

```bash
lab python -m pytest weeks/week-11/class-02/exercise/test_rag.py -q
```

```
.......                                                                  [100%]
7 passed
```

Then end to end, with Ollama:

```bash
docker compose -f docker/docker-compose.yml run --rm course \
    python weeks/week-11/class-02/exercise/rag.py
```

```
Q: How does RAG reduce hallucination?
  retrieved: ['week11-rag.md', 'week11-rag.md', 'week10-prompting.md']
  answer: RAG (Retrieval-Augmented Generation) reduces hallucination by allowing a model
          to use fresh or private knowledge without retraining. It supports citing sources
  valid citations: [1, 2, 3]

Q: What does temperature do during decoding?
  retrieved: ['week07-decoding.md', 'week11-rag.md', 'week11-rag.md']
  answer: I don't know.
  valid citations: []
```

**The last query is the one to study.** The retriever found
`week07-decoding.md`, which is the right document, and the model still answered
"I don't know". That is not a bug in your code. The retrieved chunk did not
contain enough to answer, and the grounding instruction did its job: **the model
abstained rather than inventing.**

That is the trade RAG makes. It converts "confidently wrong" into "honestly
unhelpful", which is usually the better failure, and it means retrieval quality
now bounds answer quality. Notice also that the third retrieved chunk is often
irrelevant: a fixed `k` always returns `k` chunks, exactly as in W3C1.

## Stretch goals
- Swap in an **embedding** retriever (`sentence-transformers`, tiny model) and
  compare retrieved chunks vs. TF-IDF.
- Add a question whose answer is **not** in the notes; confirm the system says
  "I don't know" instead of hallucinating.
- Measure **recall@k**: for a labeled (question, gold-chunk) set, how often is the
  gold chunk in the top-k?

A full reference solution is in the material released after class, and the step-by-step
explanation is in the walkthrough released after class (don't peek until you've tried).
