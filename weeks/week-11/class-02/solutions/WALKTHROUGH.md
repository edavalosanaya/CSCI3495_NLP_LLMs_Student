# W11C2 Walkthrough: Mini RAG, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `rag.py` in this folder. Every printed value was produced by
running it against `qwen2.5:0.5b`.

---

## Given, `chunk_documents`

Split on blank lines, assign **globally sequential** ids across the whole corpus.

**The id scheme is not incidental.** Citations in Step 4 refer to these numbers,
so restarting them per document would make citation `[1]` ambiguous. There is a
dedicated test (`test_step1_chunking_full_corpus_ids_sequential`).

**Chunking is the most under-appreciated decision in RAG.** Too large and each
retrieved chunk carries mostly irrelevant text that dilutes the prompt and
crowds the context window. Too small and a fact gets separated from the sentence
that qualifies it. Blank-line splitting is the crudest reasonable rule; real
systems chunk by tokens with overlap, or by document structure. Worth asking the
class what would break if a chunk cut a definition in half.

---

## Step 1, `TfidfRetriever.retrieve`

Transform the query with the **already-fitted** vectorizer (`transform`, not
`fit_transform`, or you refit on one query and destroy the vocabulary), cosine
against the chunk matrix, take the top `k` descending.

**This is W3C1's search engine with sklearn doing the TF-IDF.** Point that out:
students built this by hand in Week 3, including the cosine and the tie-break.
Retrieval in a production RAG system swaps TF-IDF for dense embeddings and an
approximate-nearest-neighbour index, but the shape is identical.

---

## Step 2, `build_prompt`

Numbered context, then the question, plus two instructions: answer **only** from
the context, and say "I don't know" if the context does not support an answer.

**The abstention clause is the whole architecture.** Without it, a model handed
three chunks will use them regardless of relevance, because that is what the
prompt appears to ask for. With it, the model has a licensed way to decline. This
is the same instinct as the unanswerable item in W9C2's dataset: you have to make
"no answer" a legitimate output or you will never get one.

---

## Given, `verify_citations`

Return the set of cited ids that actually appear among the retrieved chunks.

**An invented citation is worse than none**, because it carries the appearance of
evidence. Models cite fluently and inaccurately, and checking mechanically costs
a few lines. Most RAG demos skip this step, which is exactly why it is here.

Note the verification is shallow: it checks that a cited chunk *was retrieved*,
not that it *supports the claim*. The deeper check (does chunk 2 actually entail
this sentence?) is an open problem, and it is worth saying so rather than
implying the citation check makes the answer true.

---

## Running it

```
Q: How does RAG reduce hallucination?
  retrieved: ['week11-rag.md', 'week11-rag.md', 'week10-prompting.md']
  answer: RAG (Retrieval-Augmented Generation) reduces hallucination by allowing a model
          to use fresh or private knowledge without retraining...
  valid citations: [1, 2, 3]

Q: What does temperature do during decoding?
  retrieved: ['week07-decoding.md', 'week11-rag.md', 'week11-rag.md']
  answer: I don't know.
  valid citations: []
```

**Teach the third query, not the second.** The retriever found the *correct*
document (`week07-decoding.md`) and the model still said "I don't know". Students
will read that as a failure. It is the system working:

- The retrieved chunk did not contain enough to answer the question.
- The grounding instruction told the model to abstain rather than fill the gap
  from its parameters.
- So it abstained.

**The trade RAG makes** is converting "confidently wrong" into "honestly
unhelpful". That is usually the better failure mode, and it moves the bottleneck:
answer quality is now capped by **retrieval** quality. If you want a better
answer to that question, you improve the chunking or the retriever, not the
prompt.

**Two other things visible in the output.**

The third retrieved chunk is frequently irrelevant, because `k=3` always returns
three chunks whether or not three are relevant. Same fixed-`k` problem students
met in W3C1 and W3C2. Real systems apply a score threshold.

The first query retrieves `week10-prompting.md` for a CoT question and answers
correctly with citation `[1]`. Retrieval that spans documents is the normal case,
and it is why chunk ids are global.

**Connecting to the CTF next door.** W11C1's lesson was that model output must be
validated before anything downstream trusts it. `verify_citations` is that same
control applied to a claim of evidence. And the stretch goal in W11C1 (indirect
injection via a retrieved record) is precisely an attack on *this* pipeline: if a
chunk in your corpus contains "ignore previous instructions", RAG hands it to the
model as trusted context. Worth mentioning here so students see the two halves of
the week connect.
