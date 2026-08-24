---
marp: true
title: "W11C2: Retrieval-Augmented Generation"
paginate: true
---

<!-- layout: title -->

# Retrieval-Augmented Generation (RAG)

Week 11, Class 2: sketch and lecture, Quiz 11 right before the break, then the RAG lab

<!-- img: visuals/assets/photo-open-book.jpg -->
<!-- caption: Closed book or open book: which exam would you rather take? Today we let the model look things up before it answers. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8 -->


---

# Today is an extended hands-on lab

<!-- img: visuals/lab-rag-milestones.png -->
<!-- caption: A quick paired sketch and lecture, Quiz 11 right before the break, then a full ~50 min build of RAG end to end with milestones. -->
<!-- source: weeks/week-11/class-02/exercise/rag.py; Jurafsky & Martin, SLP3, Fig. 11.13; Unsplash License photos -->

- Build the whole pipeline: chunk, index, retrieve, ground, cite, verify

---

# What the five moves actually do

<!-- img: visuals/five-moves.png -->
<!-- source: original figure; syllabus/SYLLABUS.md -->

---

# What is poisoning, here?

<!-- img: visuals/poisoning-primer.png -->
<!-- source: original figure; Greshake et al. 2023 (arXiv:2302.12173) -->

---

# Warm-up: sketch it with a partner

<!-- layout: statement -->
<!-- img: visuals/assets/photo-pipeline-sketch.jpg -->
<!-- caption: Pairs, ~10 min at the whiteboard. Sketch the five moves, then for each one plant a poison and write the counter that stops it. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1453928582365-b6ad33cbcf64 -->

---

# Activity: sketch the RAG pipeline

<!-- img: visuals/activity-pipeline-sketch.png -->
<!-- source: weeks/week-11/class-02/exercise/rag.py -->

- Mode whiteboard sketch, pairs, about 10 min before the build
- For each stage: one concrete poison, one counter that stops it

---

<!-- layout: section -->

# The problem & the idea

---

# Models don't know your facts

<!-- img: visuals/frozen-weights.png -->
<!-- caption: Three gaps that no amount of clever prompting can fix. -->
<!-- source: original figure -->

---

# Don't memorize the library. Learn to look things up.

<!-- layout: statement -->
<!-- img: visuals/assets/photo-library-archive.jpg -->
<!-- caption: RAG attaches a retriever to the LLM: fetch the right passage, then answer from it. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1521587760476-6c12a4b040da -->

---

# The idea: retrieve, then generate

<!-- img: visuals/assets/slp3-fig-11-13.png -->
<!-- caption: Two separable stages: an off-the-shelf retriever plus an off-the-shelf LLM, joined by nothing more than a prompt. This is exactly the pipeline you build in today's lab. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 11.13 -->

---

# The original RAG architecture

<!-- img: visuals/assets/rag-lewis-2020-fig-1.png -->
<!-- caption: The quiz paper's version goes further: retriever and generator are fine-tuned end to end, marginalizing over the top-K retrieved documents. The index is swappable memory, so knowledge updates without retraining. -->
<!-- source: Lewis et al. 2020 (arXiv:2005.11401), Fig. 1 -->

---

<!-- layout: section -->

# How RAG works, step by step

---

# Step 1: Index your documents

<!-- img: visuals/index-step.png -->
<!-- caption: Done once, offline. Re-run only when the documents change. -->
<!-- source: original figure -->

---

# Like a card catalog for your documents

<!-- img: visuals/assets/photo-card-catalog.jpg -->
<!-- caption: An index turns a pile of text into something you can search in milliseconds. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1524995997946-a1c2e315a42f -->

---

# Watch one question go through

<!-- img: visuals/rag-flow.gif -->
<!-- caption: Question to vector to top-k passages to a grounded answer. Scores are real cosine similarities. -->
<!-- source: original animation; original figure -->

---

# Dense vs. TF-IDF retrieval

<!-- img: visuals/dense-vs-sparse.png -->
<!-- caption: Meaning vs exact terms; both feed the same retrieve-then-generate pipeline. -->
<!-- source: original figure -->

---

# Step 3: Generate (grounded)

<!-- img: visuals/generate-step.png -->
<!-- caption: Answer from the retrieved evidence and cite the source chunk numbers. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Quality, evaluation & beyond

---

# Why RAG helps

<!-- img: visuals/why-rag.png -->
<!-- caption: Fresh knowledge, less hallucination, verifiable citations, and lower cost than retraining. -->
<!-- source: original figure -->

---

# Retrieval quality caps everything

<!-- img: visuals/retrieval-ceiling.png -->
<!-- caption: If the retriever misses the passage, the generator cannot recover. Evaluate it separately. -->
<!-- source: original figure -->

---

# Failure modes & mitigations

<!-- img: visuals/failure-modes.png -->
<!-- caption: Missed retrieval, distractors, ungrounded answers, lost-in-the-middle, and their fixes. -->
<!-- source: original figure -->

---

# RAG context poisoning

<!-- img: visuals/rag-poisoning.png -->
<!-- caption: Retrieved documents enter the prompt as untrusted text; a planted doc becomes an indirect injection. -->
<!-- source: Greshake et al. 2023, arXiv:2302.12173 -->

---

# The documents in your context are untrusted input.

<!-- layout: statement -->
<!-- caption: Anyone who can write to your corpus can write to your prompt. -->
<!-- source: Greshake et al. 2023, arXiv:2302.12173 -->

---

# Defending RAG against poisoning

<!-- img: visuals/rag-defenses.png -->
<!-- caption: Same defense-in-depth playbook as the last class, applied to the retrieval pipeline. -->
<!-- source: OWASP Top 10 for LLM Apps -->

---

# Evaluating RAG

<!-- img: visuals/evaluating.png -->
<!-- caption: Score retrieval (recall@k) and generation (faithfulness, citations) independently. -->
<!-- source: original figure -->

---

# Beyond basic RAG

<!-- img: visuals/beyond-rag.png -->
<!-- caption: Re-ranking, query rewriting, Self-RAG, and agentic retrieval as a tool. -->
<!-- source: Self-RAG: Asai et al. 2023, arXiv:2310.11511 -->

---

# Quiz 11 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Covers the Week 11 lectures + RAG (Lewis et al. 2020). Clear your desk. Finish early and your break starts early; need more time and it comes out of the break. -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the extended lab: build RAG end to end. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Exercise & wrap-up

---

# Extended lab: build RAG end to end

<!-- img: visuals/exercise-code.png -->
<!-- caption: About 50 min, three milestones: chunk and index, retrieve and ground, then cite and verify. -->
<!-- source: weeks/week-11/class-02/exercise/rag.py -->

- Work through the milestones; the warm-up sketch is your map

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: A grounded answer with a verified citation back to the source chunk. -->
<!-- source: example run, TF-IDF retriever + offline stub generator -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Retrieve then generate; index once; benefits and citations; retrieval is the ceiling. -->
<!-- source: original figure -->
