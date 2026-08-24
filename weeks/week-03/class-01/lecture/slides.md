---
marp: true
title: "W3C1 Vector Semantics"
paginate: true
---

# Vector Semantics

<!-- layout: title -->
<!-- img: visuals/assets/photo-flock.jpg -->
<!-- caption: A starling is known by the flock it flies with. Can we learn a word the same way, from the words around it? -->
<!-- source: Skander Zarrad, Wikimedia Commons, CC BY-SA 4.0 -->

## Week 3, Class 1: TF-IDF, PPMI & Cosine Similarity
Lecture (~35 min), break (5), Relevance Race (~33 min)

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Figs. 5.2 & 5.5; deck figures; run of exercise/search.py -->

---

# Words as atomic symbols

<!-- img: visuals/one-hot.png -->
<!-- source: original figure -->

---

# The distributional hypothesis

<!-- layout: section -->

---

# How would you guess the meaning of a word you have never seen?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-library.jpg -->
<!-- caption: You read the words around it. So can a machine. -->
<!-- source: Unsplash License, https://unsplash.com/photos/wE37SqLAO9M (Eli Francis) -->

---

# "You shall know a word by the company it keeps"

<!-- img: visuals/distributional.png -->
<!-- source: J.R. Firth, 1957, A Synopsis of Linguistic Theory -->

- Words in similar contexts tend to have similar meanings

---

# Counting neighbors turns a word into numbers

<!-- img: visuals/assets/slp3-fig-5-2.png -->
<!-- caption: Count how often each context word appears near a target word: each row is already a vector. Cherry and strawberry look alike; digital and information look alike. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 5.2 -->

---

# Those numbers are coordinates

<!-- img: visuals/assets/slp3-fig-5-4.png -->
<!-- caption: Plot two of the dimensions from real Wikipedia counts and words become points: digital and information land in the same corner of the space. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 5.4 -->

---

# Words and documents as vectors

<!-- img: visuals/vector-space.png -->
<!-- source: original figure -->

---

# Raw counts mislead

<!-- img: visuals/raw-counts.png -->
<!-- source: original figure -->

- We want frequent-in-doc but rare-across-corpus words

---

# TF-IDF

<!-- img: visuals/tfidf.png -->
<!-- source: original figure -->

---

# PPMI: word-word association

<!-- img: visuals/ppmi.png -->
<!-- source: original figure -->

---

# Why angle, not distance?

<!-- img: visuals/assets/slp3-fig-5-5.png -->
<!-- caption: Frequent words get long vectors, so raw distance misleads. The ANGLE between cherry and information is large; between digital and information it is tiny. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 5.5 -->

---

# Cosine similarity

<!-- img: visuals/cosine-similarity.png -->
<!-- source: original figure -->

---

# Putting it together: similarity search

<!-- img: visuals/search-pipeline.png -->
<!-- source: original figure -->

---

# This already runs in production

<!-- img: visuals/assets/photo-search.jpg -->
<!-- caption: Vector similarity powers search and retrieval inside real data centers. -->
<!-- source: Unsplash License, https://unsplash.com/photos/woman-in-black-coat-tCTLkInyXVw (Christina @ wocintechchat.com) -->

---

# Sparse vs. dense

<!-- img: visuals/sparse-vs-dense.png -->
<!-- thumb: Embeddings (Google) | https://www.youtube.com/watch?v=my5wFNQpFO0 | visuals/assets/yt-embeddings.jpg -->
<!-- source: original figure -->

- good and great stay orthogonal unless they co-occur

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the Relevance Race. -->
<!-- source: original figure -->

---

# Exercise

<!-- layout: section -->

---

# Activity: the Relevance Race

<!-- img: visuals/activity-race.png -->
<!-- source: weeks/week-03/class-01/exercise/search.py -->

---

# Build a TF-IDF search engine

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-03/class-01/exercise/search.py -->

- Rank tiny documents by similarity to a query

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run of solutions/search.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
