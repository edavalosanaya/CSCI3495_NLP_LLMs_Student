---
marp: true
title: "W3C2 Word Embeddings"
paginate: true
---

# Word Embeddings

<!-- layout: title -->
<!-- img: visuals/assets/photo-chess-kingqueen.jpg -->
<!-- caption: king minus man plus woman equals queen. By the end of class you can do this arithmetic on meaning. -->
<!-- source: Bubba73 (Jud McCranie), Wikimedia Commons, CC BY-SA 3.0 -->

## Week 3, Class 2: word2vec, GloVe, Analogies & Bias
Lecture (~25 min), Quiz 3 (~10 min) right before the break, then whiteboard analogy (~12 min) and discussion (~13 min)

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Figs. 5.7 & 5.9; Mikolov et al. 2013, Fig. 1; deck figures -->

---

# Every word gets a place in one shared space

<!-- layout: statement -->
<!-- img: visuals/assets/photo-stars.jpg -->
<!-- caption: Embeddings scatter words like stars; nearby points share meaning. -->
<!-- source: Unsplash License, https://unsplash.com/photos/Y20JJ_ddy9M (Vincentiu Solomon) -->

---

# From sparse counts to dense vectors

<!-- img: visuals/sparse-vs-dense.png -->
<!-- source: original figure -->

---

# word2vec

<!-- layout: section -->

---

# word2vec's trick: turn counting into a quiz

<!-- img: visuals/w2v-quiz-task.png -->
<!-- caption: Instead of counting neighbors, train a yes/no classifier on a fake task. The quiz is a means to an end. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Ch. 5.5 running example -->

---

# The labels are free: self-supervision

<!-- img: visuals/w2v-training-pairs.png -->
<!-- caption: One window of running text hands us 4 positive pairs; k random noise words per real pair supply the negatives. This is skip-gram with negative sampling (SGNS). -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Ch. 5.5.2 training example -->

---

# Score a pair: dot product, then sigmoid

<!-- img: visuals/w2v-dot-sigmoid.png -->
<!-- caption: Last class: similar words have similar vectors. word2vec runs it backwards: MAKE the dot product big for real neighbors, small for noise. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 5.13-5.15 -->

---

# How the magic happens

<!-- img: visuals/assets/slp3-fig-5-7.png -->
<!-- caption: Start with random vectors. One window of "...apricot jam..." becomes one gradient step: pull the true neighbor closer, push k random words away. Repeat over billions of windows. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 5.7 -->

- Negative sampling: k fake pairs per real one

---

# The word2vec family: skip-gram and CBOW

<!-- img: visuals/assets/mikolov-2013-fig-1.png -->
<!-- caption: What we just built is skip-gram (right): the center word predicts each neighbor. CBOW (left) flips the arrows. The trained projection weights ARE the word vectors. -->
<!-- source: Mikolov et al. 2013 (arXiv:1301.3781), Fig. 1 -->

---

# GloVe: factor global co-occurrence

<!-- img: visuals/glove.png -->
<!-- source: Pennington et al., 2014, GloVe -->

- Optional paper, never quizzed

---

# Analogies & bias

<!-- layout: section -->

---

# The famous result: analogies

<!-- img: visuals/analogy-vectors.gif -->
<!-- caption: The offset between man and woman is the same vector that joins king to queen. A 2D schematic. -->
<!-- source: original animation; Mikolov et al., 2013 -->

---

# It is not just king and queen

<!-- img: visuals/assets/slp3-fig-5-9.png -->
<!-- caption: Real GloVe vectors, projected to 2D. Parallel offsets appear for gender pairs (a) and even for grammar: comparative and superlative forms (b). Nobody programmed this in. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 5.9 (GloVe vectors, Pennington et al. 2014) -->

---

# Why this was a big deal

<!-- img: visuals/transfer.png -->
<!-- source: original figure -->

---

# Static embeddings: the limitation

<!-- img: visuals/static-limit.png -->
<!-- source: original figure -->

---

# Should a machine learn that "programmer" means a man?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-crowd.jpg -->
<!-- caption: Embeddings learn whatever the text assumes, including who belongs where. -->
<!-- source: Unsplash License, https://unsplash.com/photos/iFgRcqHznqg (Helena Lopes) -->

---

# Embeddings encode bias

<!-- img: visuals/embedding-bias.png -->
<!-- thumb: 3 types of bias in AI | https://www.youtube.com/watch?v=59bMh59JQDo | visuals/assets/yt-ai-bias.jpg -->
<!-- source: Bolukbasi et al., 2016 -->

- Vectors learn stereotypes present in the text

---

# Why bias matters

<!-- img: visuals/bias-pipeline.png -->
<!-- source: original figure -->

- Debiasing exists but is only a partial fix

---

# Quiz 3

<!-- img: visuals/quiz.png -->
<!-- caption: Covers Week 3 lecture + word2vec (Mikolov et al., 2013). Clear your desk. -->
<!-- source: original figure -->

- Finish early? Your break starts early
- Need more time? It comes out of the break

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the king minus man whiteboard challenge. -->
<!-- source: original figure -->

---

# Whiteboard & Discussion

<!-- layout: section -->

---

# Whiteboard: king minus man plus woman

<!-- img: visuals/activity-analogy.png -->
<!-- source: toy vectors from exercise/embeddings.py; Mikolov et al. 2013 -->

---

# The algorithm is just doing math. So who owns the bias?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-balance.jpg -->
<!-- caption: A Socratic discussion. Hold this question; it sets up Week 15. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1589829545856-d10d557cf95f -->

---

# Discussion: who owns the bias?

<!-- img: visuals/discussion-questions.png -->
<!-- source: Bolukbasi et al., 2016 -->

---

# Take-home lab: probe the embeddings

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-03/class-02/exercise/embeddings.py -->

- Optional: confirm in code what you computed by hand today

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
