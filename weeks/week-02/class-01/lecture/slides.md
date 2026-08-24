---
marp: true
title: "W2C1: N-gram Language Models"
paginate: true
---

# N-gram Language Models

<!-- layout: title -->

## Week 2, Class 1: Probability, Chain Rule, Smoothing & Perplexity
Lecture (~30 min), whiteboard warm-up (~5 min), break (5), Babble-Off (~33 min)

<!-- img: visuals/assets/photo-magnetic-poetry.jpg -->
<!-- caption: Fridge poetry is a language model you play by hand: given the words so far, which tile do you reach for next? Today we make that instinct a probability. -->
<!-- source: Michael Zimmer, Wikimedia Commons, CC BY-SA 2.0 -->

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- caption: From "what is a language model?" to counting, smoothing, perplexity, and the Babble-Off. -->
<!-- source: deck figures; photos: Unsplash License; Wikimedia Commons, public domain -->

---

# You use a language model every day

<!-- img: visuals/assets/photo-phone-typing.jpg -->
<!-- caption: Phone autocomplete guesses your next word. Today we build the same idea from scratch. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1512941937669-90a1b58e7e9c -->

---

# What is a language model?

<!-- img: visuals/what-is-lm.png -->
<!-- caption: An LM assigns probability to the next word given the words so far, exactly what phone autocomplete does. -->
<!-- source: original figure -->

- It scores a whole sentence, or predicts the next word from the ones so far

---

# Why next-word prediction is central

<!-- img: visuals/why-central.png -->
<!-- caption: Predict the next word well and you implicitly learn grammar, facts, and style. -->
<!-- source: original figure -->

---

# If you can guess the next word, how much have you really learned?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-letter-dice.jpg -->
<!-- caption: Grammar, facts, and style all hide inside next-word prediction. -->
<!-- source: Wikimedia Commons, public domain, commons.wikimedia.org/wiki/File:Letra_Mix.JPG -->

---

# Probability is just counting

<!-- img: visuals/prob-basics.png -->
<!-- caption: P(event) = how often it happens divided by how many chances it had. 15 of 50 tweets contain "cat", so P = 0.30. -->
<!-- source: original figure -->

---

# Two events: AND, OR, and "given"

<!-- img: visuals/prob-and-or.png -->
<!-- caption: Intersection (AND) is the overlap, union (OR) adds both and subtracts the overlap, and conditional probability rescales to the world where B happened. -->
<!-- source: original figure -->

- P(A and B) = P(B) x P(A given B): apply this over and over and you get the chain rule

---

# The chain rule of probability

<!-- img: visuals/chain-rule.png -->
<!-- caption: Any joint probability factors exactly into a product of next-word probabilities. -->
<!-- source: original figure -->

---

# The Markov assumption

<!-- img: visuals/ngram-context.png -->
<!-- caption: Approximate the full history with the last n−1 words: unigram (0), bigram (1), trigram (2). -->
<!-- source: original figure -->

- This simplification is what makes counting feasible

---

# A corpus is just text we count

<!-- img: visuals/assets/photo-old-books.jpg -->
<!-- caption: An n-gram model learns nothing but counts of word sequences from a body of text like this. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1507842217343-583bb7270b66 -->

---

# Estimating probabilities by counting

<!-- img: visuals/counting.png -->
<!-- caption: Maximum-likelihood bigram estimate, count the pair, divide by the context count. -->
<!-- source: original figure -->

---

# A tiny worked example

<!-- img: visuals/worked-example.png -->
<!-- caption: Three short sentences give us real bigram probabilities by counting. -->
<!-- source: example after Jurafsky & Martin, SLP3 Ch. 3 -->

- Multiply the right bigrams to score a whole sentence

---

# Generating text from an n-gram model

<!-- img: visuals/generate.png -->
<!-- caption: Sampling = start at <s>, draw the next word, slide the window, repeat until </s>. -->
<!-- source: original figure -->

---

# Two vocabulary words: deterministic vs. stochastic

<!-- img: visuals/stochastic-deterministic.png -->
<!-- caption: Edit distance is deterministic (same input, same answer forever); sampling from a language model is stochastic (same prompt, different babble each run). -->
<!-- source: original figure -->

---

# The zero-probability problem

<!-- img: visuals/zero-prob.png -->
<!-- caption: One unseen bigram zeroes the whole sentence and sends perplexity to infinity. -->
<!-- source: original figure -->

---

# Add-one (Laplace) smoothing

<!-- img: visuals/add-one.png -->
<!-- caption: Pretend every n-gram was seen one extra time, so no probability is ever exactly 0. -->
<!-- source: original figure -->

---

# Perplexity: how good is the model?

<!-- img: visuals/perplexity-meter.png -->
<!-- thumb: What is perplexity? | https://www.youtube.com/watch?v=NURcDHhYe98 | visuals/assets/yt-perplexity.jpg -->
<!-- caption: Perplexity = inverse probability of the test set per word; the model's average surprise. -->
<!-- source: original figure -->

- Lower is better, like an effective branching factor

---

# Perplexity meets bias and variance

<!-- img: visuals/perplexity-bias-variance.png -->
<!-- caption: Small n underfits (high bias, both perplexities high); large n overfits the training counts (high variance, test perplexity climbs). Report perplexity on held-out text and pick the sweet spot. -->
<!-- source: original figure -->

---

# Where n-grams break down

<!-- img: visuals/breakdown.png -->
<!-- caption: Long-range dependencies, sparsity, and no generalization across similar words. -->
<!-- source: original figure -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: LM → chain rule + Markov → counting → smoothing → perplexity. -->
<!-- source: original figure -->

---

# Warm-up: compute a bigram by hand

<!-- img: visuals/activity-warmup.png -->
<!-- caption: Table teams, whiteboard, 5 minutes. Hand-compute one add-one bigram probability before we code. -->
<!-- source: example after Jurafsky & Martin, SLP3 Ch. 3 -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the Babble-Off. -->
<!-- source: original figure -->

---

# Activity: the Babble-Off

<!-- img: visuals/activity-babbleoff.png -->
<!-- caption: Teams, programming competition, ~33 minutes. Lowest perplexity and funniest babble win. -->
<!-- source: weeks/week-02/class-01/exercise/ngram_lm.py -->

---

# Exercise: build an N-gram Bard

<!-- img: visuals/exercise-task.png -->
<!-- caption: Implement four functions, then train, babble, and score on a tiny corpus. -->
<!-- source: weeks/week-02/class-01/exercise/ngram_lm.py -->

---

# Exercise: the starter code

<!-- img: visuals/exercise-code.png -->
<!-- caption: Four TODOs in ngram_lm.py, each maps to one idea from today's lecture. -->
<!-- source: weeks/week-02/class-01/exercise/ngram_lm.py -->

- Run it, then `pytest test_ngram_lm.py`

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Perplexity drops from unigram to bigram; the babble grows more fluent as n rises. -->
<!-- source: example run, solutions/ngram_lm.py -->

- Lower perplexity = better. HW1 is out today.
