---
marp: true
title: "W1C2: Text Processing"
paginate: true
---

<!-- layout: title -->

# Text Processing: Tokenization, Regex & Edit Distance

<!-- img: visuals/assets/photo-alphabet-soup.jpg -->
<!-- caption: To a computer, raw text is alphabet soup: a pile of characters. Where do the words begin? -->
<!-- source: Thriving Vegetarian, CC BY 2.0, via Wikimedia Commons -->

Week 1, Class 2: lecture (~25 min), Quiz 1 (~10 min), break, whiteboard warm-up (~15 min), exercise (~20 min)

---

# It all starts with raw text

<!-- img: visuals/assets/photo-typewriter.jpg -->
<!-- caption: Every model begins with messy human text: handwriting, tweets, code, transcripts. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1455390582262-044cdead277a -->

---

# Why text processing?

<!-- img: visuals/pipeline.png -->
<!-- caption: Before any model, raw text must become discrete units a computer can count and embed. -->
<!-- source: original figure -->

---

# How would YOU split a sentence into "words"?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-open-book.jpg -->
<!-- caption: Is "don't" one token or two? What about "New York" or a hashtag? Hold that thought. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1457369804613-52c61a468e7d -->

---

<!-- layout: section -->

# Tokenization

---

# One word, three tokenizations

<!-- img: visuals/tokenization-granularity.png -->
<!-- caption: Characters, subwords, or words. Each granularity is a different trade-off. -->
<!-- source: original figure -->

---

# Normalization

<!-- img: visuals/normalization.png -->
<!-- caption: Reducing variants to a canonical form is useful, but lossy. Pick the level that fits the task. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Regular Expressions

---

# Regular expressions

<!-- img: visuals/regex.png -->
<!-- caption: A compact language for pattern matching: metacharacters plus quantifiers, tested on real strings. We extract emails, URLs, and @mentions in the exercise. -->
<!-- source: original figure -->

---

# Write the regex

<!-- img: visuals/regex-exercise.png -->
<!-- caption: Think, then pair (5 min). Sketch a pattern for each target, then test it in the coding exercise. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Minimum Edit Distance

---

# Did you mean...?

<!-- img: visuals/assets/photo-dictionary.jpg -->
<!-- caption: Spell-checkers, fuzzy search, and DNA alignment all ask: how far apart are two strings? -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8 -->

---

# Minimum edit distance

<!-- img: visuals/edit-distance.gif -->
<!-- caption: Fewest insert, delete, or substitute edits, filled in by dynamic programming. Each edit costs 1 here (Levenshtein), so the answer is 5; J&M charge 2 per substitution and get 8. -->
<!-- source: original animation (Levenshtein; J&M SLP3 Ch. 2) -->

- Uses: spell-check, fuzzy matching, word-error-rate, diffing

---

# Edit distance: the recurrence

<!-- img: visuals/recurrence.png -->
<!-- caption: Each cell is the minimum of three neighbors (deletion, insertion, substitution). O(mn) time and space. -->
<!-- source: original figure -->

---

# A peek ahead: Byte-Pair Encoding

<!-- img: visuals/bpe.png -->
<!-- caption: Modern LLMs use learned subwords: start from characters, then merge the most frequent adjacent pair. -->
<!-- source: original figure (BPE: Sennrich et al., 2016) -->

- We build our own BPE tokenizer in Week 8

---

# Watch BPE merge, step by step

<!-- img: visuals/bpe-merge.gif -->
<!-- caption: The token sequence shrinks as the most frequent pair is merged, one step at a time. -->
<!-- source: original animation (BPE: Sennrich et al., 2016) -->

---

# Quiz 1 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Covers Week 1 lecture plus the reading (J&M Ch. 1 and 2). Clear your desk; about 10 minutes, right before the break. -->
<!-- source: original figure -->

- Covers Week 1 lecture and J&M Ch. 1 to 2
- Closed book; finish early and your break starts early, need more time and it comes out of the break

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the whiteboard warm-up. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Whiteboard, then Code

---

# Warm-up: be the tokenizer

<!-- img: visuals/activity-whiteboard.png -->
<!-- caption: Pairs, whiteboard, 15 minutes. Hand-tokenize three nasty strings, then trace the kitten to sitting DP table. -->
<!-- source: edit distance, J&M SLP3 Ch. 2 -->

---

# How every lab in this course works

<!-- img: visuals/lab-loop.png -->
<!-- caption: Read one step, write one function, run one test. Repeat. -->
<!-- source: original figure -->

---

# Enough Python for today

<!-- img: visuals/python-basics.png -->
<!-- source: weeks/week-01/class-02/exercise/text_tools.py -->

- The starter gives you the signature and the docstring; you write the body

---

# What is a test?

<!-- img: visuals/test-anatomy.png -->
<!-- caption: A test calls your function and compares the answer to the one the step promised. -->
<!-- source: weeks/week-01/class-02/exercise/test_text_tools.py -->

- Code that calls your function and checks the answer the step promised

---

# Green is not the goal, the message is

<!-- img: visuals/pytest-failure.png -->
<!-- caption: A failure tells you what you returned and what was wanted. That is a hint, not a grade. -->
<!-- source: pytest output, week 1 class 2 starter -->

---

# Now implement what you traced

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-01/class-02/exercise/text_tools.py -->

- Individual, ~20 min: turn your hand-traced logic into code
- Implement normalize, tokenize, extract, and edit_distance

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: weeks/week-01/class-02/exercise/test_text_tools.py -->

- All 10 tests green when your four functions are correct

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Text becomes data via tokenization, normalization, regex, and edit distance. -->
<!-- source: original figure -->
