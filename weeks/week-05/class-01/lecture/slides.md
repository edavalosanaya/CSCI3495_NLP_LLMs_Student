---
marp: true
title: "W5C1: Seq2seq & Attention"
paginate: true
---

# Week 5, Class 1
## Seq2seq & Attention: The Encoder-Decoder & the Attention Mechanism

<!-- img: visuals/assets/photo-rosetta-stone.jpg -->
<!-- caption: The Rosetta Stone: one decree, three scripts. Decoding any line meant constantly glancing back at the others. Today we give neural translators that same glance. -->
<!-- source: Photo: Hans Hillewaert, Wikimedia Commons, CC BY-SA 4.0 -->

Lecture (~35 min), break (5), exercise (~35 min). HW2 due today.

<!-- layout: title -->

---

# Today

<!-- img: visuals/agenda.png -->
<!-- source: tiles: Unsplash; Jurafsky & Martin SLP3 Figs. 13.18, 13.22; Bahdanau et al. 2014 Figs. 2-3; course exercise output -->

---

# Translation: read one language, write another

<!-- img: visuals/assets/photo-translation.jpg -->
<!-- caption: A sentence in, a sentence out. The model must hold the whole meaning before it writes a word. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1456513080510-7bf3a84b82f8 -->

---

# Recap: seq2seq (Sutskever et al. 2014)

<!-- img: visuals/seq2seq-bottleneck.png -->
<!-- source: Sutskever, Vinyals & Le 2014, arXiv:1409.3215; original figure -->

---

# The bottleneck problem

<!-- img: visuals/assets/slp3-fig-13-17.png -->
<!-- caption: Same figure as last class, new reading: EVERYTHING the decoder will ever know about the English sentence must squeeze through the single green box h_n. Long sentence, same-size box. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.17 -->

---

# How would you write a long essay from one sticky note?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-spotlight.jpg -->
<!-- caption: That is the decoder's problem. Attention lets it shine a spotlight back on the input. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1514525253161-7a46d19cd819 -->

---

<!-- layout: section -->

# The idea: attention

---

# The setup, formally: one frozen context

<!-- img: visuals/assets/slp3-fig-13-18.png -->
<!-- caption: The basic encoder-decoder, drawn precisely: the green c = h_n is handed to EVERY decoder step, and it never changes. Attention's move: keep all the h_e states around and rebuild a fresh c_i for each output word. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.18 -->

---

# Attention in three steps

<!-- img: visuals/attention-align.gif -->
<!-- caption: Every output word re-weights the whole input. Watch the peak jump backwards when "chat noir" comes out reversed. -->
<!-- source: original animation; original figure -->

---

# The whole computation, drawn

<!-- img: visuals/assets/slp3-fig-13-22.png -->
<!-- caption: One decoding step: every encoder state gets a weight (here .4, .3, .1, .2 from dot products with the previous decoder state), and their weighted sum becomes c_i, an input to the next decoder state. Repeat fresh for every output word. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.22 -->

---

# The math, in four lines

<!-- img: visuals/attention-math.png -->
<!-- caption: Dot-product attention, straight from the book, and exactly what you will trace on the whiteboard and then code after the break. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 13.34-13.37 -->

---

# Additive (Bahdanau) attention: the math

<!-- img: visuals/additive-math.png -->
<!-- caption: The score is a tiny MLP, hence "additive." -->
<!-- source: Bahdanau, Cho & Bengio 2014, arXiv:1409.0473; original figure -->

---

# Query, keys, values

<!-- img: visuals/qkv.png -->
<!-- source: original figure -->

---

# What attention buys us

<!-- img: visuals/attention-buys.png -->
<!-- caption: Bahdanau et al. saw the largest translation gains on long sentences. -->
<!-- source: Bahdanau, Cho & Bengio 2014, arXiv:1409.0473; original figure -->

---

# The payoff: long sentences stop hurting

<!-- img: visuals/assets/bahdanau-2014-fig-2.png -->
<!-- caption: RNNsearch is the attention model, RNNenc the fixed-vector baseline; 30 vs 50 is the training sentence length. Only the attention models hold their BLEU past 30 words: the bottleneck was real, and this plot is the proof. -->
<!-- source: Bahdanau, Cho & Bengio 2014 (arXiv:1409.0473), Fig. 2 -->

---

# Reading attention as alignment

<!-- img: visuals/assets/bahdanau-2014-fig-3ad.png -->
<!-- caption: Real learned weights, two examples: rows are French output words, columns the English source, white means high alpha. Left: "zone economique europeenne" reverses "European Economic Area". Right: French says "a dit l'homme" (said the man), and the weights jump backwards to "the man said". Nobody labeled any of this. -->
<!-- source: Bahdanau, Cho & Bengio 2014 (arXiv:1409.0473), Fig. 3 (a, d) -->

---

# Additive vs. multiplicative scoring

<!-- img: visuals/additive-vs-dot.png -->
<!-- source: original figure -->

---

# Limitations that motivate the Transformer

<!-- img: visuals/rnn-limits.png -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the whiteboard hand-trace of attention with a partner. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Activity: trace attention, then build

---

# First trace it by hand on a 3-token toy

<!-- layout: statement -->
<!-- img: visuals/assets/photo-planning-board.jpg -->
<!-- caption: Fifteen minutes drawing the alignment with a partner, then a short build of the exact same thing. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1531403009284-440f080d1e12 -->

---

# Activity: hand-trace attention on 3 tokens

<!-- img: visuals/activity-trace-attention.png -->
<!-- caption: Every number you need is in the green box, including the exp values so no calculator is required. Answers get checked on the board, then you code this exact computation. -->
<!-- source: original figure -->

---

# Now build exactly what you traced

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-05/class-01/exercise/attention.py -->

- Individual, shortened: score, softmax, weighted sum
- Then render the weights as a text heatmap

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Two heatmaps on purpose: the untrained scorer is a uniform blur (every weight ~1/3), then a few hundred gradient steps make the diagonal emerge. Attention weights are learned. -->
<!-- source: example run, weeks/week-05/class-01/solutions/attention.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->

---

# Reminders

<!-- img: visuals/rnn-limits.png -->
<!-- caption: Next class: the Transformer drops recurrence. -->
<!-- source: original figure -->

- HW2 due today; project proposal due next class
- Required reading: Attention Is All You Need (Vaswani 2017)
- Optional, not quizzed: Bahdanau (2014)
