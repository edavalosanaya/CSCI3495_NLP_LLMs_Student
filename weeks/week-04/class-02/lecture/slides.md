---
marp: true
title: "W4C2: RNNs & LSTMs"
paginate: true
---

# Week 4, Class 2: Sequence Models

## RNN & LSTM lecture, then Quiz 4 (~10 min) right before the break, then the char-RNN lab

<!-- layout: title -->
<!-- img: visuals/assets/photo-canal-lock.jpg -->
<!-- caption: A canal lock moves a boat one gated chamber at a time, and each gate decides what flows through. Ask the class what plays the part of the water once we call this an LSTM. -->
<!-- source: Jaggery, Geograph Britain and Ireland, CC BY-SA 2.0 -->

---

# From averaging to reading in order

<!-- layout: statement -->
<!-- img: visuals/assets/photo-reading.jpg -->
<!-- caption: Today we read a sequence one step at a time, keeping a memory. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1541963463532-d68292c34b19 -->

---

# Last class threw away order

<!-- img: visuals/read-in-order.png -->
<!-- caption: Same six tokens; only the model on the right can tell who sat on what. -->
<!-- source: original figure -->

---

# The recurrent idea

<!-- img: visuals/rnn-unrolled.gif -->
<!-- caption: Watch one time step at a time: read the next character plus the incoming memory, update the memory, emit a prediction. Every box is the SAME network; the animation loops in slideshow mode. -->
<!-- source: original animation (visuals/_gen_rnn_gif.py) -->

---

# The book's formal view: one loop

<!-- img: visuals/assets/slp3-fig-13-1.png -->
<!-- caption: The whole animation collapses to this: the hidden layer's activation feeds back as part of its own next input. That single dashed loop is the memory. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.1 (after Elman 1990) -->

---

# What the hidden state is for

<!-- img: visuals/hidden-state.png -->
<!-- caption: h_t is a running summary of everything seen so far; the prediction at step t is a function of h_t. -->
<!-- source: original figure -->

---

# Job 1: language modeling and generation

<!-- img: visuals/assets/slp3-fig-13-9.png -->
<!-- caption: Autoregressive generation: sample a word, feed it back in as the next input, repeat. Exactly what your char-RNN dinosaur lab does after the break. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.9 -->

---

# Job 2: label every token

<!-- img: visuals/assets/slp3-fig-13-7.png -->
<!-- caption: Same loop, different head: emit one output PER time step and you get sequence labeling, here part-of-speech tags for "Janet will back the bill". -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.7 -->

---

# Job 3: classify the whole sequence

<!-- img: visuals/assets/slp3-fig-13-8.png -->
<!-- caption: Or keep ONLY the last hidden state, which has read everything, and hand it to last class's feedforward classifier. Spam filter, sentiment, topic: one vector in, one label out. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.8 -->

---

# The problem with vanilla RNNs

<!-- layout: section -->

---

# The vanishing gradient problem

<!-- img: visuals/vanishing.png -->
<!-- caption: Backprop through time multiplies many Jacobians; factors < 1 shrink the gradient to zero, > 1 explode it. -->
<!-- source: original figure -->

---

# Gating: the fix behind LSTMs

<!-- img: visuals/lstm-gates.png -->
<!-- thumb: LSTM networks (MATLAB) | https://www.youtube.com/watch?v=5dMXyiWddYs | visuals/assets/yt-lstm.jpg -->
<!-- caption: The LSTM (Hochreiter & Schmidhuber, 1997): a cell-state conveyor belt guarded by three sigmoid gates. Forget erases, input admits the tanh candidate, output reveals a filtered copy as h_t. -->
<!-- source: original figure -->

---

# Inside the LSTM cell

<!-- img: visuals/assets/slp3-fig-13-13.png -->
<!-- caption: The conveyor belt from the last slide is the top c path. Find the three sigmoid gates guarding it: f (forget old memory), i (admit the tanh candidate g), o (expose h_t). -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.13 -->

---

# The LSTM, in equations

<!-- img: visuals/lstm-eqs.png -->
<!-- caption: For reference, not memorization: eight lines, one per arrow of Fig. 13.13. Notice the pattern repeating: every gate is a sigmoid mask followed by an element-wise product. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 13.20-13.27 -->

---

# How different is an LSTM, really?

<!-- img: visuals/assets/slp3-fig-13-14.png -->
<!-- caption: From the outside, barely at all: a feedforward unit (a) takes x, an RNN unit (b) adds h_t-1, an LSTM unit (c) just adds the context line c. Same socket, smarter unit; the rest of the network never notices. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.14 -->

---

# Putting it together

<!-- layout: statement -->
<!-- img: visuals/assets/photo-puzzle.jpg -->
<!-- caption: Gates and recurrence combine into encoder-decoder sequence models. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1612611741189-a9b9eb01d515 -->

---

# One family, four architectures

<!-- img: visuals/assets/slp3-fig-13-15.png -->
<!-- caption: Everything today on one slide: the three jobs you already saw (a, b, c) plus the new one, (d) encoder-decoder: one RNN squeezes the input into a context, another unrolls the output. That is your reading. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.15 -->

---

# Why this matters: seq2seq

<!-- img: visuals/assets/slp3-fig-13-17.png -->
<!-- caption: Fig 13.15's panel (d) with real words: read "the green witch arrived", hand over ONE hidden state (the green h_n), and unroll Spanish from it. Your reading (Sutskever et al. 2014) built exactly this with LSTMs; feeding the source reversed lifted BLEU from 25.9 to 30.6. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 13.17 -->

---

# The context vector means something

<!-- img: visuals/assets/sutskever-2014-fig-2.png -->
<!-- caption: PCA of the encoder's final hidden states. Swap John and Mary and the point changes cluster (left); rewriting active as passive barely moves it (right). Word order matters, a bag of words could not do this. -->
<!-- source: Sutskever, Vinyals & Le 2014 (arXiv:1409.3215), Fig. 2 -->

---

# The bottleneck (and a teaser)

<!-- img: visuals/bottleneck.png -->
<!-- caption: One fixed vector must hold the whole input; translation quality measurably drops on long sentences. -->
<!-- source: original figure -->

---

# RNNs vs. what's coming

<!-- img: visuals/rnn-vs-transformer.png -->
<!-- caption: RNNs read step by step with limited parallelism; Transformers attend all at once and dominate LLMs. -->
<!-- source: original figure -->

---

# Quiz 4

<!-- img: visuals/quiz.png -->
<!-- caption: Clear your desk: closed-book quiz over Week 4 lecture + seq2seq (Sutskever et al., 2014), ~10 minutes. Finish early and your break starts early; need more time and it comes out of the break. -->
<!-- source: quizzes/quiz-04.md -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the char-RNN dinosaur name lab. -->
<!-- source: original figure -->

---

# Exercise

<!-- layout: section -->

---

# Can a network trained only on real dinosaur names invent brand-new ones?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-dinosaur.jpg -->
<!-- caption: Your char-level RNN is about to find out. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1606856110002-d0991ce78250 -->

---

# Exercise: a char-level RNN

<!-- img: visuals/exercise-code.png -->
<!-- caption: Fill in three TODOs, the RNN forward pass, training pairs, and autoregressive sampling. -->
<!-- source: weeks/week-04/class-02/exercise/char_rnn.py -->

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Loss drops and the RNN invents new dinosaur-shaped names, some real, some delightfully made up. -->
<!-- source: example run, seeded (torch.manual_seed(1)) -->

---

# Activity: vote on the best invented name

<!-- img: visuals/activity-name-vote.png -->
<!-- caption: Pairs, 5 min. Trade your favorite generated names and vote on the most plausible dinosaur. -->
<!-- source: original figure -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: RNNs loop a shared-weight cell with a hidden memory; LSTMs gate it; seq2seq motivates attention. -->
<!-- source: original figure -->

---

# Next class

<!-- img: visuals/next-up.png -->
<!-- caption: Next week: attention lets the decoder look back at all encoder states, and the Transformer drops recurrence. -->
<!-- source: original figure -->
