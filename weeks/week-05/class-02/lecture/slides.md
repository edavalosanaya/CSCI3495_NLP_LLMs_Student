---
marp: true
title: "W5C2: The Transformer"
paginate: true
---

# Week 5, Class 2
## The Transformer: Self-Attention, Multi-Head & Positional Encoding

Recap (~10 min), Quiz 5 (~10 min), break, extended lab (~48 min). Project proposal due today (written submission only).

<!-- layout: title -->
<!-- img: visuals/assets/photo-lego-build.jpg -->
<!-- caption: Lab day: today you build, piece by piece, the block every modern LLM is made of. -->
<!-- source: Nenad Stojkovic, Wikimedia Commons, CC BY 2.0 -->

---

# Where we left off

<!-- img: visuals/where-left-off.png -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Self-attention

---

# Self-attention: the core move

<!-- img: visuals/assets/slp3-fig-8-3.png -->
<!-- caption: The book's example: building the representation of "it", the model attends hard to "chicken" and "road", the two things "it" could refer to. Every token does this over the whole context, at every layer. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.3 -->

---

# Only look left: causal self-attention

<!-- img: visuals/assets/slp3-fig-8-4.png -->
<!-- caption: A language model must not see its own answer: each position attends to itself and everything BEFORE it, never after. Keep this picture; the mask slide shows how it is enforced. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.4 -->

---

# One head, in equations

<!-- img: visuals/attention-head-math.png -->
<!-- caption: The per-token version (no matrix tricks yet): project x into its three roles, score against every earlier token, softmax, blend the values. Reference slide for the lab. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 8.10-8.14 -->

---

# Query, Key, Value from one input

<!-- img: visuals/self-attention.png -->
<!-- source: Vaswani et al. 2017, arXiv:1706.03762; original figure -->

---

# The same computation, drawn

<!-- img: visuals/assets/slp3-fig-8-5.png -->
<!-- caption: Computing a_3, step by numbered step: generate k, q, v; compare q_3 to the keys; scale by sqrt(d_k); softmax; weigh the values; sum; reshape with W^O. Match each arrow to an equation on the previous slide. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.5 -->

---

# Now parallelize: pack the tokens into one matrix X

<!-- img: visuals/parallel-math.png -->
<!-- caption: The per-token equations, rewritten for all N tokens at once: three multiplies make Q, K, V; one multiply scores every pair; mask, softmax, blend. No loop over positions anywhere. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 8.32-8.34 -->

---

# All pairs at once, and the causal mask

<!-- img: visuals/assets/slp3-fig-8-9-10.png -->
<!-- caption: In practice one matrix multiply QK^T scores every pair at once (top), but the upper triangle is the FUTURE. The mask sets those cells to -infinity, so after softmax they get exactly zero weight (bottom): no information leaks backwards from tokens the model has not generated yet. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Figs. 8.9-8.10 -->

---

# The parallel computation, drawn

<!-- img: visuals/assets/slp3-fig-8-11.png -->
<!-- caption: Top row: X times W^Q, W^K, W^V gives every query, key, and value. Bottom row: QK^T scores all pairs, the mask blanks the future, and the weighted sum of V finishes all N positions in one shot. Match each arrow to the equations on the previous slide. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.11 -->

---

# Scaled dot-product attention

<!-- img: visuals/scaled-dot-formula.png -->
<!-- source: Vaswani et al. 2017, arXiv:1706.03762, Eq. 1; original figure -->

---

# Why divide by sqrt(d_k)?

<!-- img: visuals/why-sqrt-dk.png -->
<!-- source: Vaswani et al. 2017, arXiv:1706.03762, sec. 3.2.1; original figure -->

---

<!-- layout: section -->

# Multi-head & position

---

# Multi-head attention is like a string section

<!-- layout: statement -->
<!-- img: visuals/assets/photo-orchestra.jpg -->
<!-- caption: Many players read the same score at once, each listening for something different, then blend. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1465847899084-d164df4dedc6 -->

---

# One attention unit, then h of them in parallel

<!-- img: visuals/assets/vaswani-2017-fig-2.png -->
<!-- caption: The paper's two diagrams. Left is exactly the formula from two slides ago as a circuit, and what you build first in the lab; right runs h copies on lower-dim projections, concats, and projects back. -->
<!-- source: Vaswani et al. 2017 (arXiv:1706.03762), Fig. 2 -->

---

# Multi-head, concretely

<!-- img: visuals/assets/slp3-fig-8-6.png -->
<!-- caption: The book's version with A = 4 heads: each head has its own W^Q, W^K, W^V, their outputs are concatenated and W^O projects back to model size. Same picture as the paper's, with the shapes written on every wire. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.6 -->

---

# Multi-head, in equations

<!-- img: visuals/multihead-math.png -->
<!-- caption: Three lines on top of the single-head math: A copies of the projections, A heads, one concat and one mix-down. Reference slide for milestone 2 of the lab. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Eqs. 8.15-8.19 -->

---

# What heads learn (intuition)

<!-- img: visuals/heads-learn.png -->
<!-- source: original figure -->

---

# But attention has no sense of order

<!-- img: visuals/no-order.png -->
<!-- source: original figure -->

---

# The simple fix: add position embeddings

<!-- img: visuals/assets/slp3-fig-8-14.png -->
<!-- caption: Give every position its own embedding and just ADD it to the token embedding; now "bites" at position 2 and "bites" at position 4 enter the stack as different vectors. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.14 -->

---

# The paper's version: sinusoids

<!-- img: visuals/positional-encoding.png -->
<!-- caption: Vaswani et al. use fixed sinusoids at many frequencies instead of learned positions; both work, and the sinusoids need no training. -->
<!-- source: Vaswani et al. 2017, arXiv:1706.03762, sec. 3.5; original figure -->

---

<!-- layout: section -->

# The full block & families

---

# One token's journey: the residual stream

<!-- img: visuals/assets/slp3-fig-8-1.png -->
<!-- caption: The book's mental model: each token flows UP its own column (the residual stream), and the attention and feedforward components ADD information to it as it rises. Attention is the only place columns talk to each other. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.1 -->

---

# The transformer block, precisely

<!-- img: visuals/assets/slp3-fig-8-7.png -->
<!-- caption: The block on the residual stream: layer norm, multi-head attention (the only part that reads other streams, dashed arrows), add; layer norm, feedforward, add. Stack N of these. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.7 -->

---

# What attention is FOR: moving information

<!-- img: visuals/assets/slp3-fig-8-8.png -->
<!-- caption: One arrow to remember: an attention head copies information from token A's stream into token B's stream. Everything else in the block just processes what has already arrived. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.8 -->

---

# The full Transformer, assembled

<!-- img: visuals/assets/vaswani-2017-fig-1.png -->
<!-- thumb: The Transformer (HF) | https://www.youtube.com/watch?v=H39Z_720T5s | visuals/assets/yt-transformer.jpg -->
<!-- caption: The most reproduced figure in modern NLP. Every piece you have met is here: embeddings plus positional encoding, multi-head attention, feed-forward, residual Add & Norm, stacked N=6 times per side. The decoder adds masked self-attention and cross-attention into the encoder. -->
<!-- source: Vaswani et al. 2017 (arXiv:1706.03762), Fig. 1 -->

---

# The last stop: the language modeling head

<!-- img: visuals/assets/slp3-fig-8-15.png -->
<!-- caption: Top of the stack: the final hidden vector is multiplied by the unembedding matrix to get one logit per vocabulary word, and a softmax turns the logits into next-word probabilities. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.15 -->

---

# The whole thing: a decoder-only LM

<!-- img: visuals/assets/slp3-fig-8-16.png -->
<!-- caption: Input token, embeddings plus positions, N stacked blocks, LM head, predicted next token. This is the architecture of GPT-style models, assembled entirely from today's parts. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.16 -->

---

# Information flows up, and only leftward

<!-- img: visuals/assets/slp3-fig-8-2.png -->
<!-- caption: The whole stack at once: each column rises through the blocks, and the arrows show attention pulling information from PRECEDING columns only. Deep layers can see everything any earlier token computed. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 8.2 -->

---

# Encoder, decoder, or both?

<!-- img: visuals/families.png -->
<!-- source: original figure -->

---

# Built for parallel hardware

<!-- img: visuals/assets/photo-parallel.jpg -->
<!-- caption: No recurrence means the whole sequence runs at once. GPUs love that, and so does scaling. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1518770660439-4636190af475 -->

---

# Why the Transformer won

<!-- img: visuals/why-won.png -->
<!-- caption: The result: the backbone of every modern LLM, from BERT and GPT onward. -->
<!-- source: original figure -->

---

# Quiz 5 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Covers Week 5 lecture + Attention Is All You Need (Vaswani 2017). Clear your desk; finish early and your break starts early, need more time and it comes out of the break. -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the extended attention-block lab. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Extended lab: the attention block

---

# Today is a full-period coding lab

<!-- layout: statement -->
<!-- img: visuals/assets/photo-coding.jpg -->
<!-- caption: After the recap and the quiz, the rest of class is yours to build the core of every LLM. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1515879218367-8466d910aaa4 -->

---

# Lab plan and checkpoint milestones

<!-- img: visuals/activity-lab.png -->
<!-- source: original figure -->

---

# Milestone build: scaled dot-product + multi-head

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-05/class-02/exercise/attention_lab.py -->

- Verify with the provided pytest suite at each milestone

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run, weeks/week-05/class-02/solutions/attention_lab.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->

---

# Reminders

<!-- img: visuals/reminders.png -->
<!-- source: original figure; Font Awesome Free (CC BY 4.0) -->
