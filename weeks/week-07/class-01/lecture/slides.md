---
marp: true
title: "W7C1: The GPT Family & Decoding Strategies"
paginate: true
---

# Week 7, Class 1

<!-- layout: title -->

The GPT Family & Autoregressive Generation

<!-- img: visuals/assets/photo-fork-road.jpg -->
<!-- caption: Two roads diverged in a wood. A language model hits a fork like this at EVERY word, with fifty thousand branches. Who picks? -->
<!-- source: Ann Cook, geograph.org.uk via Wikimedia Commons, CC BY-SA 2.0 -->

---

# Today's path

<!-- img: visuals/agenda.png -->
<!-- source: photos: Unsplash License; figures: SLP3 Fig. 7.11, Brown et al. 2020 Fig. 1.2, course figures -->

---

# Recap: BERT vs. GPT

<!-- img: visuals/bert-vs-gpt.png -->
<!-- source: original figure -->

- This week we focus on the generation side

---

# Three ways to wire a Transformer

<!-- img: visuals/assets/slp3-fig-7-3.png -->
<!-- caption: Decoders (GPT) attend only leftward and generate tokens; encoders (BERT) attend in both directions and output representations; encoder-decoders (the original Transformer) encode a full input, then a decoder generates from it. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 7.3 -->

---

# The GPT family

<!-- img: visuals/gpt-family.png -->
<!-- source: original figure; params per Radford et al. 2018/2019 & Brown et al. 2020 -->

- Each release: bigger model, more data, same core recipe

---

# Autoregressive generation

<!-- img: visuals/autoregressive.png -->
<!-- source: original figure -->

---

# One word at a time

<!-- img: visuals/assets/photo-writing.jpg -->
<!-- caption: Like a pen filling a page, the model commits to one token, then conditions on it for the next. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1455390582262-044cdead277a -->

---

# If the model only predicts probabilities, who actually picks the word?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-dice.jpg -->
<!-- caption: A decoding strategy is the dice roll: it turns one distribution into the next token. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1605870445919-838d190e8e1b -->

---

# The engine under generation

<!-- img: visuals/assets/slp3-fig-7-1.png -->
<!-- caption: The model reads the context and outputs a probability for every possible next word. Pick one, append it to the context, and run the model again: that loop IS autoregressive generation. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 7.1 -->

---

# The decoding problem

<!-- img: visuals/decoding-strategies.png -->
<!-- source: original figure -->

- Each step: a distribution over tens of thousands to hundreds of thousands of tokens, depending on the tokenizer. How do you choose?

---

# Greedy decoding

<!-- img: visuals/greedy.png -->
<!-- source: original figure -->

---

# Beam search

<!-- img: visuals/beam-search.gif -->
<!-- caption: Width 2, with real cumulative log-probs. Greedy's first pick loses to the sequence that finishes better. -->
<!-- source: original animation; original figure -->

---

# Sampling

<!-- img: visuals/sampling.png -->
<!-- source: original figure -->

---

# Top-k sampling

<!-- img: visuals/topk.png -->
<!-- source: original figure -->

---

# Top-p (nucleus) sampling

<!-- img: visuals/sampling-knobs.gif -->
<!-- caption: One softmax, swept from cold to hot. The nucleus is recomputed every frame. -->
<!-- source: original animation; original figure; nucleus sampling, Holtzman et al. 2020 -->

---

# Temperature in action

<!-- img: visuals/assets/slp3-fig-7-11.png -->
<!-- caption: Follow the "all" row: 0.95 at tau 0.1, 0.25 at tau 100. Temperature never reorders the tokens, it only changes how much the ranking matters. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 7.11 -->

---

# Putting it together

<!-- img: visuals/strategy-table.png -->
<!-- source: original figure -->

---

# The repetition problem

<!-- img: visuals/repetition.png -->
<!-- source: original figure -->

---

# Activity: predict before you run

<!-- img: visuals/activity-tps.png -->
<!-- source: original figure -->

---

# Predict, then prove it

<!-- layout: statement -->
<!-- img: visuals/assets/photo-dice.jpg -->
<!-- caption: Think (1 min), pair (2 min), share (2 min): which knob makes the output loop? Then run decoding.py and check. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1605870445919-838d190e8e1b -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then decoding.py and the Ollama playground to test your prediction. -->
<!-- source: original figure -->

---

# The required reading: GPT-3

<!-- img: visuals/gpt3-fewshot.png -->
<!-- source: Brown et al. 2020, "Language Models are Few-Shot Learners" (arXiv:2005.14165) -->

- Required, quizzed: GPT-3 (Brown 2020)
- Optional, not quizzed: Chinchilla, Kaplan, GPT-2

---

# Bigger models squeeze more from a prompt

<!-- img: visuals/assets/gpt3-2020-fig-1-2.png -->
<!-- caption: The task: strip random symbols from a word. All sizes see the same examples, but only 175B converts each extra example into a large accuracy jump. A steeper curve means better in-context learning, the theme of Week 10. -->
<!-- source: Brown et al. 2020, "Language Models are Few-Shot Learners" (arXiv:2005.14165), Fig. 1.2 -->

---

# Connecting to the exercise

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-07/class-01/exercise/decoding.py -->

- Then feel each knob on a real local LLM via Ollama

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run of weeks/week-07/class-01/solutions/decoding.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
