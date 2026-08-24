---
marp: true
title: "W8C1: Tokenization at Scale & Instruction Tuning"
paginate: true
---

# Week 8, Class 1
## Tokenization at Scale, Pretraining Data & Instruction Tuning

<!-- layout: title -->
<!-- img: visuals/assets/photo-lego-bricks.jpg -->
<!-- caption: An LLM never sees words. It sees bricks. Today: where the bricks come from, and what they cost. -->
<!-- source: Alan Chia, Wikimedia Commons, CC BY-SA 2.0 -->

---

# Today

<!-- img: visuals/roadmap.png -->
<!-- caption: From how a tokenizer learns, to why tokens cost, to turning a base model into an assistant. HW4 is released today. -->
<!-- source: tiles: SLP3 Fig. 2.6 (Jurafsky & Martin); course figures; Unsplash -->

---

# Where tokenization sits

<!-- img: visuals/tokenization-pipeline.png -->
<!-- caption: Word-level explodes the vocabulary; character-level explodes sequence length. LLMs take a learned middle path: subwords. -->
<!-- source: original figure -->

---

<!-- layout: section -->
# Byte-Pair Encoding

---

# How BPE learns a vocabulary

<!-- img: visuals/assets/slp3-fig-2.6.png -->
<!-- caption: The loop you implement today: k merges, each adding exactly one new token to V. The ordered merge list IS the tokenizer. -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 2.6, adapted from Bostrom & Durrett 2020 -->

---

# A worked example

<!-- img: visuals/bpe-merges.png -->
<!-- caption: Each merge adds exactly one token. Frequent sequences become single tokens; rare strings stay in pieces. -->
<!-- source: original figure -->

---

# Encoding new text

<!-- img: visuals/encode-newword.png -->
<!-- caption: Apply the learned merges in the order they were learned. Unseen words still encode, with no out-of-vocabulary problem, ever. -->
<!-- source: original figure -->

---

# Byte-level BPE & cousins

<!-- img: visuals/bpe-variants.png -->
<!-- caption: Starting from raw bytes makes the vocabulary universal. WordPiece and Unigram are the common alternatives. -->
<!-- source: GPT-2 (Radford et al. 2019); original figure -->

---

# The hidden cost: fertility

<!-- img: visuals/fertility.png -->
<!-- caption: The same meaning costs far more tokens in some languages, a quiet source of language inequality in LLMs. -->
<!-- source: original figure (illustrative numbers) -->

---

# One meaning, many scripts

<!-- img: visuals/assets/photo-languages.jpg -->
<!-- caption: A tokenizer trained mostly on English splits other languages into more pieces, so the same idea costs more tokens. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1543165796-5426273eaab3 -->

---

# Tokens are the unit of everything

<!-- img: visuals/tokens-unit.png -->
<!-- caption: Context windows, pricing, and scaling-law data budgets are all counted in tokens, not words. -->
<!-- source: original figure -->

---

<!-- layout: section -->
# From Base Model to Assistant

---

# Pretraining data

<!-- img: visuals/assets/photo-letters.jpg -->
<!-- caption: A base model reads a wall of text like this: trillions of filtered, deduplicated tokens scraped from the web and books. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1457369804613-52c61a468e7d -->

---

# What goes in, and what to fear

<!-- img: visuals/pretraining-data.png -->
<!-- caption: Duplication, contamination, and bias are baked-in hazards of the pretraining corpus. -->
<!-- source: original figure -->

---

# It only ever learned to predict the next word.

<!-- layout: statement -->
<!-- img: visuals/assets/photo-library.jpg -->
<!-- caption: So why does ChatGPT answer your questions? That gap is what the rest of today is about. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1481627834876-b7833e8f5570 -->

---

# The base model is not an assistant

<!-- img: visuals/base-not-assistant.png -->
<!-- caption: A pretrained model only predicts the next token; it learned the form of text, not the intent to be helpful. -->
<!-- source: original figure -->

---

# Instruction tuning

<!-- img: visuals/instruction-tuning.png -->
<!-- caption: Fine-tune on (instruction, response) pairs. FLAN and T0 showed this improves zero-shot generalization to unseen tasks. -->
<!-- source: FLAN, Wei et al. 2021 (arXiv:2109.01652) -->

---

# Instruction tuning vs. pretraining

<!-- img: visuals/pretrain-vs-sft.png -->
<!-- caption: Same next-token loss; radically different data is what changes the model's behavior. -->
<!-- source: original figure -->

---

# Where instruction data comes from

<!-- img: visuals/instruction-sources.png -->
<!-- caption: Human-written, templated, or model-generated; diversity and quality matter more than raw volume. -->
<!-- source: Self-Instruct (Wang et al. 2022); Alpaca (Taori et al. 2023) -->

---

# Looking ahead to alignment

<!-- img: visuals/ahead-alignment.png -->
<!-- caption: SFT makes a model follow instructions; RLHF and DPO make it match which answers humans prefer. -->
<!-- source: original figure -->

---

# Warm-up: be the BPE algorithm

<!-- img: visuals/activity-whiteboard.png -->
<!-- caption: On paper, in pairs: hand-run the first two merges on low, lower, newest before you write any code. -->
<!-- source: original figure; worked corpus after Sennrich et al. 2016 -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the BPE coding exercise. -->
<!-- source: original figure -->

---

# Exercise: train a BPE tokenizer

<!-- img: visuals/exercise-code.png -->
<!-- caption: Implement count_pairs, merge_pair, and train_bpe, the merge loop at the heart of every modern tokenizer. -->
<!-- source: weeks/week-08/class-01/exercise/bpe.py -->

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Ten learned merges on a tiny corpus, then encoding the unseen word "lowest" into subword pieces. -->
<!-- source: example run, weeks/week-08/class-01/solutions/bpe.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Subwords, fertility, tokens as the universal unit, and instruction tuning. Next class: alignment (RLHF, DPO) + Quiz 8. -->
<!-- source: original figure -->
