---
marp: true
title: "W6C1: Contextual Representations & Transfer Learning"
paginate: true
---

<!-- layout: title -->

# Contextual Representations & Transfer Learning

## Week 6, Class 1: ELMo and the pretrain to finetune paradigm

<!-- img: visuals/assets/photo-chameleon.jpg -->
<!-- caption: A chameleon takes its color from its surroundings. Ask the class: which word in English does the same? -->
<!-- source: Charles J. Sharp, Wikimedia Commons, CC BY-SA 4.0 -->

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- source: photos: Wikimedia Commons (CC BY-SA 3.0), Unsplash; ELMo diagram: Jay Alammar (jalammar.github.io) -->

---

# Where we are in the course

<!-- img: visuals/course-roadmap.png -->
<!-- source: figures: SLP3 Figs. 5.9, 13.13; Vaswani 2017 Fig. 1; Jay Alammar (jalammar.github.io) -->

---

# Does a word mean the same thing every time you say it?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-conversation.jpg -->
<!-- caption: Meaning lives in context. Hold that thought as we leave static vectors behind. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1522202176988-66273c2fd55f -->

---

<!-- layout: section -->

# The problem with static embeddings

---

# Static word embeddings are frozen

<!-- img: visuals/static-frozen.png -->
<!-- caption: word2vec and GloVe: one fixed vector per word, looked up every time. -->
<!-- source: original figure -->

- One vector must average together all senses of a word

---

# A card catalog: one fixed card per word

<!-- img: visuals/assets/photo-card-catalog.jpg -->
<!-- caption: A static model is a card catalog: look up "bank", get the SAME card back every time, no matter which sentence you came from. -->
<!-- source: Dr. Marcus Gossler, Wikimedia Commons, CC BY-SA 3.0 -->

---

# The polysemy problem

<!-- img: visuals/static-vs-contextual.png -->
<!-- caption: "river bank" and "savings bank" collapse to the same static vector. -->
<!-- source: original figure -->

- The meaning lives in the context, which static vectors throw away

---

<!-- layout: section -->

# Contextual representations

---

# ELMo's architecture: two LMs, forward and backward

<!-- img: visuals/assets/elmo-alammar-bilm.png -->
<!-- caption: Yes, that is THE Elmo, the diagram the internet learned this from. A forward LSTM LM reads left to right, a backward one reads right to left, two layers each; every word gets hidden states from BOTH directions. -->
<!-- source: Jay Alammar, "The Illustrated BERT, ELMo, and co." (jalammar.github.io); ELMo: Peters et al. 2018, arXiv:1802.05365 -->

- The conceptual leap from word2vec: the hidden states ARE the representations

---

# One word, many vectors: mixing the LSTM layers

<!-- img: visuals/assets/elmo-alammar-embedding.png -->
<!-- caption: How the contextual embedding is built: concatenate each layer's forward and backward states, weight each layer (s_0, s_1, s_2 are learned per task), and sum. The result is the embedding of "stick" in THIS sentence. -->
<!-- source: Jay Alammar, "The Illustrated BERT, ELMo, and co." (jalammar.github.io); ELMo: Peters et al. 2018, arXiv:1802.05365 -->

- Bidirectional LSTM language model; representation = learned mix of all layers

---

# Why combine all the layers?

<!-- img: visuals/layer-roles.png -->
<!-- caption: ELMo's empirical finding: lower layers capture syntax, higher layers capture word sense. -->
<!-- source: Peters et al. 2018, arXiv:1802.05365 -->

---

# The famous "play" test

<!-- img: visuals/assets/elmo-2018-table-4.png -->
<!-- caption: GloVe's neighbors are stuck in sports; the biLM retrieves a sentence matching EACH sense of "play." -->
<!-- source: Peters et al. 2018 (arXiv:1802.05365), Table 4 -->

---

# "Shallow" bidirectionality

<!-- img: visuals/shallow-bidir.png -->
<!-- caption: A forward LM and a backward LM, trained independently, then concatenated. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Pretrain, then finetune

---

# Feature-based transfer

<!-- img: visuals/transfer-learning.png -->
<!-- caption: Pretrain once on cheap unlabeled text; reuse the features across many tasks. -->
<!-- source: original figure -->

---

# Two flavors of transfer

<!-- img: visuals/feature-vs-finetune.png -->
<!-- caption: Feature-based (ELMo) freezes the model; fine-tuning (BERT, GPT) keeps training it. -->
<!-- source: original figure -->

- Both start from self-supervised language-model pretraining

---

# Self-supervised pretraining

<!-- img: visuals/self-supervised.png -->
<!-- caption: The training signal comes from the text itself: predict the missing word. -->
<!-- source: original figure -->

---

# Why this was a turning point

<!-- img: visuals/before-after.png -->
<!-- caption: 2018 flipped the default: start from a pretrained model, then finetune. -->
<!-- source: original figure -->

---

# ELMo's results

<!-- img: visuals/assets/elmo-2018-table-1.png -->
<!-- caption: One frozen biLM, six different tasks: every baseline jumps past the prior state of the art. -->
<!-- source: Peters et al. 2018 (arXiv:1802.05365), Table 1 -->

---

# Limitations of ELMo

<!-- img: visuals/elmo-limits.png -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the jigsaw teach-back in teams of three. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Activity: jigsaw teach-back

---

# Teach each other the paradigm

<!-- layout: statement -->
<!-- img: visuals/assets/photo-team.jpg -->
<!-- caption: Each group owns one piece, becomes its expert, then teaches the rest of the team. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1522071820081-009f0129c71c -->

---

# Activity: jigsaw the pretrain to finetune story

<!-- img: visuals/activity-jigsaw.png -->
<!-- source: original figure -->

---

# Confirm it live: the polysemy demo

<!-- img: visuals/exercise-output.png -->
<!-- source: example run, prajjwal1/bert-tiny (illustrative values) -->

- Static cosine = 1.000 (identical); contextual cosine is lower (sense-dependent)
- The 3-function coding is now an optional take-home lab

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
