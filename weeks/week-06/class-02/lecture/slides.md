---
marp: true
title: "W6C2: BERT & Masked Language Modeling"
paginate: true
---

<!-- layout: title -->

# BERT & Masked Language Modeling

## Week 6, Class 2: Bidirectional pretraining and fine-tuning

<!-- img: visuals/assets/photo-hangman-wide.jpg -->
<!-- caption: You have been doing masked language modeling since elementary school. What word is _AN__AN, and how did your brain fill the blanks? -->
<!-- source: "Hangman game," Wikimedia Commons, CC0 -->

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- source: figures: Devlin et al. 2019 Figs. 1 & 3; Jay Alammar (jalammar.github.io); letter-dice photo: Unsplash -->

---

# Recap: shallow vs. deep bidirectionality

<!-- img: visuals/assets/bert-2019-fig-3.png -->
<!-- caption: ELMo tapes two one-way LSTMs together; in BERT every layer sees both sides at once. -->
<!-- source: Devlin et al. 2019 (arXiv:1810.04805), Fig. 3 -->

---

# If you could read the whole sentence at once, would you still go left to right?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-circuit.jpg -->
<!-- caption: Transformers read every token in parallel. BERT uses that to look both ways at once. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1518770660439-4636190af475 -->

---

<!-- layout: section -->

# BERT

---

# BERT (Devlin et al., 2019)

<!-- img: visuals/bert-overview.png -->
<!-- caption: Bidirectional Encoder Representations from Transformers, NAACL 2019 (posted Oct 2018). -->
<!-- source: Devlin et al. 2019, arXiv:1810.04805 -->

---

# Objective 1: Masked LM

<!-- img: visuals/masked-lm-anim.gif -->
<!-- caption: 80% become [MASK], 10% a random word, 10% are left alone and still predicted. -->
<!-- source: original animation; Devlin et al. 2019, arXiv:1810.04805 -->

- Now every layer can attend in both directions safely

---

# The 15% masking recipe

<!-- img: visuals/masking-recipe.png -->
<!-- source: Devlin et al. 2019, arXiv:1810.04805 -->

---

# Where the training signal comes from

<!-- img: visuals/assets/slp3-fig-9-3.png -->
<!-- caption: Loss flows only through the 3 selected tokens; "the" was swapped for "apricot," yet the model must still recover "the." -->
<!-- source: Jurafsky & Martin, SLP3 (Jan 2026 draft), Fig. 9.3 -->

---

# Objective 2: Next Sentence Prediction

<!-- img: visuals/nsp.png -->
<!-- caption: Predict whether segment B really follows segment A (50/50). -->
<!-- source: Devlin et al. 2019, arXiv:1810.04805 -->

---

<!-- layout: section -->

# Inputs & fine-tuning

---

# BERT's input representation

<!-- img: visuals/assets/bert-2019-fig-2.png -->
<!-- caption: Three embeddings summed per token; the A/B segment row is what lets ONE encoder take sentence pairs. -->
<!-- source: Devlin et al. 2019 (arXiv:1810.04805), Fig. 2 -->

---

# The [CLS] token

<!-- img: visuals/cls-token.png -->
<!-- source: Devlin et al. 2019, arXiv:1810.04805 -->

---

# WordPiece tokenization

<!-- img: visuals/wordpiece.png -->
<!-- source: original figure (Devlin et al. 2019) -->

- We go deeper on subword tokenization (BPE) in Week 8

---

# One pretrained model, every task

<!-- img: visuals/assets/bert-2019-fig-1.png -->
<!-- caption: Everything transfers except the tiny output layer; MNLI, NER and SQuAD all start from the same weights. -->
<!-- source: Devlin et al. 2019 (arXiv:1810.04805), Fig. 1 -->

---

# BERT-base vs. BERT-large

<!-- img: visuals/base-vs-large.png -->
<!-- caption: 110M vs. 340M parameters; both dwarf ELMo and both fine-tune fast. -->
<!-- source: Devlin et al. 2019, arXiv:1810.04805 -->

- Today we use a tiny BERT so it runs on a laptop CPU in seconds

---

# BERT's results

<!-- img: visuals/bert-results.png -->
<!-- source: Devlin et al. 2019, arXiv:1810.04805 -->

---

<!-- layout: section -->

# Encoders, decoders & the BERT family

---

# Encoder vs. decoder models

<!-- img: visuals/encoder-vs-decoder.png -->
<!-- source: original figure -->

- BERT cannot generate fluently; it is built to understand

---

# The BERT family

<!-- img: visuals/bert-family.png -->
<!-- source: original figure -->

---

# Quiz 6 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Clear your desk. Finish early and your break starts early; need more time and it comes out of the break. -->
<!-- source: original figure -->

- Covers Week 6 lecture + BERT (Devlin 2019)
- Finish early and your break starts early; need more time and it comes out of the break

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then MASK roulette in pairs. -->
<!-- source: original figure -->

---

<!-- layout: section -->

# Activity + exercise: MLM then fine-tuning

---

# Can you out-guess the model?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-letter-game.jpg -->
<!-- caption: Before we fine-tune, a quick competition: predict what BERT will put in the blank. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1611996575749-79a3a250f948 -->

---

# Activity: MASK roulette

<!-- img: visuals/activity-mask-roulette.png -->
<!-- source: original figure -->

---

# Two functions to implement

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-06/class-02/exercise/bert_mlm.py -->

- Fill the [MASK] with a fill-mask pipeline; fine-tune a tiny BERT classifier

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run, prajjwal1/bert-tiny (illustrative values) -->

- Top [MASK] predictions, then test accuracy after a few epochs

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
