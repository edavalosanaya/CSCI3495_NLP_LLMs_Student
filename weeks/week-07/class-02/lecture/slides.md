---
marp: true
title: "W7C2: Scaling Laws & Emergent Abilities"
paginate: true
---

# Week 7, Class 2

<!-- layout: title -->

Scaling Laws & Emergent Abilities

Agenda: lecture, Quiz 7, break, activities, recap

<!-- img: visuals/assets/photo-chinchilla.jpg -->
<!-- caption: Meet the chinchilla. Ask the class: why is one of the most important LLM papers of 2022 named after this rodent? -->
<!-- source: Guerin Nicolas, Wikimedia Commons, CC BY-SA 3.0 -->

---

# The central question

<!-- img: visuals/central-question.png -->
<!-- source: original figure -->

---

# Three axes of scale

<!-- img: visuals/three-axes.png -->
<!-- source: original figure -->

---

# Scale costs real hardware

<!-- img: visuals/assets/photo-datacenter.jpg -->
<!-- caption: More parameters and more data mean more of this: racks of GPUs running for weeks. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1558494949-ef010cbdcc31 -->

---

# If you doubled your compute budget, where should it go: a bigger model or more data?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-scales.jpg -->
<!-- caption: Parameters on one pan, training tokens on the other. Chinchilla settles the balance soon. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1589829545856-d10d557cf95f -->

---

# Scaling laws (Kaplan et al., 2020)

<!-- img: visuals/assets/kaplan-2020-fig-1.png -->
<!-- caption: One of the most reproduced plots in modern ML. A straight line on log-log axes IS a power law; each panel holds when the other two factors are not the bottleneck. Note the exponents: loss falls slowly, so gains cost orders of magnitude. -->
<!-- source: Kaplan et al. 2020, "Scaling Laws for Neural Language Models" (arXiv:2001.08361), Fig. 1 -->

- Forecasts a big model's loss from small runs, within the studied compute range; extrapolation far beyond it is unreliable

---

# Why power laws matter

<!-- img: visuals/why-powerlaws.png -->
<!-- source: original figure -->

---

# Kaplan's original takeaway

<!-- img: visuals/assets/kaplan-2020-fig-3.png -->
<!-- caption: Kaplan's advice for a billion-fold compute increase: pour almost all of it into model size, barely any into data. This chart is why GPT-3 got 175B parameters but only about 300B tokens. Chinchilla revisits this split next. -->
<!-- source: Kaplan et al. 2020, "Scaling Laws for Neural Language Models" (arXiv:2001.08361), Fig. 3 -->

---

# The Chinchilla result

<!-- img: visuals/chinchilla-result.png -->
<!-- source: Hoffmann et al. 2022, "Training Compute-Optimal LLMs" (arXiv:2203.15556) -->

---

# Compute-optimal scaling

<!-- img: visuals/assets/chinchilla-2022-fig-3.png -->
<!-- caption: Each U-curve fixes a FLOP budget and trades parameters against tokens: too big is as wasteful as too small. At Gopher's budget the valley bottoms read off 63B parameters and 1.4T tokens, the recipe that became Chinchilla (70B). -->
<!-- source: Hoffmann et al. 2022, "Training Compute-Optimal Large Language Models" (arXiv:2203.15556), Fig. 3 -->

---

# Emergent abilities

<!-- img: visuals/emergence.png -->
<!-- source: original figure; Wei et al. 2022, "Emergent Abilities of Large Language Models" -->

---

# Emergence: nuance & debate

<!-- img: visuals/emergence-debate.png -->
<!-- source: original figure; Schaeffer et al. 2023 on metric artifacts -->

---

# Loss ≠ usefulness

<!-- img: visuals/loss-vs-useful.png -->
<!-- source: original figure -->

---

# What "large" really buys you

<!-- img: visuals/large-buys.png -->
<!-- source: original figure; GPT-3 family sizes per Brown et al. 2020; emergence per Wei et al. 2022 -->

---

# Quiz 7 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Quiz 7 runs right before the break: finish early and your break starts early. -->
<!-- source: original figure -->

- Clear your desk; covers Week 7 lecture + GPT-3 (Brown 2020)
- Finish early and your break starts early; need more time and it comes out of the break

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then Budget the compute at the whiteboard. -->
<!-- source: original figure -->

---

# Activity A: budget the compute

<!-- img: visuals/activity-budget.png -->
<!-- source: original figure; Chinchilla rule, Hoffmann et al. 2022 -->

---

# Activity B: structured debate

<!-- img: visuals/activity-debate.png -->
<!-- source: original figure -->

---

# Take a side and defend it

<!-- layout: statement -->
<!-- img: visuals/assets/photo-debate.jpg -->
<!-- caption: Is bigger always better? Is emergence real or a metric mirage? Argue your assigned position with evidence. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1475721027785-f74eccf877e2 -->

---

# Optional take-home: measure scaling

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-07/class-02/exercise/scaling.py -->

- Use simple arithmetic or strict-format tasks, not tiny MMLU runs near the 25% chance floor
- Tiny suites are noisy: read the direction of the trend, not one-item flips

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
