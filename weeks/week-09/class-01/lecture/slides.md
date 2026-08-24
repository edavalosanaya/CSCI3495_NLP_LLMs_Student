---
marp: true
title: "W9C1: Efficient LLMs: LoRA, PEFT & Quantization"
paginate: true
---

# Efficient LLMs: LoRA, PEFT & Quantization
## Week 9, Class 1: running big models on a laptop

<!-- layout: title -->
<!-- img: visuals/assets/photo-bonsai.jpg -->
<!-- caption: A bonsai is a full tree, deliberately kept tiny. Ask the class: what would you prune from a 7B-parameter model, and what must survive? -->
<!-- source: Sage Ross, Wikimedia Commons, CC BY-SA 3.0 -->

---

# Today's roadmap

<!-- img: visuals/roadmap.png -->
<!-- source: photos: Unsplash; LoRA figure: Hu et al. 2021 (arXiv:2106.09685), Fig. 1; other tiles: course figures -->

- HW4 (fine-tuning & adapting LLMs) is due today

---

# Billions of weights, one laptop

<!-- img: visuals/assets/photo-circuit-chip.jpg -->
<!-- caption: A 7B model is billions of numbers. The whole game today is making those numbers fit and move on modest hardware. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1518770660439-4636190af475 -->

---

# Why full fine-tuning doesn't scale

<!-- img: visuals/fullft-cost.png -->
<!-- source: original figure; Adam memory rule-of-thumb -->

---

# Parameter-Efficient Fine-Tuning

<!-- img: visuals/peft-family.png -->
<!-- source: original figure -->

---

# LoRA, visually

<!-- img: visuals/assets/lora-2021-fig-1.png -->
<!-- caption: B starts at zero, so training begins exactly at the pretrained model; after training, B times A merges back into W, so inference costs nothing extra. -->
<!-- source: Hu et al. 2021 (arXiv:2106.09685), Fig. 1 -->

- B starts at 0; after training, B and A merge back into W: no extra inference latency

---

# LoRA: the core idea

<!-- img: visuals/lora-math.png -->
<!-- source: after Hu et al., 2021, arxiv.org/abs/2106.09685 -->

---

# Why LoRA is a big deal

<!-- img: visuals/assets/lora-2021-fig-2.png -->
<!-- caption: The lone blue dot is full fine-tuning of GPT-3 175B. LoRA (pink) holds that accuracy with about 10,000x fewer trainable parameters, while prefix methods degrade as they grow. -->
<!-- source: Hu et al. 2021 (arXiv:2106.09685), Fig. 2 -->

- Blue dot = full fine-tuning of GPT-3 175B; LoRA (pink) matches it with ~10,000x fewer params

---

# Quantization: fewer bits per weight

<!-- img: visuals/quant-bits.png -->
<!-- source: original figure -->

---

# How quantization works

<!-- img: visuals/quant-intuition.png -->
<!-- source: original figure -->

---

# QLoRA: the combination

<!-- img: visuals/qlora.png -->
<!-- source: after Dettmers et al., 2023, arxiv.org/abs/2305.14314 -->

---

# Could you fine-tune a model on your laptop tonight?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-laptop-code.jpg -->
<!-- caption: With QLoRA, the honest answer is yes. Hold that thought for the exercise. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1498050108023-c5249f4df085 -->

---

# Running models locally with Ollama

<!-- img: visuals/ollama-local.png -->
<!-- source: original figure; model size from ollama.com/library/qwen2.5 -->

---

# The efficiency toolbox

<!-- img: visuals/toolbox.png -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the quantization bake-off. -->
<!-- source: original figure -->

---

# Activity: quantization bake-off

<!-- img: visuals/activity-bakeoff.png -->
<!-- source: original figure -->

---

# Exercise: build a LoRA layer

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-09/class-01/exercise/lora_lab.py -->

- Freeze the base weight; train only A and B
- Then run a quantization bake-off at 8/4/2 bits

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run in the course Docker image (seed=0) -->

- Loss collapses; only 48 trainable params; error grows as bits drop

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->

---

# Project work and feedback

<!-- img: visuals/project-session.png -->
<!-- caption: Last 15 minutes: work with your team while I come around. Checkpoint presentations are next week. -->
<!-- source: project/README.md milestones -->
