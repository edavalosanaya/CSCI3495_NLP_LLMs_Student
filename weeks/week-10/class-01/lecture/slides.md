---
marp: true
title: "W10C1, In-Context Learning & Prompt Engineering"
paginate: true
---

# Week 10, Class 1
## In-Context Learning & Prompt Engineering Foundations

<!-- layout: title -->
<!-- img: visuals/assets/photo-magic-lamp.jpg -->
<!-- caption: A genie grants exactly what you SAY, not what you mean. Today: learning to word the wish. -->
<!-- source: Amr Mounir, Wikimedia Commons, CC BY-SA 2.0 (cropped) -->

---

# Today

<!-- img: visuals/agenda.png -->
<!-- source: tile figures: Brown et al. 2020 (arXiv:2005.14165) Fig. 2.1; course figures -->

---

# A new way to "program" models

<!-- img: visuals/finetune-vs-prompt.png -->
<!-- source: original figure; concept from Brown et al. 2020 (arXiv:2005.14165) -->

---

# Shots, not gradient updates

<!-- layout: figure -->
<!-- img: visuals/assets/brown-2020-fig-2-1.png -->
<!-- crop: 0.100,0,0.095,0.185 -->
<!-- caption: Figure 2.1: Zero-shot, one-shot and few-shot, contrasted with traditional fine-tuning. The panels to the right show four methods for performing a task with a language model, fine-tuning is the traditional method, whereas zero-, one-, and few-shot, which we study in this work, require the model to perform the task with only forward passes at test time. We typically present the model with a few dozen examples in the few shot setting. Exact phrasings for all task descriptions, examples and prompts can be found in Appendix G. -->
<!-- source: Brown et al. 2020 (arXiv:2005.14165), Fig. 2.1 -->

---

# The headline result (GPT-3, 2020)

<!-- img: visuals/assets/brown-2020-fig-1-3.png -->
<!-- caption: Faint lines are 42 individual benchmarks, bold lines their mean. The gap between few-shot and zero-shot widens with scale: bigger models extract more from the same prompt. -->
<!-- source: Brown et al. 2020 (arXiv:2005.14165), Fig. 1.3 -->

---

# Anatomy of a prompt

<!-- img: visuals/prompt-anatomy.png -->
<!-- source: original figure; Brown et al. 2020 (arXiv:2005.14165) -->

- Instruction, demonstrations, query, and output cue

---

# Why does ICL work at all?

<!-- img: visuals/why-icl.png -->
<!-- source: original figure; Min et al. 2022 (arXiv:2202.12837) -->

---

# If the model learns from the prompt, is the prompt now your program?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-typing-prompt.jpg -->
<!-- caption: This class: treat your prompt like code you write, test, and debug. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1517694712202-14dd9538aa97 -->

---

# What makes prompts work

<!-- img: visuals/prompts-work.png -->
<!-- source: original figure -->

---

# What makes prompts break

<!-- img: visuals/prompts-break.png -->
<!-- source: original figure -->

---

# Prompt sensitivity is real

<!-- img: visuals/sensitivity.png -->
<!-- source: original figure (illustrative numbers) -->

---

# Engineering prompts

<!-- layout: section -->

---

# Engineer prompts like an experiment

<!-- img: visuals/experiment-loop.png -->
<!-- source: original figure -->

---

# Decoding settings affect prompts too

<!-- img: visuals/fixed-decoding.png -->
<!-- source: original figure -->

---

# "Prompt golf"

<!-- img: visuals/prompt-golf.png -->
<!-- source: original figure -->

---

# Prompt patterns you'll reuse

<!-- img: visuals/patterns.png -->
<!-- source: original figure -->

---

# Prompts are just text: injection

<!-- img: visuals/injection.png -->
<!-- source: original figure; prompt-injection concept (Willison 2022) -->

---

# HW5 is out

<!-- img: visuals/hw5.png -->
<!-- source: weeks/homeworks/hw5 -->

---

# Assigned reading this week

<!-- img: visuals/reading.png -->
<!-- caption: One required paper, quizzed in Class 2. Optional papers are enrichment, never quizzed. -->
<!-- source: resources/landmark-papers.md -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the checkpoint peer-review round. -->
<!-- source: original figure -->

---

# Mid-semester checkpoint: peer review

<!-- img: visuals/checkpoint-review.png -->
<!-- caption: 30 minutes: groups of four, one page each, everyone presents once and critiques three times. -->
<!-- source: project/checkpoint/TEMPLATE.md -->

---

# Game on: Prompt Golf Arena

<!-- layout: statement -->
<!-- img: visuals/assets/photo-bullseye.jpg -->
<!-- caption: Hit 100% on the eval set, then say it in fewer words. Kickoff now, finish with HW5. -->
<!-- source: Santeri Viinamäki, Wikimedia Commons, CC BY 4.0 -->

---

# Activity: Prompt Golf Arena

<!-- img: visuals/activity-golf-arena.png -->
<!-- source: weeks/week-10/class-01/exercise/prompt_lab.py -->

---

# Build the harness behind it

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-10/class-01/exercise/prompt_lab.py -->

- Each team builds a few-shot prompt and scores variants on the labeled set

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run, qwen2.5:0.5b via Ollama -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
