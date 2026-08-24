---
marp: true
title: "W9C2: Evaluation & Benchmarks"
paginate: true
---

# Evaluation & Benchmarks
## Week 9, Class 2: benchmarks & hallucination, then Quiz 9 right before the break

<!-- layout: title -->
<!-- img: visuals/assets/photo-bubble-sheet.jpg -->
<!-- caption: Bubble sheets made grading kids scalable; MMLU grades models the same way. Ask the class: what can a bubble sheet never measure? -->
<!-- source: Onderwijsgek, Wikimedia Commons, CC BY-SA 2.5 NL -->

---

# Today's roadmap

<!-- img: visuals/roadmap.png -->
<!-- source: photos: Unsplash; MMLU items: Hendrycks et al. 2021 (arXiv:2009.03300), Figs. 3-4; other tiles: course figures -->

---

# How do you grade a machine?

<!-- img: visuals/assets/photo-exam-measure.jpg -->
<!-- caption: We grade students with exams. Benchmarks are the exams we write for models, and they are just as easy to get wrong. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1434030216411-0b793f4b4173 -->

---

# Why is evaluating LLMs hard?

<!-- img: visuals/why-hard.png -->
<!-- source: original figure -->

---

# What is a benchmark?

<!-- img: visuals/benchmark-anatomy.png -->
<!-- source: original figure -->

---

# Three landmark benchmarks

<!-- img: visuals/benchmark-compare.png -->
<!-- source: GLUE 1804.07461; MMLU 2009.03300; HELM 2211.09110 -->

---

# GLUE

<!-- img: visuals/assets/glue-2018-table-1.png -->
<!-- source: Wang et al. 2018 (arXiv:1804.07461), Table 1 -->

---

# MMLU

<!-- img: visuals/assets/mmlu-2021-fig-3-4.png -->
<!-- caption: Real items from 3 of the 57 subjects. Four options means random guessing scores 25%; multiple choice makes grading trivial but invites contamination. -->
<!-- source: Hendrycks et al. 2021 (arXiv:2009.03300), Figs. 3-4 -->

- Real items from 3 of 57 subjects; with 4 options, random guessing scores 25%

---

# HELM

<!-- img: visuals/helm.png -->
<!-- source: Liang et al., 2022, arxiv.org/abs/2211.09110 -->

---

# Metrics matter as much as data

<!-- img: visuals/metrics.png -->
<!-- source: original figure -->

- Pick the metric that matches what you actually care about

---

# If a model tops every leaderboard, is it actually good?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-ruler-measure.jpg -->
<!-- caption: Not necessarily. Contamination, narrow metrics, and overfitting can fake a high score. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1453928582365-b6ad33cbcf64 -->

---

# Hallucination

<!-- img: visuals/hallucination.png -->
<!-- source: original figure; example matches the exercise dataset -->

---

# Why models hallucinate

<!-- img: visuals/why-hallucinate.png -->
<!-- source: original figure -->

---

# Two kinds of hallucination

<!-- img: visuals/factuality-types.png -->
<!-- source: original figure; intrinsic vs extrinsic framing, Ji et al. 2023 survey -->

---

# Faithfulness vs factuality

<!-- img: visuals/faithfulness-vs-factuality.png -->
<!-- source: original figure -->

---

# Abstention: saying "I don't know"

<!-- img: visuals/abstention.png -->
<!-- source: original figure -->

---

# Would you bet your job on this answer?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-fact-check.jpg -->
<!-- caption: If the cost of a wrong answer is high, "I don't know" is the right answer. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1454165804606-c3d57bc86b40 -->

---

# How do we measure factuality?

<!-- img: visuals/measure-factuality.png -->
<!-- source: original figure -->

---

# What if the test set was already in the training data?

<!-- img: visuals/contamination.png -->
<!-- caption: Then the score measures memorization, not ability. Benchmarks are public text, so they get scraped into pretraining corpora. -->
<!-- source: original figure -->

---

# Evaluating responsibly

<!-- img: visuals/responsible.png -->
<!-- source: original figure -->

---

# Should a robot grade a robot?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-judge-gavel.jpg -->
<!-- caption: We do it all the time now. The question is whether to trust the verdict. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1589829545856-d10d557cf95f -->

---

# LLM-as-a-judge

<!-- img: visuals/llm-as-judge.png -->
<!-- source: Zheng et al., 2023, arxiv.org/abs/2306.05685 -->

---

# The judge has biases

<!-- img: visuals/judge-biases.png -->
<!-- source: Zheng et al., 2023, arxiv.org/abs/2306.05685 -->

---

# Quiz 9

<!-- img: visuals/quiz.png -->
<!-- caption: Clear your desk. Covers the Week 9 lecture + LoRA (Hu 2021); optional readings are never quizzed. Finish early and your break starts early; need more time and it comes out of the break. -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then design a benchmark, then break it. -->
<!-- source: original figure -->

---

# Activity: design a benchmark, then break it

<!-- img: visuals/activity-benchmark.png -->
<!-- source: original figure -->

---

# Build it, swap it, break it

<!-- layout: statement -->
<!-- img: visuals/assets/photo-redteam.jpg -->
<!-- caption: Design an eval on paper, hand it to another team, and red-team theirs for contamination, position bias, and Goodharting. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1504917595217-d4dc5ebe6122 -->

---

# The punchline: catch a biased judge

<!-- img: visuals/position-bias-demo.png -->
<!-- source: original figure; swap test, Zheng et al. 2023 -->

- The provided swap-test makes every team's red-team concrete

---

# Optional take-home: the eval harness

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-09/class-02/exercise/eval_harness.py -->

- Score items, flag the impossible one, and run the swap test in code
- A short demo today; the full harness is take-home

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->

- A benchmark is a map, not the territory
