---
marp: true
title: "W10C2, Chain-of-Thought & Advanced Prompting"
paginate: true
---

# Chain-of-Thought, Self-Consistency & Decomposition
## Week 10, Class 2: CoT lecture, Quiz 10 right before the break, then the verify lab

<!-- layout: title -->
<!-- img: visuals/assets/photo-dominoes.jpg -->
<!-- caption: One nudge, then every step sets up the next. What happens when one domino is out of line? -->
<!-- source: Kurt:S, Wikimedia Commons, CC BY 2.0 -->

---

# Where we left off

<!-- img: visuals/recap-prev.png -->
<!-- source: original figure -->

---

# One-step answers fail multi-step problems

<!-- img: visuals/one-step-fail.png -->
<!-- source: original figure -->

---

# Could you solve this in one glance, or do you need to show your work?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-chalkboard-math.jpg -->
<!-- caption: Hard problems need steps. So do language models. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1509228468518-180dd4864904 -->

---

# Chain-of-thought

<!-- layout: section -->

---

# CoT: put the steps in the exemplar

<!-- img: visuals/assets/wei-2022-fig-1.png -->
<!-- caption: The most reproduced figure of the prompting era. Only the highlighted worked solution differs between the two prompts, yet it flips the cafeteria answer from 27 to 9. -->
<!-- source: Wei et al. 2022 (arXiv:2201.11903), Fig. 1 -->

---

# Why does CoT help?

<!-- img: visuals/why-cot.png -->
<!-- source: original figure; Wei et al. 2022 (arXiv:2201.11903) -->

---

# CoT and scale

<!-- layout: figure -->
<!-- img: visuals/assets/wei-2022-fig-4.png -->
<!-- crop: 0,0,0,0.200 -->
<!-- caption: Figure 4: Chain-of-thought prompting enables large language models to solve challenging math problems. Notably, chain-of-thought reasoning is an emergent ability of increasing model scale. Prior best numbers are from Cobbe et al. (2021) for GSM8K, Jie et al. (2022) for SVAMP, and Lan et al. (2021) for MAWPS. -->
<!-- source: Wei et al. 2022 (arXiv:2201.11903), Fig. 4 -->

- Blue pulls away from black only at each family's largest model; small models gain nothing or get worse

---

# Zero-shot CoT

<!-- img: visuals/zero-shot-cot.png -->
<!-- source: original figure; Kojima et al. 2022 (arXiv:2205.11916) -->

---

# Sample many chains, then vote

<!-- img: visuals/assets/wang-2022-fig-1.png -->
<!-- crop: 0,0.177,0,0 -->
<!-- caption: Self-consistency: greedy decoding commits to one flawed path ($14); sampling at temperature > 0 explores several, and the stray $26 path is outvoted by the two that agree on $18. -->
<!-- source: Wang et al. 2022 (arXiv:2203.11171), Fig. 1 -->

---

# Decomposition: break the task apart

<!-- img: visuals/decomposition.png -->
<!-- source: original figure; least-to-most, Zhou et al. 2022 (arXiv:2205.10625) -->

---

<!-- layout: section -->
# From CoT to Thinking Models

---

# What if we stopped prompting for reasoning and trained it in?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-chalkboard-math.jpg -->
<!-- caption: The 2024 to 2025 idea: make the chain a learned behavior, not a prompt you write. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1509228468518-180dd4864904 -->

---

# Scale inference, not just parameters

<!-- img: visuals/test-time-compute.png -->
<!-- source: original schematic; test-time compute scaling, Snell et al. 2024 (arXiv:2408.03314) -->

---

# CoT to trained reasoning

<!-- img: visuals/cot-to-reasoning.png -->
<!-- source: original figure; Wei et al. 2022 (2201.11903); DeepSeek-R1 2025 (2501.12948) -->

---

# Reasoning models: what RL training buys

<!-- img: visuals/assets/deepseek-r1-2025-fig-1.png -->
<!-- caption: RL on verifiable rewards, the Week 8 alignment idea aimed at reasoning: AIME accuracy climbs past the average human competitor while responses grow 10x longer. Thinking longer is learned, never prompted. -->
<!-- source: DeepSeek-AI 2025 (arXiv:2501.12948), Fig. 1 -->

- Same skepticism applies: plausible chain does not guarantee a correct answer

---

# Extract the answer reliably

<!-- img: visuals/parse-answer.png -->
<!-- source: original figure -->

---

# When NOT to use CoT

<!-- img: visuals/when-not.png -->
<!-- source: original figure -->

---

# Risks: plausible ≠ correct

<!-- img: visuals/plausible.png -->
<!-- source: original figure -->

---

# Looking ahead

<!-- img: visuals/looking-ahead.png -->
<!-- source: schedule/SCHEDULE.md; thumbnails: Wei et al. 2022 (2201.11903) Fig. 1, Lewis et al. 2020 (2005.11401) Fig. 1, Yao et al. 2022 (2210.03629) Fig. 1 -->

---

# Quiz 10 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Covers Week 10 lecture + Chain-of-Thought (Wei et al. 2022); optional readings are never quizzed. Clear your desk: finish early and your break starts early, need more time and it comes out of the break. -->
<!-- source: quizzes/quiz-10.md -->

- Clear your desk: finish early and your break starts early; need more time and it comes out of your break

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the Predict, then Verify CoT activity. -->
<!-- source: original figure -->

---

# Whiteboards out: predict before you run

<!-- layout: statement -->
<!-- img: visuals/assets/photo-whiteboard-sketch.jpg -->
<!-- caption: Pairs, ~25 min. Hand-write a CoT trace and a decomposition tree, predict if CoT helps, then verify. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1454165804606-c3d57bc86b40 -->

---

# Activity: Predict, then Verify CoT

<!-- img: visuals/activity-tps.png -->
<!-- source: weeks/week-10/class-02/exercise/cot_lab.py -->

---

# Verify your predictions

<!-- img: visuals/exercise-output.png -->
<!-- source: example run, qwen2.5:0.5b via Ollama -->

- Did the scores match what you predicted on the whiteboard?

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->

---

# Midterm next class

<!-- img: visuals/midterm-heads-up.png -->
<!-- caption: One handwritten 8.5 x 11 sheet, both sides, name on it, handed in with the exam. Study guide has a full-length practice exam. -->
<!-- source: exams/midterm-study-guide.md -->
