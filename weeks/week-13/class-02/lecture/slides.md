---
marp: true
title: "W13C2: Reasoning Agents & Agent Evaluation"
paginate: true
---

# Reasoning Agents & Agent Evaluation
## CSCI 3495, Week 13 Class 2

Lecture first, Quiz 13 right before the break, then the leaderboard showdown

<!-- layout: title -->
<!-- img: visuals/assets/photo-dog-agility.jpg -->
<!-- caption: Agility judges never score the prettiest jump; they count obstacles cleared on a fixed course. How should we score an agent? -->
<!-- source: Ron Armstrong, Wikimedia Commons, CC BY 2.0 -->

---

# Last class

<!-- img: visuals/last-class.png -->
<!-- source: original figure; weeks/week-13/class-01/solutions/run_demo.py -->

---

# How do we know it works?

<!-- layout: section -->

---

# If two agents disagree, which one do you trust?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-scoreboard.jpg -->
<!-- caption: Not the smoother talker. The one that solves more tasks on a fixed suite. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1551288049-bebda4e38f71 -->

---

# Why evaluating agents is hard

<!-- img: visuals/why-hard.png -->
<!-- source: original figure -->

---

# The golden rule

<!-- img: visuals/golden-rule.png -->
<!-- caption: This rule is the closing discussion of today's activity. -->
<!-- source: original figure -->

---

# Anatomy of an agent eval

<!-- img: visuals/agent-eval.png -->
<!-- caption: The pipeline you build in today's exercise, one function per box. -->
<!-- source: original figure -->

---

# Building a task suite

<!-- img: visuals/task-suite.png -->
<!-- caption: Three of the five tasks in the exercise suite; your team adds a sixth. -->
<!-- source: weeks/week-13/class-02/exercise/tasks.py -->

---

# What to measure

<!-- img: visuals/what-to-measure.png -->
<!-- source: original figure -->

---

# pass@1, pass@k, and Reflexion lift

<!-- img: visuals/passk.png -->
<!-- source: original figure; Reflexion (Shinn et al., 2023) -->

---

# The lift, measured

<!-- img: visuals/assets/shinn-2023-fig-4a.png -->
<!-- caption: Trial 0 is pass@1; every retry adds one reflection. The dashed baselines retry without reflecting and stay flat: the lift comes from the lesson, not the extra attempts. -->
<!-- source: Reflexion (Shinn et al., 2023), arXiv:2303.11366, Fig. 4a (HotPotQA) -->

---

# Determinism: reproducible scores

<!-- img: visuals/determinism.png -->
<!-- source: original figure -->

---

# Evaluation pitfalls

<!-- img: visuals/pitfalls.png -->
<!-- caption: Each of these will cost points on the team leaderboard today. -->
<!-- source: original figure -->

---

# Reading the leaderboard

<!-- img: visuals/leaderboard.png -->
<!-- caption: A real run of this lab: four strategies, 20 GSM8K problems, qwen2.5:1.5b. The cheapest one won, and the two that call tools came third and second. -->
<!-- source: weeks/week-13/class-02/solutions/run_bench.py; GSM8K (Cobbe et al. 2021, MIT) -->

---

# From benchmark slices to agents

<!-- img: visuals/benchmarks.png -->
<!-- source: original figure -->

---

# Assigned reading (recap for the quiz)

<!-- img: visuals/reading.png -->
<!-- caption: Reflexion (Shinn 2023) is required; Generative Agents (Park 2023) is optional, never quizzed. -->
<!-- source: weeks/week-13/class-01/readings.md -->

---

# Quiz 13 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Covers Week 13 lecture + Reflexion (Shinn 2023). Clear your desk: finish early and your break starts early; need more time and it comes out of the break. -->
<!-- source: quizzes/quiz-13.md -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the class-wide leaderboard showdown. -->
<!-- source: original figure -->

---

# Activity: class-wide leaderboard showdown

<!-- img: visuals/activity.png -->
<!-- source: weeks/week-13/class-02/exercise/README.md -->

---

# Exercise: build the eval harness

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-13/class-02/exercise/eval_suite.py -->

- The success check runs outside the agent; never pass it in

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Your harness should print exactly this table before you add new tasks. -->
<!-- source: weeks/week-13/class-02/solutions/eval_suite.py -->

---

# Discuss: why grade the outcome, not the prose?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-scoreboard.jpg -->
<!-- caption: Find a fluent-but-wrong trace and a terse-but-right one. A pretty rationale is not a passing grade. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1551288049-bebda4e38f71 -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
