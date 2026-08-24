---
marp: true
title: "W12C2: Designing Tools & the Agent Loop"
paginate: true
---

<!-- layout: title -->

# Designing Tools & the Agent Loop

Week 12, Class 2: brief intro and failure modes, Quiz 12 right before the break, then the extended agent build

<!-- img: visuals/assets/photo-crash-test.jpg -->
<!-- caption: Engineers crash cars on purpose to learn what breaks before drivers do. Today we crash-test an agent: five failure modes, five guards. -->
<!-- source: Photo: Brady Holt, CC BY 3.0, via Wikimedia Commons -->

---

# Today is an extended hands-on lab

<!-- img: visuals/lab-agent-milestones.png -->
<!-- caption: A brief intro first, then Quiz 12 right before the break, then a full ~50 min build of the robust ReAct loop and its guards, with milestones. -->
<!-- source: weeks/week-12/class-02/exercise; tile images from this deck's figures; photo: Unsplash License -->

---

# Last class

<!-- img: visuals/last-class.png -->
<!-- source: original figure -->

---

# The loop, as real code

<!-- img: visuals/loop-code.png -->
<!-- source: weeks/week-12/class-02/solutions/agent.py -->

---

# Designing a good tool

<!-- img: visuals/good-tool.png -->
<!-- source: original figure -->

---

# A clever model does not make a robust agent

<!-- layout: statement -->
<!-- img: visuals/assets/photo-debug.jpg -->
<!-- caption: Most agent failures live in the loop and the tool contracts, not the model. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1581091226825-a6a2a5aee158 -->

---

# Five failure modes (and defenses)

<!-- layout: section -->

---

# Failure modes at a glance

<!-- img: visuals/failure-modes.png -->
<!-- caption: Most agent bugs are loop-control and tool-contract problems, not the model being dumb. -->
<!-- source: original figure -->

---

# 1. Runaway / infinite loop

<!-- img: visuals/fm-runaway.png -->
<!-- source: original figure -->

---

# 2. Malformed action

<!-- img: visuals/fm-malformed.png -->
<!-- source: original figure -->

---

# 3. Tool error / bad input

<!-- img: visuals/fm-toolerror.png -->
<!-- source: weeks/week-12/class-02/solutions/agent.py -->

---

# 4. Unsafe action

<!-- img: visuals/fm-unsafe.png -->
<!-- source: original figure -->

---

# Defending against tool-call abuse

<!-- img: visuals/tool-abuse-defense.png -->
<!-- caption: Privilege separation plus confirm-before-act: a tricked model still cannot run a risky tool unapproved. -->
<!-- source: OWASP Top 10 for LLM Apps -->

---

# 5. Hallucinated observation

<!-- img: visuals/fm-hallucinated.png -->
<!-- source: original figure -->

---

# Putting it together

<!-- layout: section -->

---

# A robust loop: all five guards

<!-- img: visuals/robust-table.png -->
<!-- source: weeks/week-12/class-02/exercise -->

---

# You cannot fix what you cannot see

<!-- img: visuals/assets/photo-circuit.jpg -->
<!-- caption: Like wiring on a board, an agent is only debuggable when every step is traced. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1518770660439-4636190af475 -->

---

# Observability: log the trace

<!-- img: visuals/observability.png -->
<!-- source: weeks/week-12/class-02/solutions/agent.py -->

---

# How many tools? How big a budget?

<!-- img: visuals/tradeoffs.png -->
<!-- source: original figure -->

---

# Assigned reading (recap for the quiz)

<!-- img: visuals/reading.png -->
<!-- source: resources/landmark-papers.md -->

---

# Quiz 12 (paper)

<!-- img: visuals/quiz.png -->
<!-- caption: Coverage: Week 12 lecture + ReAct (Yao et al., 2022). ~10 min, closed book. Clear your desk; finish early and your break starts early, need more time and it comes out of the break. -->
<!-- source: quizzes/quiz-12.md -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the extended lab: build the robust agent. -->
<!-- source: original figure -->

---

# Extended lab: build the robust agent

<!-- img: visuals/exercise-tools.png -->
<!-- source: weeks/week-12/class-02/exercise/tools.py -->

- About 50 min across milestones: tools first, then parse_action, run_tool, and the guarded run_agent

---

# What success looks like

<!-- img: visuals/exercise-output.png -->
<!-- source: example run: pytest test_agent.py (no Ollama needed) -->

---

# Recap

<!-- img: visuals/recap-end.png -->
<!-- source: original figure -->
