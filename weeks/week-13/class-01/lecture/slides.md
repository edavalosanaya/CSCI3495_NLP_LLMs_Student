---
marp: true
title: "W13C1: Memory, Planning & Reflection"
paginate: true
---

# Memory, Planning & Reflection
## Agent architectures, CSCI 3495, Week 13 Class 1

Lecture (~28 min), break (5), design-then-build activity (~37 min)

<!-- layout: title -->
<!-- img: visuals/assets/photo-elephant.jpg -->
<!-- caption: An elephant never forgets. Your W12 agent forgets everything the instant its loop ends; today we fix that. -->
<!-- source: Giles Laurent, Wikimedia Commons, CC BY-SA 4.0 -->

---

# Today

<!-- img: visuals/today-roadmap.png -->
<!-- caption: From a bare ReAct loop to a memory + planning + reflection agent, then we build it. -->
<!-- source: deck figures; Reflexion (Shinn et al., 2023), arXiv:2303.11366, Fig. 2a; Unsplash License photo -->

---

# Recap: the ReAct agent (W12)

<!-- img: visuals/react-recap.png -->
<!-- caption: Same loop you built in W12. Today we wrap it with three upgrades. -->
<!-- source: original figure; ReAct (Yao et al., 2022) -->

---

# Smarter scaffolding

<!-- layout: section -->

---

# What would you write down to do better next time?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-journal.jpg -->
<!-- caption: That note is exactly what an agent stores as a reflection. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1517842645767-c639042777db -->

---

# The agent architecture

<!-- img: visuals/agent-architecture.png -->
<!-- caption: One controller (the ReAct loop) wrapped with planning, memory, and reflection. -->
<!-- source: original figure -->

---

# Memory: two kinds

<!-- img: visuals/memory-two-kinds.png -->
<!-- caption: The same demo task, seen by each kind of memory. -->
<!-- source: original figure; weeks/week-13/class-01/solutions/run_demo.py -->

---

# Why memory matters

<!-- img: visuals/why-memory.png -->
<!-- caption: Long-term memory holds reflections, short notes that feed into the next prompt. -->
<!-- source: original figure -->

---

# Planning: decompose before acting

<!-- img: visuals/planning.png -->
<!-- caption: A plan made before the loop turns one hard problem into a few easy steps. -->
<!-- source: original figure -->

---

# Reflection: the Reflexion idea

<!-- img: visuals/reflexion-idea.png -->
<!-- source: original figure; Reflexion (Shinn et al., 2023), arXiv:2303.11366 -->

---

# Reflexion, as the paper draws it

<!-- img: visuals/assets/shinn-2023-fig-2a.png -->
<!-- caption: Map it to today's lab: Actor = your react_attempt, Evaluator = the success() check, and the reflective text is what memory.add() stores. -->
<!-- source: Reflexion (Shinn et al., 2023), arXiv:2303.11366, Fig. 2a -->

---

# Reflexion: why it works

<!-- img: visuals/reflexion-why.png -->
<!-- source: Reflexion (Shinn et al., 2023), arXiv:2303.11366 -->

---

# The Reflexion loop

<!-- img: visuals/reflexion-loop.png -->
<!-- caption: This flowchart is exactly the function you implement in today's exercise. -->
<!-- source: original figure; Reflexion (Shinn et al., 2023) -->

---

# A concrete trace

<!-- img: visuals/concrete-trace.png -->
<!-- caption: The exact trace you will reproduce in today's exercise. -->
<!-- source: weeks/week-13/class-01/solutions/run_demo.py -->

---

# Related: Generative Agents

<!-- img: visuals/assets/park-2023-fig-1.png -->
<!-- caption: Optional reading, not quizzed: 25 characters in Smallville run on the same recipe you build today, a memory stream, reflection, and a daily plan each. -->
<!-- source: Generative Agents (Park et al., 2023), arXiv:2304.03442, Fig. 1 -->

---

# Design choices & trade-offs

<!-- img: visuals/tradeoffs.png -->
<!-- caption: Four dials you set before shipping an agent; the exercise picks defaults for each. -->
<!-- source: original figure -->

---

# Assigned reading this week

<!-- img: visuals/reading.png -->
<!-- caption: Required: Reflexion (Shinn 2023). Optional, not quizzed: Generative Agents (Park 2023). -->
<!-- source: weeks/week-13/class-01/readings.md -->

---

# Does the memory actually pay off?

<!-- img: visuals/memory-payoff.png -->
<!-- caption: Measured on this lab's own ten problems: carrying reflections across problems takes first-attempt success from 0/10 to 5/10. -->
<!-- source: weeks/week-13/class-01/solutions/run_suite.py, qwen2.5:1.5b -->

---

# Activity: design it, then build it

<!-- img: visuals/activity.png -->
<!-- caption: One target task for everyone, a menu of tools, and an external evaluator that tells the agent what it got wrong. -->
<!-- source: weeks/week-13/class-01/exercise/README.md -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then Phase 1: design your agent on paper. -->
<!-- source: original figure -->

---

# First, design on paper

<!-- layout: statement -->
<!-- img: visuals/assets/photo-whiteboard-design.jpg -->
<!-- caption: In teams of 3, diagram the agent loop before you write a line of code. What memory? When to plan? When to reflect? -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1531403009284-440f080d1e12 -->

---

# Then build the loop you designed

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-13/class-01/exercise/agent.py -->

- Compare the code you write to the design you drew

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: If your run matches this, your Reflexion loop works. -->
<!-- source: weeks/week-13/class-01/solutions/run_demo.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->
