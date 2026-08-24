---
marp: true
title: "W14C1: Agentic Workflows & Orchestration"
paginate: true
---

# Week 14, Class 1
## Agentic Workflows & Orchestration

Lecture (~25 min), jigsaw + seated lightning share, then routing in pairs

<!-- layout: title -->
<!-- img: visuals/assets/photo-conductor-orchestra.jpg -->
<!-- caption: A conductor plays no instrument, yet the whole orchestra depends on one. What plays that role in a system of LLM calls? -->
<!-- source: Photo: AnnaLesniewski, Wikimedia Commons, CC0 -->

---

# Today

<!-- img: visuals/today.png -->
<!-- source: composite of this deck's figures; activity photo: Unsplash License -->

---

# From one call to a system

<!-- img: visuals/single-to-systems.png -->
<!-- source: original figure -->

- The skill shifts from prompting to systems design

---

# A team of specialists, not one genius

<!-- img: visuals/assets/photo-team-collab.jpg -->
<!-- caption: Like a project team, a good LLM system splits work into focused roles that each do one job well. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1522071820081-009f0129c71c -->

---

# The patterns

<!-- layout: section -->

---

# A grounding source

<!-- img: visuals/simplest-first.png -->
<!-- source: Anthropic, "Building Effective Agents" (2024) -->

- An engineering guide, not a paper
- Most agent projects fail by being over-engineered

---

# Workflow or agent?

<!-- img: visuals/workflow-vs-agent.png -->
<!-- source: Anthropic, "Building Effective Agents" (2024) -->

- Most production needs are workflows, not full agents

---

# The building block: the augmented LLM

<!-- img: visuals/augmented-llm.png -->
<!-- source: Anthropic, "Building Effective Agents" (2024) -->

---

# The five patterns at a glance

<!-- img: visuals/workflow-patterns.png -->
<!-- source: Anthropic, "Building Effective Agents" (2024) -->

- Today: patterns 1 to 4; evaluator-optimizer next class

---

# Pattern 1: prompt chaining

<!-- img: visuals/prompt-chaining.png -->
<!-- source: original figure; Anthropic, "Building Effective Agents" (2024) -->

---

# Pattern 2: routing

<!-- img: visuals/routing.png -->
<!-- source: original figure; Anthropic, "Building Effective Agents" (2024) -->

---

# Pattern 3: parallelization

<!-- img: visuals/parallelization.png -->
<!-- source: original figure; Anthropic, "Building Effective Agents" (2024) -->

---

# Pattern 4: orchestrator and workers

<!-- img: visuals/orchestrator-workers.png -->
<!-- source: original figure; Anthropic, "Building Effective Agents" (2024) -->

---

# What is the simplest design that could possibly work?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-circuit.jpg -->
<!-- caption: Ask this before adding a single agent loop. Complexity is a cost, not a feature. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1518770660439-4636190af475 -->

---

# Routing in depth (our exercise)

<!-- img: visuals/router-anatomy.png -->
<!-- source: original figure -->

- Make the router deterministic and testable

---

# Why structure beats one mega-prompt

<!-- img: visuals/structure-benefits.png -->
<!-- source: original figure -->

- Software engineering with stochastic parts

---

# Multi-agent systems: a caution

<!-- img: visuals/multi-agent-caution.png -->
<!-- source: original figure; Anthropic, "Building Effective Agents" (2024) -->

- Often a workflow is simpler and more reliable

---

# Designing good tool and step interfaces

<!-- img: visuals/tool-interfaces.png -->
<!-- source: Anthropic, "Building Effective Agents" (2024) -->

- The model is only as good as its interface
- Prompt-engineer your tools as carefully as your prompts

---

# Recap

<!-- img: visuals/recap.png -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the pattern jigsaw and lightning share. -->
<!-- source: original figure -->

---

# Activity

<!-- layout: section -->

---

# Activity: jigsaw + poster lightning share

<!-- img: visuals/activity.png -->
<!-- source: weeks/week-14/class-01/exercise/README.md -->

- Lightning share is the flex step if we run tight on time

---

# Poster one pattern, then teach it back

<!-- layout: statement -->
<!-- img: visuals/assets/photo-gallery-walk.jpg -->
<!-- caption: Each team owns one pattern: a diagram, a real use-case, and a "when NOT to use an agent" caveat. Then teach it back in 60 seconds from your seats. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1559223607-a43c990c692c -->

---

# Then build it: router and workers

<!-- img: visuals/exercise-code.png -->
<!-- source: weeks/week-14/class-01/exercise/workflow.py -->

- In pairs, implement the routing pattern

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- source: example run, mock LLM (deterministic tests) -->

---

# Reading & next class

<!-- img: visuals/reading.png -->
<!-- source: anthropic.com/research/building-effective-agents -->

- Next: reflection, evaluator and optimizer, when NOT to use agents
- Quiz 14 plus project feedback session
