---
marp: true
title: "W12C1: What is an LLM Agent? ReAct & Tool Use"
paginate: true
---

# CSCI 3495
## LLM Agents: ReAct, Tool Use & Toolformer
### Week 12, Class 1

<!-- layout: title -->
<!-- img: visuals/assets/photo-swiss-army-knife.jpg -->
<!-- caption: One body, many tools, and a decision about which one to reach for. That decision is the whole subject of today. -->
<!-- source: Photo: Andrew Toskin, CC BY-SA 2.0, via Wikimedia Commons -->

---

# Today

<!-- img: visuals/today-agenda.png -->
<!-- caption: Phase 5 begins, creating agents. HW6 (build an agent) goes out today. -->
<!-- source: original figure; tile images: ReAct (Yao et al., 2022) Fig. 1, Toolformer (Schick et al., 2023) Fig. 1, Unsplash photos -->

---

# Recap: where we are

<!-- img: visuals/recap-where.png -->
<!-- source: original figure (schedule/SCHEDULE.md); thumbnails: CoT (Wei et al., 2022) Fig. 1, RAG (Lewis et al., 2020) Fig. 1, ReAct (Yao et al., 2022) Fig. 1 -->

---

# What is an LLM agent?

<!-- img: visuals/agent-def.png -->
<!-- source: original figure -->

---

# If a model can call tools, who is really in control?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-humanoid.jpg -->
<!-- caption: Keep this question in mind: the model proposes, the loop decides, the tools act. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1531746790731-6c087fecd65a -->

---

# Why add tools at all?

<!-- img: visuals/why-tools.png -->
<!-- source: original figure -->

---

# Agents & ReAct

<!-- layout: section -->

---

# Reason only, act only, or both?

<!-- img: visuals/reason-act-react.png -->
<!-- caption: Chain-of-thought is the left loop, plain tool calling is the right loop. ReAct is what happens when you join them. -->
<!-- source: original figure, after Yao et al. 2022 (arXiv:2210.03629) -->

---

# The ReAct loop (Yao et al., 2022)

<!-- img: visuals/react-loop.png -->
<!-- caption: Reasoning + Acting interleaved: Thought → Action → Observation, repeated. -->
<!-- source: original figure; ReAct (Yao et al., 2022), arXiv:2210.03629 -->

---

# A ReAct trace, line by line

<!-- img: visuals/react-loop-anim.gif -->
<!-- caption: The arrow that matters points backwards: after every observation it returns to Thought, and only finish exits. -->
<!-- source: original animation; original figure (ReAct-style trace) -->

---

# Why interleave reasoning and acting?

<!-- img: visuals/assets/react-2022-fig1-hotpotqa.png -->
<!-- caption: CoT alone hallucinates (red); Act-only retrieves but cannot plan past a dead end; ReAct chains searches to the right answer. You will see this figure again in the Quiz 12 paper. -->
<!-- source: ReAct (Yao et al., 2022), arXiv:2210.03629, Fig. 1 (HotpotQA half) -->

---

# Anatomy of a ReAct step

<!-- img: visuals/anatomy.png -->
<!-- source: original figure -->

---

# Building the loop

<!-- layout: section -->

---

# How the loop actually runs

<!-- img: visuals/loop-steps.png -->
<!-- source: original figure -->

---

# Designing an action grammar

<!-- img: visuals/action-grammar.png -->
<!-- source: original figure -->

---

# Tools = typed functions with a contract

<!-- img: visuals/tool-contract.png -->
<!-- source: original figure -->

---

# Safety: never eval() model output

<!-- img: visuals/safety-eval.png -->
<!-- source: original figure -->

---

# Before agents act: talking to software safely

<!-- layout: section -->

---

# The problem: prose is not an API

<!-- img: visuals/prose-not-api.png -->
<!-- caption: A chatty sentence is brittle to parse; the agent loop needs typed, structured fields. -->
<!-- source: original figure -->

---

# Ask for structured output

<!-- img: visuals/ask-json.png -->
<!-- caption: JSON out of prose, validated before use. The fence around untrusted input is also your first injection defense. -->
<!-- source: original figure -->

---

# Function (tool) calling

<!-- img: visuals/function-calling.png -->
<!-- caption: You declare a signature, the model proposes name plus JSON arguments, the loop executes. This is the Action step of ReAct, standardized. -->
<!-- source: original figure -->

---

# Prompt injection

<!-- img: visuals/injection-types.png -->
<!-- caption: Untrusted text smuggles commands into the prompt: direct from the user, or indirect via pages and files the agent reads. -->
<!-- source: Greshake et al. 2023, arXiv:2302.12173 -->

---

# Tools and agents widen the attack surface

<!-- img: visuals/assets/greshake-2023-fig-3.png -->
<!-- caption: The attacker never talks to the model: a planted instruction (1) rides retrieval (3) into your agent, then tool access (4) turns bad text into exfiltration (5) and user manipulation (6). -->
<!-- source: Greshake et al. 2023 (arXiv:2302.12173), Fig. 3 -->

---

# Defense in depth

<!-- img: visuals/defenses.png -->
<!-- caption: Input and output validation, tool allow-lists, privilege separation, human-in-the-loop, and treating fetched text as untrusted. You will attack these guards in the game today. -->
<!-- source: original figure; OWASP Top 10 for LLM Apps -->

---

# Toolformer (Schick et al., 2023)

<!-- img: visuals/toolformer.png -->
<!-- source: original figure; Toolformer (Schick et al., 2023), arXiv:2302.04761 -->

---

# Toolformer in action: inline API calls

<!-- img: visuals/assets/toolformer-2023-fig1.png -->
<!-- caption: One syntax, four tools: the fine-tuned model decides mid-sentence to call QA, Calculator, MT, or WikiSearch and splices the result back into its own text. No loop, no scaffold. -->
<!-- source: Toolformer (Schick et al., 2023), arXiv:2302.04761, Fig. 1 -->

---

# ReAct vs. Toolformer

<!-- img: visuals/react-vs-toolformer.png -->
<!-- source: original figure; Yao et al. 2022 / Schick et al. 2023 -->

---

# Where agents go wrong (preview)

<!-- img: visuals/failure-preview.png -->
<!-- source: original figure (full treatment in Class 2) -->

---

# Agents already act in the world

<!-- img: visuals/assets/photo-robot.jpg -->
<!-- caption: A service robot reads a screen, decides, and acts. The same loop drives software agents. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1485827404703-89b55fcc595e -->

---

# Agents in the wild

<!-- img: visuals/agents-wild.png -->
<!-- thumb: Multi-Agent Hide & Seek | https://www.youtube.com/watch?v=kopoLzvh5jY | visuals/assets/yt-hide-and-seek.jpg -->
<!-- source: original figure -->

---

# Assigned reading this week

<!-- img: visuals/reading.png -->
<!-- source: resources/landmark-papers.md -->

---

# Pair up: rogue LLM vs guard hunter

<!-- layout: statement -->
<!-- img: visuals/assets/photo-strategy-duel.jpg -->
<!-- caption: Adversarial role-play in pairs, ~30 min. One plays the rogue LLM, one hunts the guard that stops each attack. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1505373877841-8d25f7d46678 -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the Break-the-Agent game, head to head. -->
<!-- source: original figure -->

---

# Activity: Break the Agent, head to head

<!-- img: visuals/activity-break-agent.png -->
<!-- source: weeks/week-12/class-01/exercise/break_the_agent.py -->

---

# What the game looks like

<!-- img: visuals/exercise-game.png -->
<!-- source: example run: break_the_agent.py (no Ollama needed) -->

---

# Recap

<!-- img: visuals/recap-end.png -->
<!-- source: original figure -->
