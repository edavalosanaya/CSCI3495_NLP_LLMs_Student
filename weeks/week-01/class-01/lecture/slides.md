---
marp: true
title: "W1C1: Welcome to NLP & LLMs"
paginate: true
---

# CSCI 3495
## Natural Language Processing & Large Language Models
### Week 1, Class 1: Welcome & The Big Picture

Lecture (~30 min), break (5), setup (~25 min), pair scavenger hunt (~10 min)

<!-- layout: title -->
<!-- img: visuals/assets/photo-library-reading.jpg -->
<!-- caption: Shelves of language, waiting to be read by something that cannot read. That is the whole problem. -->
<!-- source: photo: Unsplash License -->

---

# Ice breaker: Find Your Training Data

<!-- img: visuals/icebreaker-bingo.png -->
<!-- caption: Nine squares, nine different classmates. Sheets are on your desk; you have eight minutes. -->
<!-- source: handout at weeks/week-01/class-01/icebreaker-bingo.md -->

---

# Who am I?

<!-- img: visuals/who-am-i.png -->
<!-- caption: Two minutes of me, so the next fifteen weeks are not a stranger talking at you. -->
<!-- source: personal photos; game and series cover art copyright their publishers -->

---

# My background

<!-- img: visuals/my-background.png -->
<!-- caption: St. Mary's in San Antonio, then Vanderbilt in Nashville. -->
<!-- source: campus photos and marks copyright the respective universities -->

---

# Installing Docker

<!-- img: visuals/docker-install.png -->
<!-- thumb: Docker in 100 Seconds | https://www.youtube.com/watch?v=Gjnup-PuquQ | visuals/assets/yt-docker-100s.jpg -->
<!-- link: Windows | https://docs.docker.com/desktop/setup/install/windows-install/ -->
<!-- link: macOS | https://docs.docker.com/desktop/setup/install/mac-install/ -->
<!-- link: Linux | https://docs.docker.com/desktop/setup/install/linux/ -->
<!-- caption: Do this before next class. Everything we run this semester runs inside this image. -->
<!-- source: docs.docker.com/desktop/setup/install/ -->

---

# Schedule & syllabus

<!-- layout: section -->

Open both now. We will walk the semester end to end.

---

# Today's agenda

<!-- img: visuals/today-roadmap.png -->
<!-- caption: Six stops. We end with a language model running on your own laptop. -->
<!-- source: original figure -->

---

# What is Natural Language Processing (NLP)?

<!-- img: visuals/assets/thinking.gif -->
<!-- source: reaction GIF -->

- Take thirty seconds. Say it in your own words before I say it in mine.

---

# Language in, action out

<!-- img: visuals/what-is-nlp.png -->
<!-- thumb: ML in 100 Seconds | https://www.youtube.com/watch?v=PeMlggyqz0Y | visuals/assets/yt-ml-100s.jpg -->
<!-- caption: NLP turns human language into useful action: search, translation, voice, and chat. -->
<!-- source: original figure -->

---

# Language is everywhere

<!-- img: visuals/reflect-language.png -->
<!-- caption: Every conversation, message, and search is text waiting to be understood by a machine. -->
<!-- source: original figure; photo: Unsplash License, https://images.unsplash.com/photo-1517245386807-bb43f82c33c4 -->

---

# If a computer can finish your sentence, does it understand you?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-ai-abstract.jpg -->
<!-- caption: Hold that question. We will keep asking it all semester. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1620712943543-bcc4688e7485 -->

---

# Predicting the next word

<!-- img: visuals/autocomplete.gif -->
<!-- caption: A local LLM completes a prompt one token at a time. That is the whole trick, scaled up. -->
<!-- source: original animation (qwen2.5:0.5b style) -->

---

# Why is language hard?

<!-- img: visuals/why-hard.png -->
<!-- caption: Four reasons language resists computation: ambiguity, context, world knowledge, productivity. -->
<!-- source: original figure; Winograd schema example, Levesque et al. 2012 -->

---

# A very short history

<!-- img: visuals/history-timeline.png -->
<!-- caption: Seven decades: rules to statistics to neural nets to Transformers to pretraining to LLMs and agents. -->
<!-- source: original figure -->

---

# This course in one diagram

<!-- img: visuals/course-arc.png -->
<!-- caption: Seven phases over 15 weeks, from text foundations to agents, workflows, and ethics. -->
<!-- source: original figure; see schedule/SCHEDULE.md -->

---

# How class works (every session)

<!-- img: visuals/class-rhythm.png -->
<!-- caption: Every session is ~35 min lecture, a 5 min break, then ~35 min of hands-on coding. -->
<!-- source: original figure -->

---

# Learning together

<!-- img: visuals/learning-together.png -->
<!-- caption: This is a hands-on, build-it course. Bring your laptop and your questions. -->
<!-- source: original figure; photo: Unsplash License, https://images.unsplash.com/photo-1523240795612-9a054b0db644 -->

---

# Grade breakdown

<!-- img: visuals/assessment-donut.png -->
<!-- caption: Project 35, Homeworks 27, Quizzes 15, Participation 8, Final 8, Midterm 7. Details in the syllabus. -->
<!-- source: syllabus/SYLLABUS.md -->

---

# The semester project

<!-- img: visuals/project-paths.png -->
<!-- caption: Pick a path: new method, reproduction, or benchmark. Solo or in a team of up to three. -->
<!-- source: project/README.md -->

---

# Three milestones

<!-- img: visuals/project-milestones.png -->
<!-- caption: Weeks 5, 10 and 15. The full rubrics live in project/RUBRICS.md. -->
<!-- link: NeurIPS LaTeX template (Overleaf) | https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc -->
<!-- source: project/README.md; project/RUBRICS.md -->

---

# Tools: free & open-source

<!-- img: visuals/tools-logos.png -->
<!-- caption: Everything we use is free, open-source, and runs on a laptop CPU. No API keys, no GPU. -->
<!-- source: logos copyright their projects (Simple Icons, CC0) -->

---

# AI-use policy (important)

<!-- img: visuals/ai-policy.png -->
<!-- caption: Use AI as an assistant and an object of study, but disclose it, own every line, and keep exams on paper. -->
<!-- source: syllabus/SYLLABUS.md -->

---

# Assigned reading this week

<!-- img: visuals/reading-slp3.png -->
<!-- caption: Jurafsky & Martin, SLP3 (free draft): Ch. 1 and 2. Quiz 1 next class covers this plus today. -->
<!-- source: web.stanford.edu/~jurafsky/slp3/ -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then we get Docker running and put a real language model on your laptop. -->
<!-- source: original figure -->

---

# Exercise: your first local LLM prompt

<!-- img: visuals/hello-code.png -->
<!-- source: weeks/week-01/class-01/exercise/hello_nlp.py -->

- Edit the prompt, then try temperature 0.0 vs 1.2
- Run it and read the model's answer

---

# What you should see

<!-- img: visuals/hello-output.png -->
<!-- source: example run, qwen2.5:0.5b via Ollama -->

- A real LLM, running locally on your laptop CPU

---

# Activity: First-Prompt Scavenger Hunt

<!-- img: visuals/activity-scavenger.png -->
<!-- caption: Pairs, programming, 10 minutes. Find one win and one fail, then post to the shared board. -->
<!-- source: weeks/week-01/class-01/exercise/hello_nlp.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Language is hard; the field went from rules to LLMs to agents; we build it hands-on. -->
<!-- source: original figure -->
