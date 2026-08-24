---
marp: true
title: "W15C1: Ethics & Society"
paginate: true
---

# Week 15, Class 1

## Ethics & Society: The Stakes of Language Technology

Lecture (~25 min), team pre-mortems on real deployments, bias-probe demo

<!-- layout: title -->
<!-- img: visuals/assets/photo-grey-parrot.jpg -->
<!-- caption: Meet the original language model. An African grey can repeat thousands of phrases it has heard. What, if anything, does it understand? -->
<!-- source: photo: Acabashi, Wikimedia Commons, CC BY-SA 4.0 -->

---

# Today

<!-- img: visuals/today-roadmap.png -->
<!-- caption: Six stops. We end by running a pre-mortem on a real deployment. -->
<!-- source: photos: Unsplash License; Acabashi, Wikimedia Commons (CC BY-SA 4.0); figure: Bommasani et al. 2021, Fig. 2 -->

---

# Why this, why now

<!-- img: visuals/build-vs-question.png -->
<!-- caption: Every modeling choice is also an ethical choice. -->
<!-- source: original figure -->

---

# Foundations of critique

<!-- layout: section -->

---

# "Stochastic Parrots" (Bender et al., 2021)

<!-- img: visuals/parrots-paper.png -->
<!-- caption: This week's required reading; it supplies today's method and appears on the final exam. -->
<!-- source: Bender et al., FAccT 2021, dl.acm.org/doi/10.1145/3442188.3445922 -->

---

# Form vs. meaning

<!-- img: visuals/form-vs-meaning.png -->
<!-- thumb: Tracing an LLM's thoughts | https://www.youtube.com/watch?v=Bj9BD2D3DzA | visuals/assets/yt-tracing-thoughts.jpg -->
<!-- caption: A cited critique, not a settled fact; what LLMs "understand" is still an open debate. -->
<!-- source: Bender et al., FAccT 2021 -->

---

# The six fault lines

<!-- layout: section -->

---

# Fault line 1: Bias & fairness

<!-- img: visuals/bias-fairness.png -->
<!-- caption: You will measure exactly this pipeline in the take-home bias probe. -->
<!-- source: Bender et al., FAccT 2021 -->

---

# Fault line 2: Misinformation

<!-- img: visuals/misinformation.png -->
<!-- caption: The hallucination problem from Week 9, industrialized. -->
<!-- source: original figure; hallucination (Week 9) -->

---

# Fault line 3: Environmental cost

<!-- img: visuals/environment-cost.png -->
<!-- caption: The bill is real, unequally paid, and mostly spent on runs you never see. -->
<!-- source: Strubell et al. 2019 Table 1; Bender et al. 2021 section 3; Patterson et al. 2021 -->

---

# Fault line 4: Copyright & data provenance

<!-- img: visuals/copyright.png -->
<!-- caption: Courts are testing these questions right now. -->
<!-- source: original figure; datasheets/data statements -->

---

# Fault line 5: Labor

<!-- img: visuals/labor.png -->
<!-- caption: Ask who does the work, and who captures the value. -->
<!-- source: original figure -->

---

# Fault line 6: Safety & policy

<!-- img: visuals/safety.png -->
<!-- caption: No consensus answers yet; this is a live policy debate. -->
<!-- source: original figure -->

---

# One model beneath everything

<!-- img: visuals/assets/bommasani-2021-fig-2.png -->
<!-- caption: The Stanford report's own diagram, from the paper that coined "foundation model". Optional reading, never quizzed. The ethics hook is homogenization: one shared model means its flaws reach every adapted task at once. -->
<!-- source: Bommasani et al. 2021 (arXiv:2108.07258), Fig. 2 -->

---

# If it reads every book, does it owe the authors anything?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-robot-reading.jpg -->
<!-- caption: Sit with this one. The hardest questions in this field are not technical. -->
<!-- source: Unsplash License, https://unsplash.com/photos/black-and-white-robot-toy-on-red-wooden-table-zwd435-ewb4 -->

---

# The landscape, at a glance

<!-- img: visuals/ethics-landscape.png -->
<!-- caption: Six interacting fault lines, technical and social at once. You can't fix them with code alone, nor ignore them while writing code. -->
<!-- source: original figure; Bender et al. 2021 & Bommasani et al. 2021 -->

---

# Planning is the mitigation

<!-- img: visuals/premortem.png -->
<!-- caption: Bender et al. do not end with a filter to apply. They end with a step you run before the thing exists. -->
<!-- source: Bender et al., FAccT 2021, section 6; Klein, Harvard Business Review, 2007 -->

---

# What responsible builders can do

<!-- img: visuals/responsible.png -->
<!-- caption: One artifact, five habits: a model card for a fictional moderation classifier. -->
<!-- source: original figure (worked example); model cards (Mitchell et al., 2019); datasheets (Gebru et al., 2018) -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then you run a pre-mortem on a real deployment. -->
<!-- source: original figure -->

---

# Exercise

<!-- layout: section -->

---

# Activity: pre-mortem a real deployment (teams)

<!-- img: visuals/activity-premortem.png -->
<!-- caption: Five teams, five cases. Nobody defends a position today; every team is on the build side. -->
<!-- source: exercise/premortem-guide.md -->

---

# Running the pre-mortem

<!-- img: visuals/premortem-steps.png -->
<!-- caption: Four steps, 23 minutes. Keep an eye on the clock yourselves. -->
<!-- source: method: Klein 2007, via Bender et al. 2021 section 6 -->

---

# What your whiteboard should look like

<!-- img: visuals/premortem-board.png -->
<!-- caption: Two minutes per team at the board. Draw this shape, fill in your own case. -->
<!-- source: worked example, exercise/premortem-guide.md -->

---

# Take-home demo: measure the bias

<!-- img: visuals/bias-output.png -->
<!-- source: example run, weeks/week-15/class-01/exercise/bias_probe.py -->

- A measure-first WEAT-style probe on toy embeddings, fully offline
- Optional take-home lab; the point is to arrive with a number, not an opinion

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Ethics is built in, not bolted on. -->
<!-- source: original figure -->

---

# Looking ahead

<!-- img: visuals/looking-ahead.png -->
<!-- caption: One class left. Presentations start next time, come with your demo ready to run. -->
<!-- source: schedule/SCHEDULE.md; readings.md -->
