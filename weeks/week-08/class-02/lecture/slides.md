---
marp: true
title: "W8C2: Alignment: RLHF, InstructGPT & DPO"
paginate: true
---

# Week 8, Class 2
## Alignment: RLHF, InstructGPT & DPO

<!-- layout: title -->
<!-- img: visuals/assets/photo-dog-treat.jpg -->
<!-- caption: Reward the behavior you prefer and it sticks. Today: doing exactly this to a language model. -->
<!-- source: Pete Bellis, via Wikimedia Commons, CC0 -->

---

# Today

<!-- img: visuals/agenda.png -->
<!-- caption: Alignment, RLHF, DPO, and whose values first; Quiz 8 right before the break; then the standoff and reward-model fit. -->
<!-- source: tiles: Ouyang et al. 2022 Fig. 2; Rafailov et al. 2023 Fig. 1; DeepSeek-R1 2025 Fig. 1; Unsplash -->

---

<!-- layout: section -->
# The Alignment Problem

---

# Capable is not the same as aligned

<!-- img: visuals/alignment-problem.png -->
<!-- caption: We want helpful, honest, and harmless outputs. SFT teaches form; alignment teaches preference. -->
<!-- source: original figure -->

---

# Why not just write more demonstrations?

<!-- img: visuals/compare-vs-write.png -->
<!-- caption: Humans are far better at comparing two responses than at producing the ideal one. -->
<!-- source: original figure -->

---

# Alignment is a human handshake

<!-- img: visuals/assets/photo-handshake.jpg -->
<!-- caption: Every aligned model is built on thousands of human judgments about which answer is better. People are in the loop. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1521791136064-7986c2920216 -->

---

<!-- layout: section -->
# RLHF: the InstructGPT Pipeline

---

# InstructGPT (Ouyang et al., 2022)

<!-- img: visuals/assets/instructgpt-2022-fig-1.png -->
<!-- caption: The dashed line means "tied with the 175B SFT model." Even the 1.3B aligned models clear it; plain 175B GPT-3 never does. Alignment, not scale, drove perceived quality. -->
<!-- source: InstructGPT, Ouyang et al. 2022 (arXiv:2203.02155), Fig. 1 -->

---

# The RLHF pipeline

<!-- img: visuals/assets/instructgpt-2022-fig-2.png -->
<!-- caption: The most reproduced diagram in alignment. Note the human's job shrinks each step: write a full answer, then only rank answers, then step away entirely. -->
<!-- source: InstructGPT, Ouyang et al. 2022 (arXiv:2203.02155), Fig. 2 -->

---

# Stage 1: Supervised fine-tuning

<!-- img: visuals/stage1-sft.png -->
<!-- caption: Humans write demonstrations; the base model imitates them. A reasonable, not-yet-preference-aligned policy. -->
<!-- source: original figure -->

---

# Stage 2: Reward model

<!-- img: visuals/stage2-rm.png -->
<!-- caption: Humans rank sampled responses; a model learns to score any response with a pairwise ranking loss. -->
<!-- source: original figure -->

---

# Stage 3: RL optimization (PPO)

<!-- img: visuals/stage3-ppo.png -->
<!-- caption: PPO pushes the policy toward higher reward, with a KL penalty keeping it close to the SFT model. -->
<!-- source: PPO, Schulman et al. 2017; original figure -->

---

# Why the KL penalty matters

<!-- img: visuals/reward-hacking.png -->
<!-- caption: Over-optimize the imperfect reward model and you get reward hacking: high RM score, low human preference. -->
<!-- source: original figure (illustrative) -->

---

# RLHF is powerful but painful

<!-- img: visuals/rlhf-painful.png -->
<!-- caption: Four models, unstable RL, and a separate reward-model run. Naturally: can we skip the RL loop? -->
<!-- source: original figure -->

---

# What if we could skip the reinforcement learning entirely?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-road.jpg -->
<!-- caption: That is exactly the question DPO answered in 2023. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1469854523086-cc02fe5d8800 -->

---

<!-- layout: section -->
# DPO: Alignment without RL

---

# The DPO shortcut

<!-- img: visuals/assets/dpo-2023-fig-1.png -->
<!-- caption: The RLHF objective has a closed-form optimal policy, so the same preference pairs can train the LM directly: the model implicitly acts as its own reward model. -->
<!-- source: DPO, Rafailov et al. 2023 (arXiv:2305.18290), Fig. 1 -->

---

# DPO vs. RLHF

<!-- img: visuals/dpo-vs-rlhf.png -->
<!-- caption: DPO matches or exceeds PPO-RLHF on the paper's benchmarks with a simpler, more stable recipe (Rafailov et al. 2023). -->
<!-- source: DPO, Rafailov et al. 2023 (arXiv:2305.18290) -->

---

<!-- layout: section -->
# A Third Path: Verifiable Rewards

---

# What if the reward is just "is this answer correct?"

<!-- layout: statement -->
<!-- img: visuals/assets/photo-road.jpg -->
<!-- caption: For math and code, a program can grade the answer. No human labels, no learned reward model. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1469854523086-cc02fe5d8800 -->

---

# RLVR: Reinforcement Learning with Verifiable Rewards

<!-- img: visuals/rlvr-idea.png -->
<!-- caption: The reward comes from a verifier (unit tests, a checked numeric answer), not from a learned reward model. -->
<!-- source: RLVR, DeepSeek-Math (Shao et al. 2024, arXiv:2402.03300) and Tulu 3 (Lambert et al. 2024, arXiv:2411.15124) -->

---

# GRPO: score answers against each other

<!-- img: visuals/grpo-group.png -->
<!-- caption: Sample a group of answers to one prompt; each answer's advantage is its reward minus the group average. No value network. -->
<!-- source: GRPO, DeepSeek-Math (Shao et al. 2024, arXiv:2402.03300) -->

---

# DeepSeek-R1: pure RL elicits reasoning

<!-- img: visuals/assets/deepseek-r1-2025-fig-1.png -->
<!-- caption: GRPO on rule-based rewards alone lifts AIME pass@1 from 15.6% to 71.0%, past the human-participant line. The right panel is why: the model teaches itself to think in ever-longer chains (the aha moment). -->
<!-- source: DeepSeek-R1, DeepSeek-AI 2025 (arXiv:2501.12948), Fig. 1 -->

---

# RLHF vs. DPO vs. RLVR

<!-- img: visuals/three-paradigms.png -->
<!-- caption: Preferences teach taste for open-ended answers; verifiable rewards teach correctness on checkable tasks. Different signals, different jobs. -->
<!-- source: Ouyang 2022 (2203.02155); Rafailov 2023 (2305.18290); DeepSeek-R1 2025 (2501.12948) -->

---

<!-- layout: section -->
# Whose Values?

---

# A balancing act

<!-- img: visuals/assets/photo-scale.jpg -->
<!-- caption: Helpfulness and harmlessness pull in opposite directions. Alignment quietly decides where the needle rests. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1589994965851-a8f479c573a9 -->

---

# Helpfulness vs. harmlessness

<!-- img: visuals/help-vs-harm.png -->
<!-- caption: These goals conflict; alignment chooses a point on the trade-off, encoded in the preference data. -->
<!-- source: HH-RLHF, Bai et al. 2022 (arXiv:2204.05862) -->

---

# Whose preferences become the model's values?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-crowd.jpg -->
<!-- caption: A small group of labelers and their guidelines speak for everyone the model will ever serve. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1529156069898-49953e39b3ac -->

---

# Who decides what is preferred?

<!-- img: visuals/whose-preferences.png -->
<!-- caption: Labeler demographics and guidelines shape the model; preference data is a policy document as much as a dataset. -->
<!-- source: original figure -->

---

# Quiz 8

<!-- img: visuals/quiz.png -->
<!-- caption: Covers the Week 8 lecture + InstructGPT/RLHF (Ouyang 2022); optional readings are never quizzed. Clear your desk: finish early and your break starts early; need more time and it comes out of the break. -->
<!-- source: original figure -->

---

# Break

<!-- img: visuals/break.png -->
<!-- caption: Five minutes: stretch, refill water, compare notes. Then the reward-model standoff. -->
<!-- source: original figure -->

---

# Activity: the reward-model standoff

<!-- img: visuals/activity-standoff.png -->
<!-- caption: Teams label the same preference set independently, then surface clashes on sycophancy, honesty, and harm. -->
<!-- source: original figure -->

---

# Whose judgment becomes the reward?

<!-- layout: statement -->
<!-- img: visuals/assets/photo-standoff.jpg -->
<!-- caption: Compare your team's rankings before fitting. Where you disagree is exactly where the model's values get decided. -->
<!-- source: Unsplash License, https://images.unsplash.com/photo-1543269865-cbf427effbad -->

---

# Then fit the reward model

<!-- img: visuals/exercise-code.png -->
<!-- caption: Feed your labels in: fit per-response scores by gradient descent on the Bradley-Terry loss, exactly RLHF's Stage 2. -->
<!-- source: weeks/week-08/class-02/exercise/preferences.py -->

---

# What you should see

<!-- img: visuals/exercise-output.png -->
<!-- caption: Your comparisons become scalar scores; the implied ranking recovers the preferences you labeled. -->
<!-- source: example run, weeks/week-08/class-02/solutions/preferences.py -->

---

# Recap

<!-- img: visuals/recap.png -->
<!-- caption: Alignment, RLHF's three stages, InstructGPT's result, DPO's simplification, and the helpfulness/harmlessness trade-off. -->
<!-- source: original figure -->
