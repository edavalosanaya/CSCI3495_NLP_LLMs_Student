# Semester Project: NLP & LLMs

A semester-long, original project. You will choose a direction, scope it, get feedback at two checkpoints, and deliver a paper-style report (LaTeX, NeurIPS template) and a talk.

## Three tracks (pick one)
1. **New method:** propose and test a novel idea (a prompting strategy, an agent design, a small architectural/training tweak, a new evaluation).
2. **Reproduction:** re-implement and verify results from a published paper (fully or a key subset) and report what replicated.
3. **Benchmark / analysis:** systematically measure and compare models/methods, or analyze a phenomenon (bias, robustness, prompt sensitivity, scaling).

All tracks must run on **free, open-source tools** (Python + Ollama/Hugging Face) on a laptop CPU. No paid APIs required. Small models (≤ ~3B) and small datasets are expected. **Scientific rigor matters more than scale.**

## Teams
Individuals or teams of up to **3**. Scope scales with team size (a 3-person team is expected to do ~2-3× a solo project). Every member must contribute and be able to explain the whole project.

**Scope expectation by team size** (state your target scope in the proposal):
- **Solo:** one clear question with a baseline + one comparison or ablation.
- **2 people:** the above plus a second axis (an extra model/dataset/method or a deeper analysis).
- **3 people:** roughly 2-3× a solo project (e.g., multiple models and datasets, or several ablations, or method + reproduction + analysis).

**Individual accountability.** Because the project is up to 35% of the course grade and done in teams, a **confidential peer evaluation** at the final milestone adjusts each member's individual project grade by up to **±15%** (a member who contributed a roughly equal share is unaffected). Every final report must also include a **per-member contribution statement**. This is a fairness adjustment layer only; the total project weighting stays 35%. See `RUBRICS.md` for the exact rule.

## Timeline & milestones
| Milestone | Week | Weight | Deliverable |
|-----------|------|-------:|-------------|
| **Proposal** | 5 | 5% | **1-page** written plan, submission only (`proposal/`) |
| **Mid-semester checkpoint** | 10 | 10% | 1-page summary (abstract, method, preliminary results) + in-class peer-review round (`checkpoint/`) |
| **Final report & presentation** | 15 / finals | 20% | 6-8 page NeurIPS-format report + **5-8 min** talk/demo + **2 min** Q&A (`final/`) |

A **work + instructor-feedback session** is provided in **Week 14, Class 2** before final presentations; project work/feedback time is also built into **Week 9**.

### How the checkpoint runs (Week 10, Class 1, 30 min)
There are no whole-class presentations. Instead:
- **Before class:** each project writes **one page** (one side) covering **abstract, method, and preliminary results**. Bring a copy for yourself; you hand it in at the end.
- **In class:** you sit in a **review group of four**, drawn from **different projects** so you get outside eyes.
- **Four rounds of ~7 minutes:** one person presents their page for **3 minutes**, the other three respond for **4 minutes**. Rotate until everyone has presented once.
- **Feedback the group owes each presenter:** what is the actual claim and does the page support it, what is the biggest risk to finishing by Week 15, and one concrete experiment to run next.
- **The instructor sits in with one group per round** and joins the critique; every page is collected and returned with written comments.

This scales to a full class and gets every student critiqued by three peers plus the instructor, which a round of 5-minute talks at the front of the room does not.

## Deliverable formats
- **Reports:** PDF written like a scientific paper, in **LaTeX** using the **NeurIPS** template
  ([Overleaf](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc), style file `neurips_2026.sty`). Section-by-section guidance is in
  `final/report-template.md`. Use clear figures/tables; cite sources. No LaTeX experience is
  assumed; Overleaf runs in a browser and is free.
- **Code:** a reproducible repo/folder with a README and a single command (or notebook) to reproduce key results. Must run in the course Docker image.
- **Presentations:** a **5-8 minute** talk or live demo followed by **2 minutes of Q&A**.
  Slides in any tool (Marp/PDF/Slides). Everyone on the team presents.

## Topic ideas (starters)
- Compare decoding strategies (greedy/beam/top-p/temperature) on a small generation task.
- Reproduce a chain-of-thought result on a small arithmetic/reasoning set with a local model.
- Measure prompt sensitivity: how much do paraphrased prompts change accuracy?
- Build and evaluate a RAG system over a small document collection; ablate retrieval.
- LoRA-fine-tune a small model on a niche task; quantify the gain vs. base.
- Bias probing of word embeddings or a small LLM; quantify and discuss.
- Build a tool-using agent and measure its success rate on a task suite.
- Benchmark several Ollama models on a slice of MMLU; analyze error types.

## Academic integrity
You may use LLMs as coding assistants (disclose it), but the ideas, analysis, and writing must be yours, and you must be able to explain everything. Reproductions must clearly separate "from the paper" vs. "our results."

## Grading
Each milestone has a rubric in its folder (`proposal/`, `checkpoint/`, `final/`). See `RUBRICS.md` for the consolidated view.
