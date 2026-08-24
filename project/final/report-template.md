# [Project Title]

**Authors:** [names]  ·  **CSCI 3495, Final Report**

*6-8 pages, written like a scientific paper. Due finals week. Worth 20% (report + presentation).*

**Format: LaTeX, NeurIPS template.** Open the
[NeurIPS template on Overleaf](https://www.overleaf.com/latex/templates/formatting-instructions-for-neurips-2026/bjdwqfdkyftc) (style file `neurips_2026.sty`) and write into it.
The sections below are the content this course expects; keep the template's formatting as-is.
No prior LaTeX experience is assumed.

## Abstract
~150 words: problem, approach, key result.

## 1. Introduction
Problem, motivation, and your contribution/claim in 1-2 paragraphs.

## 2. Related work
The papers you build on or compare against (cite properly).

## 3. Method
What you did, precisely enough to reproduce. Include a diagram if helpful.

## 4. Experimental setup
Data, models (Ollama/HF), hyperparameters, baselines, metrics, compute (CPU).

## 5. Results
Tables/figures with baselines. State what worked and what didn't; negative results are fine and valued.

## 6. Analysis & discussion
Why did you get these results? Error analysis, ablations, limitations.

## 6b. Limitations, bias & societal impact (required)
A short, honest paragraph (required, graded): the main limitations of your study; any bias or fairness concerns in the data, models, or task; and one or two sentences on the broader societal impact (who could be helped or harmed if this were deployed). Negative or uncertain findings are fine.

## 7. Conclusion
Takeaways and future work.

## Reproducibility
How to reproduce key results in the course Docker image (one command or notebook).

## References
[1] ...

## Contribution statement (required for teams)
Include a **one-line per-member contribution statement** (one bullet per member) stating what each person did. Example:
- **[Member 1]:** built the RAG pipeline and ran the retrieval ablation.
- **[Member 2]:** curated the dataset, wrote the eval harness, and drafted Sections 4-5.
- **[Member 3]:** ran baselines, made the figures, and led the analysis.

A **confidential peer evaluation** is submitted separately to the instructor at this milestone (see `RUBRICS.md`); it privately splits 100 points across members and can adjust each member's individual project grade by up to ±15%.
