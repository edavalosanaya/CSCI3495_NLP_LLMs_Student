# Project Grading Rubrics

## Proposal (5%)
| Criterion | Pts |
|-----------|----:|
| Clear, well-motivated problem & question | 1 |
| Relevant background / paper grounding | 1 |
| Concrete, feasible plan (data, models, eval) | 2 |
| Risks & feasibility (CPU/Docker) addressed | 1 |
| **Total** | **5** |

*Scope check (not scored separately):* the proposed scope should match team size (solo = baseline + one comparison; 2 = a second axis; 3 = ~2-3x a solo project). The instructor calibrates scope here so the final peer evaluation is about effort share, not scope mismatch.

## Mid-semester checkpoint (10%)
| Criterion | Pts |
|-----------|----:|
| Demonstrated progress since proposal | 3 |
| A working baseline with a real result | 3 |
| Clear remaining plan (and risks named) | 2 |
| Quality of the feedback you give your review group | 2 |
| **Total** | **10** |

*The deliverable is a **one-page** summary (abstract, method, preliminary results) presented in a four-person review group in Week 10, Class 1. The last 2 points are for being a useful reviewer, not for polish on your own slides: specific, actionable criticism of the three projects you hear.*

## Final report & presentation (20%)
| Criterion | Pts |
|-----------|----:|
| Problem framing & related work | 3 |
| Method correctness & clarity | 4 |
| Experiments: baselines, metrics, rigor | 3 |
| Analysis (error analysis, ablations, limitations) | 3 |
| **Limitations, bias & societal impact** | 2 |
| Reproducibility (runs in Docker, documented) | 2 |
| Writing quality | 1 |
| Presentation (5-8 min talk/demo + 2 min Q&A, all members) | 2 |
| **Total** | **20** |

The **Limitations, bias & societal impact** line requires a short, honest paragraph on the project's limitations, any bias or fairness concerns in the data/models/task, and a sentence on broader societal impact (see the report template). This is required for every project so ethics is assessed via the project, not only the exam.

**Rigor over scale:** small models/datasets done carefully beat large, sloppy efforts. Honest negative results are credited.

## Individual accountability (peer evaluation)

The three project milestones (proposal 5% + checkpoint 10% + final 20% = **35%**) are graded at the **team** level. To keep grading fair when contributions are uneven, a **confidential peer evaluation** at the **final** milestone adjusts each member's individual project grade. This is an adjustment layer, **not** a new grade component: the total project weighting stays 35%.

**How it works (solo projects skip this):**
- At the final submission, every team member privately submits a short form (to the instructor, not shared with teammates) that **splits 100 points across all members including themselves** to reflect each person's share of the real work, plus a one-line justification per teammate.
- The instructor averages the peer-assigned shares for each member and compares that member's average share to an **equal share** (100 / team size). This yields a **contribution factor**.
- Each member's **individual project grade** (their share of the full 35%) is scaled by that factor, **capped to the range [0.85, 1.15]** so no single review can swing a grade by more than **±15%**. A member who contributed a normal, roughly equal share is unaffected (factor = 1.0).

**Exact scaling rule.**
```
equal_share      = 100 / team_size
avg_share_i      = mean of the shares teammates assigned to member i
contribution_i   = avg_share_i / equal_share
factor_i         = clamp(contribution_i, 0.85, 1.15)     # ±15% cap
individual_grade_i = team_project_grade * factor_i        # then clamp to [0, max]
```
- Grades are capped at the maximum (you cannot exceed 100% of the project weight).
- **Guardrails:** the instructor reviews any factor that falls below 0.90 or above 1.10 against the git history, contribution statements, and checkpoint records before applying it; a single outlier review that conflicts with the evidence may be discarded. Non-submission of the peer form defaults that reviewer to an equal split.

**Worked example (3-person team, project grade 30/35).** Equal share = 33.3. Peers give member A an average share of 40 → contribution 1.20 → clamped **factor 1.15** → A's grade = 30 × 1.15 = 34.5, capped at 35 → **34.5/35**. Member C averages 25 → contribution 0.75 → clamped **factor 0.85** → C's grade = 30 × 0.85 = **25.5/35**. Member B averages ~33.3 → factor 1.0 → **30/35** (unchanged).
