# Pre-mortem Workshop: Case Briefs, Worksheet & Facilitation Guide

A **pre-mortem** is a post-mortem you run before you build. Instead of asking
"what could go wrong?" (which invites optimism), you assume the project has
**already failed** and work backwards to the causes. The technique is Gary
Klein's ("Performing a Project Premortem," *Harvard Business Review*, 2007);
Bender et al. 2021 recommend it in section 6 as a way to build "an evaluation
culture that considers not only average-case performance and best-case
performance, but also worst-case performance."

Nobody argues a side today. Every team is on the build side, and the job is to
find the failure before it finds a person.

## Facilitator timeline (~40 min after the break)

| Min | Step |
|-----|------|
| 0-3 | Count off 1-5, teams find tables, hand out the case brief + worksheet |
| 3-8 | **Step 1**: write the obituary headline (one sentence, with a number) |
| 8-16 | **Step 2**: reverse-engineer four causes; tag fault line + pipeline stage |
| 16-20 | **Step 3**: score each cause (bad x likely x hard to notice), pick the top one |
| 20-26 | **Step 4**: name one concrete build change for the top two causes |
| 26-28 | Move to the boards, sketch the three-box chain |
| 28-38 | **Presentations**: 2 min per team, one failure + one fix |
| 38-40 | Debrief: which fault line showed up in every case? |

## The five cases

Each team gets ONE. They are deliberately ordinary: none is a cartoon villain,
and each is something a vendor is selling today.

### Team 1, Machine translation
A county emergency-management office auto-translates evacuation orders, boil-water
notices, and shelter addresses into 30 languages and pushes them as SMS. Volume is
too high and the deadlines too short for human review.
**Direct users:** county staff. **Indirect:** residents who read no English, at 2am,
acting immediately on what the text says.
*Real precedent worth naming in the debrief:* in 2017 a Palestinian man was arrested
after Facebook's machine translation rendered his "good morning" post as "attack them"
(cited in Bender et al. 2021). Fluent output, zero grounding, no reviewer.

### Team 2, Tutoring
A university deploys a 24/7 LLM tutor for intro CS. It answers homework questions,
explains error messages, and drafts the feedback TAs later sign off on.
**Direct users:** enrolled students. **Indirect:** students who have nobody else to ask,
and who therefore rely on it most; TAs whose judgment gets anchored by the draft.

### Team 3, Accounting
A small-business accounting product ships an agent that reads uploaded receipts,
categorizes every expense, and drafts the quarterly filing. The owner clicks approve.
**Direct users:** the business owner. **Indirect:** the tax authority, an auditor two
years later, employees whose reimbursements get miscoded.

### Team 4, Legal
A legal-aid nonprofit uses an LLM to draft eviction-defense filings for tenants who
cannot afford a lawyer. Two staff attorneys review what they can; the rest goes out.
**Direct users:** the nonprofit's staff. **Indirect:** tenants with a hearing next week,
the court clerk, the judge reading the filing.
*Real precedent:* *Mata v. Avianca* (S.D.N.Y. 2023), where filed briefs cited cases that
did not exist.

### Team 5, Medical
A clinic uses an LLM to draft visit summaries and to sort overnight patient-portal
messages into "urgent" and "routine" before a nurse sees the queue.
**Direct users:** clinicians. **Indirect:** the patient whose message got sorted into
"routine," the on-call nurse trusting the sort.

## The worksheet (copy this onto the board or a sheet per team)

**Step 1, the obituary.** It is one year after launch. The system has been shut down
and it made the news. Write the headline in one sentence, and put a **number** in it.

> "____________________________________________________ ."

A headline like "the model was biased" fails this step. "Sorted 1,200 urgent messages
as routine over eight months" passes: it names who, how many, and for how long.

**Step 2, four causes.** For each, fill the row:

| # | What went wrong | Fault line | Where it entered |
|---|-----------------|-----------|------------------|
| 1 | | bias / misinfo / environment / copyright / labor / safety | data, training, prompt, missing human, missing eval |
| 2 | | | |
| 3 | | | |
| 4 | | | |

Push past the first two. The obvious failure is already on somebody's checklist;
the pre-mortem exists to surface the other kind.

**Step 3, ranking.** Score each cause 1-3 on three axes and multiply:

| Axis | 1 | 2 | 3 |
|------|---|---|---|
| How bad if it happens | annoying | costly | someone is harmed |
| How likely | rare | plausible | expected |
| How hard to notice | obvious | needs looking | invisible for months |

The product runs 1 to 27. **Hard to notice is the axis people underweight**: a loud
failure gets fixed in a week, a quiet one ships for a year.

**Step 4, the change.** For your top two causes, name ONE concrete change to how the
system gets built, trained, or evaluated. "Add a disclaimer" is not a change; it moves
the risk onto the user. Pick from what this course actually built:

- curate and **datasheet** the training data (W15) instead of taking what scrapes easily
- **fine-tune** on in-domain examples (W9) so the failure mode is at least in distribution
- **retrieval** with a required citation (W11) so claims are checkable
- **refuse and escalate** below a confidence threshold (W12) instead of always answering
- a named **human sign-off** step, at a specific point, on a specific subset
- a held-out **eval set** plus a **red-team** suite (W9, W12) that includes the failure you just invented
- **logging and monitoring** after launch, so the quiet failure stops being quiet

## The board sketch (2 minutes per team)

Draw three boxes and two arrows: **data -> model -> people**. Write the failure into
whichever box it starts in, and pin the fix to the box it acts on. Circle the score.

Teams present **one** failure and **one** fix, not their whole worksheet. The
interesting claim is usually *where* in the chain the failure entered, since it is
often not the model.

## Debrief questions (5 min, whole room)

1. Which fault line appeared in **every** case? (Usually bias, sometimes safety.)
2. Whose failure was **hardest to notice**? That team has the most dangerous system.
3. Did anybody's fix require **not building it**, or building something smaller?
   Bender et al. count that as a legitimate output of a pre-mortem, not a failure of nerve.
4. Which fixes were engineering, and which were organizational? Note how many of the
   real safeguards are not code.

## Notes for the instructor

- **Participation-graded**, not right/wrong. Reward specificity: a named number, a named
  stage, a named person who gets hurt.
- Watch for two failure modes at the tables: **vagueness** ("it could be biased"), and
  **disclaimer fixes** ("we'd tell users to double-check"). Both are worth interrupting.
- If a team stalls at step 2, ask: "who is the person least like you who has to use this,
  and what does the system assume about them that is false?"
- The `bias_probe.py` take-home connects directly: it is the measurement instrument for
  the fault line that shows up most often here.
