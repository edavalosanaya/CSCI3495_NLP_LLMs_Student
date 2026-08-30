# W15C1 Activity: Pre-mortem Workshop (in class) + Bias Probe (take-home)

Two parts: the **in-class** activity is a **team pre-mortem** on a real deployment;
the **bias-probing demo** is a **measure-first / take-home lab** that runs fully offline.
Evidence beats opinion: measure first, then argue with numbers.

## Before you code: the picture and the math

![Bias pipeline: human text encodes stereotypes, models learn and amplify them, harm falls unevenly](../lecture/visuals/bias-fairness.png)

The take-home lab measures the first arrow of that pipeline directly. Your finished code reproduces this run:

![Example run of bias_probe.py showing per-word associations and the overall EFFECT score](../lecture/visuals/bias-output.png)

The three functions you implement are exactly these three equations (same notation as the docstrings):

$$\cos(a, b) = \frac{a \cdot b}{\lVert a \rVert \, \lVert b \rVert}$$

$$s(w, A, B) = \frac{1}{|A|} \sum_{a \in A} \cos(w, a) \; - \; \frac{1}{|B|} \sum_{b \in B} \cos(w, b)$$

$$\mathrm{effect}(X, Y, A, B) = \frac{1}{|X|} \sum_{x \in X} s(x, A, B) \; - \; \frac{1}{|Y|} \sum_{y \in Y} s(y, A, B)$$

In words: `cosine` scores how similar two word vectors are, `association` asks whether one word sits closer to attribute set A (male terms) or set B (female terms), and `effect` averages those associations over the career words X versus the care words Y. A large positive `effect` means the embedding space encodes the stereotype pattern career leans male, care leans female. **Check yourself before coding:** in the example run, why is `association(male - female)` for *nurse* negative at -1.016? (Because *nurse* has a higher mean cosine to the female attribute words than to the male ones, so the subtraction in $s(w, A, B)$ comes out below zero.)

---

## Part A: Pre-mortem workshop (TEAM): IN CLASS

A **pre-mortem** is a post-mortem you run before you build: you assume the project has
**already failed**, then reverse-engineer the causes while the plan is still cheap to
change. Klein (2007) invented it for project management; Bender et al. (2021, section 6)
recommend it for language models specifically, as a way to evaluate the **worst** case
rather than the average one.

You are not arguing a side. Every team is on the build side.

**Teams of 3-4.** Count off 1-5; your number is your case:

| Team | Case |
|------|------|
| 1 | A county auto-translates evacuation orders into 30 languages, no human review |
| 2 | A university ships a 24/7 LLM tutor for intro CS that also drafts TA feedback |
| 3 | An agent categorizes a small business's expenses and drafts its quarterly filing |
| 4 | A legal-aid nonprofit drafts eviction-defense filings for tenants with no lawyer |
| 5 | A clinic drafts visit summaries and triages overnight patient-portal messages |

**The four steps (~23 min):**

1. **Write the obituary (5 min).** One year post-launch, your system is shut down and in
   the news. Write the headline in one sentence, with a **number** in it. "It was biased"
   does not count; "sorted 1,200 urgent messages as routine over eight months" does.
2. **Reverse-engineer four causes (8 min).** For each, tag **which of the six fault lines**
   it is, and **where in the pipeline it entered**: the data, the training, the prompt, the
   missing human, or the missing evaluation. Push past the first two, the obvious ones are
   already on somebody's checklist.
3. **Rank them (4 min).** Score each cause 1-3 on *how bad*, *how likely*, and *how hard to
   notice*, then multiply (1 to 27). Hard-to-notice is the axis people underweight: a loud
   failure gets fixed in a week, a quiet one ships for a year.
4. **Change the build, not the disclaimer (6 min).** For your top two causes, name ONE
   concrete change: curate + datasheet the data · fine-tune in-domain · retrieval with a
   required citation · refuse-and-escalate below a confidence threshold · a named human
   sign-off step · a held-out eval set plus a red-team suite · logging and monitoring.

**Then 2 minutes at the whiteboard.** Draw three boxes and two arrows,
**data → model → people**, put the failure in the box it starts in, and pin the fix to the
box it acts on. Present **one** failure and **one** fix, not the whole worksheet.

Full case briefs, the scoring table and the worksheet are in `premortem-guide.md`.

---

## Part B: Bias probe (measure-first lab, offline)

**You will write three functions** in `bias_probe.py`, one per step, each with
its own check. This is a simplified WEAT (Caliskan et al. 2017).

`lab` is a shortcut for the long docker command. Set it up once per
terminal session, using the line for **your** shell:

```
# macOS / Linux (bash, zsh)
alias lab='docker compose -f docker/docker-compose.yml run --rm --no-deps course'

# Windows, PowerShell
function lab { docker compose -f docker/docker-compose.yml run --rm --no-deps course @args }

# Windows, Command Prompt
doskey lab=docker compose -f docker/docker-compose.yml run --rm --no-deps course $*
```

Rather work inside the image? This opens a shell there, and then every
command below runs without its `lab` prefix:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps course bash
```

Stuck for more than a few minutes on a step? The reference solution and a
step-by-step `WALKTHROUGH.md` are in `../solutions/`. **These labs are not
graded**, so reading them is not cheating: getting unstuck and finishing the
idea beats staring at a blank function.

```bash
lab python -m pytest weeks/week-15/class-01/exercise/test_bias_probe.py -k step1 -q
```

---

### Step 1, Cosine similarity

**Write:** `cosine(a, b)`. Return 0.0 for a zero vector.

**Done when:** `-k step1` gives `4 passed, 6 deselected`.

Fourth time in the course (W3C1, W3C2, W6C1, here). Worth noticing that the
measurement instrument for a social-bias probe is the same three lines as the
search engine.

---

### Step 2, Association

**Write:** `association(w, A, B)`, the mean cosine of `w` with attribute set `A`
minus the mean cosine with `B`. Positive leans toward `A`.

**Done when:** `-k step2` gives `3 passed, 7 deselected`.

---

### Step 3, Effect size

**Write:** `effect(X, Y, A, B)`: average `association` over target set `X`,
average it over `Y`, return the difference.

**Done when:** `-k step3` gives `3 passed, 7 deselected`.

**One test checks the sign flips when the attribute sets are swapped.** That is a
sanity property, not a detail: a probe whose sign does not respond to its own
definition is measuring nothing.

---

### Step 4, Run it

```bash
lab python -m pytest weeks/week-15/class-01/exercise/test_bias_probe.py -q
```

```
..........                                                               [100%]
10 passed
```

```bash
lab python weeks/week-15/class-01/exercise/bias_probe.py
```

```
  engineer     association(male - female) = +1.014
  programmer   association(male - female) = +1.021
  scientist    association(male - female) = +0.869
  nurse        association(male - female) = -1.016
  teacher      association(male - female) = -0.884
  homemaker    association(male - female) = -1.132
  EFFECT (career leans male & care leans female) = +1.979
```

**Read the caveat before the numbers.** These vectors are **hand-built for the
exercise**, so the association was put there deliberately. This run proves your
arithmetic works; it proves nothing about the world.

What *is* real is the instrument. Caliskan et al. 2017 ran this same probe on
word embeddings trained on ordinary web text and recovered documented human
implicit-association biases, including this career/family pattern. You have built
the measuring device; the finding belongs to the papers.

**Bring your number back.** "It's just math" is easier to say before someone has
computed the effect size themselves.

## Feed it back into your pre-mortem
Use your `effect` score the way step 2 asks you to: "this embedding associates technical
roles with men by +X, and my case pipes exactly that representation into a decision about
a person." A cause with a measured number attached outranks four with adjectives.

## Stretch goals
- Add your own target/attribute word lists and re-measure.
- Add a **neutral** sanity check (e.g., `tree`, `river`), the effect should vanish.
- Read about the original **WEAT** (Caliskan, Bryson, Narayanan, *Science* 2017) and
  discuss why measuring bias is itself a contested, value-laden choice.

Reference solution: `../solutions/bias_probe.py` (don't peek until you've tried!).
