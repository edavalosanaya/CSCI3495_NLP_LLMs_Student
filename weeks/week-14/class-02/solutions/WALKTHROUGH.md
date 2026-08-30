# W14C2 Walkthrough: Reproducibility check

Instructor reference. There is no student implementation this session: the
checker ships complete, and the lab is running it on their own project before the
final report.

---

## What the tool does, and what it cannot do

`repro_check.py` is a **static** check. It parses the file with `ast` and looks
for seeding calls. It never executes anything, makes no network calls, and can
therefore be run safely on a student's unfamiliar code.

That limitation produces the session's best teaching moment, and it is built into
Step 2 of the README on purpose:

```
Reproducibility check: weeks/week-02/class-01/solutions/ngram_lm.py
  [WARN] no RNG seeding found, set seeds for reproducible results.
```

**That warning is wrong.** `ngram_lm.py` is fully deterministic; it constructs a
local `random.Random(seed)` inside `generate` rather than calling a global
`random.seed`. The checker looks for the global pattern and misses the local one.

Use it. Ask the class what the tool actually measured (a syntactic pattern) versus
what it claims to measure (reproducibility), and note that this is the same gap
they met in W9C2's keyword-based hallucination detector and W7C2's substring
grader. **Cheap automated checks report what they can see.** They are worth
running and worth not trusting.

---

## Running the session

The checker is a five-minute item inside a session that is mostly project
work and 1-on-1 feedback. Do not let it expand.

**The instruction that matters is Step 4's second half**: run the real entry point
in Docker, twice, and compare the output. Every "works on my machine" problem
students hit at the final is caught by that, and no static tool can substitute
for it.

**Common real findings** when students run this on their own projects:

- A notebook that only runs top-to-bottom after a manual cell was executed first.
- Seeds set for `torch` but not `random` or `numpy`, so results move anyway.
- Data loaded from an absolute path on the student's laptop.
- A model downloaded at runtime with no pinned revision.

Only the second of those is visible to the checker. The others surface when they
run it in the container, which is the habit this session is trying to build.
