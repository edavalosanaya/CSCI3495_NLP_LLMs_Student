# W15C1 Walkthrough: Bias probe, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.** The complete file is `bias_probe.py` in this folder.

This is a simplified WEAT (Caliskan et al. 2017), and how it is framed matters
more than the code, which is short.

---

## Step 1, `cosine`

Fourth appearance in the course: W3C1 (sparse dicts), W3C2 (numpy), W6C1 (lists),
here. **Worth naming the repetition explicitly**: the instrument used to measure
social bias in a representation is the same three lines as the search engine from
week three. There is no special "bias mathematics".

Zero-vector guard as always.

---

## Step 2, `association`

Mean cosine with attribute set `A` minus mean cosine with `B`. Positive leans
toward `A`.

**Sets, not single words**, and that is a deliberate improvement over a
single-pair direction. One word pair carries its own idiosyncrasies; averaging
over a set of attribute words is what makes the direction about the *category*
rather than about one token.

---

## Step 3, `effect`

Average `association` over target set `X`, average over `Y`, take the difference.

**The sign-flip test is a real check, not a formality.** If swapping the
attribute sets does not flip the sign, the function is not measuring the thing it
claims to. Probes that fail their own symmetry properties are a genuine problem
in this literature.

---

## Running it

```
  engineer     association(male - female) = +1.014
  programmer   association(male - female) = +1.021
  scientist    association(male - female) = +0.869
  nurse        association(male - female) = -1.016
  teacher      association(male - female) = -0.884
  homemaker    association(male - female) = -1.132
  EFFECT (career leans male & care leans female) = +1.979
```

**Lead with the caveat, then the numbers.** These vectors are hand-built for the
exercise. The association was put there on purpose, so this run demonstrates that
the student's arithmetic is correct and demonstrates nothing whatsoever about
language models or the world. Overclaiming here would be exactly the failure the
session is about.

**What is real is the instrument.** Caliskan et al. 2017 ran this probe on
embeddings trained on ordinary web text and recovered documented human
implicit-association results, including this career/family pattern; Bolukbasi et
al. 2016 found the same in word2vec. Students have built the measuring device;
the empirical finding belongs to those papers, and the stretch goal (run it on
real vectors) is where the two meet.

**Why the lab is "measure first, then argue".** The pre-mortem in Part A is
better when at least one person at the table has computed an effect size: a
cause with a measured number attached outranks four with adjectives. "It's
just math" is a harder position to hold after you have written the math and
watched it produce a number with a sign and a magnitude. Equally, students who
have seen how much the result depends on *which words you put in the sets* are
better placed to be skeptical of any single published bias number.

**The three questions worth putting on the board**, all of which the code makes
concrete:

1. Who chose the word sets, and would a different choice change the sign?
2. The vectors learned this from text nobody curated. Who is responsible?
3. If this embedding sits inside a resume screener, at what point in the pipeline
   would anyone have noticed?
