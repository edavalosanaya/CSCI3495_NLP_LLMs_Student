# W1C2 Lab: Tokenizer, Regex Extractor & Edit Distance

Build the four text tools that every later week leans on: cleaning text,
splitting it into tokens, pulling structured things out of it with regexes, and
measuring how far apart two strings are.

**You will write two functions** in `text_tools.py`. The other two are already
written for you, to read and run. Every step has its own check.

## Before you code: the picture and the math

The heart of this exercise is `edit_distance(a, b)`: the fewest single-character inserts, deletes, and substitutes that turn string $a$ into string $b$. You fill a table $D$ where $D[i,j]$ is the distance between the first $i$ characters of $a$ and the first $j$ characters of $b$, using the base cases and recurrence from lecture:

$$D[i,0] = i, \qquad D[0,j] = j$$

$$D[i,j] = \min \begin{cases} D[i-1,j] + 1 & \text{(deletion)} \\ D[i,j-1] + 1 & \text{(insertion)} \\ D[i-1,j-1] + [\,a_i \neq b_j\,] & \text{(substitution, cost 0 if the characters match)} \end{cases}$$

![Edit distance recurrence: each cell is the min of three neighbors](../lecture/visuals/recurrence.png)

Filled in row by row, the table looks like this worked example from the slides, `intention` to `execution`, where the answer is the bottom-right cell:

![Completed DP table for intention to execution, distance 5](../lecture/visuals/edit-distance.png)

Your finished code computes exactly that bottom-right cell, $D[\text{len}(a), \text{len}(b)]$, plus three regex-based text tools (`normalize`, `tokenize`, `extract`) that clean raw text and pull out emails, URLs, and @mentions with patterns like `@\w+`. Note we use cost 1 per substitution (Levenshtein), not the cost-2 variant J&M also mention. **Check yourself before coding:** in the recurrence, which of the three neighbor cells does a substitution come from, and when is its added cost 0? (The diagonal cell $D[i-1,j-1]$; cost 0 when $a_i = b_j$.)

## Warm-up Worksheet: Be the Tokenizer (pairs, whiteboard, ~15 min)

Before writing any code, work these out by hand with your partner at the board:

1. **Hand-tokenize three nasty strings.** Where do you split? Mark every spot you
   disagree on:
   - a contraction: `don't`
   - a URL: `http://t.co/x`
   - an emoji string: `see you :)`
2. **Hand-trace the edit-distance DP table** for `kitten` -> `sitting`. Fill the
   `(len+1) x (len+1)` grid using the base cases `D[i,0]=i`, `D[0,j]=j` and the
   min-of-three recurrence (insert / delete / substitute). The answer is **3**.
3. **Keep your table visible.** In Step 4 you implement exactly what you just
   traced, and you will compare your table against the code's.

## How this lab works

Each step tells you **what to write**, then exactly **how to check it**. The four
steps are independent, so a stuck step does not block the next one, but do them
in order the first time through.

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

Check **one step**:

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step1 -q
```

Check **everything**:

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -q
```

Some steps are **already written for you** and marked `(given)`. Run their
check, read the code, and use it as the pattern for the steps you do write. A
step you have not written yet reports `skipped`, never a failure, so the only
red you will ever see is a real wrong answer.

Stuck for more than a few minutes? Open `../solutions/WALKTHROUGH.md` at the
matching step. The full reference solution sits in `../solutions/` too. **These
labs are not graded**, so reading them is not cheating: getting unstuck and
finishing the idea beats staring at a blank function. Read the step you are on, not the whole file.

---

### Step 0, Orientation (nothing to write)

Open a Python shell in the container and confirm the file imports:

```bash
lab python
```

```python
>>> import sys; sys.path.insert(0, "weeks/week-01/class-02/exercise")
>>> import text_tools
>>> text_tools.normalize("  Hi   THERE ")
'hi there'
>>> text_tools.tokenize("hi")
NotImplementedError
```

`normalize` is written for you, so it answers. `tokenize` raises, and that error
is the starting line, not a bug: every function you still have to write raises
it until you fill in the body.

While you are here, get a feel for the one library you need:

```python
>>> import re
>>> re.sub(r"\s+", " ", "a   b\n\nc")
'a b c'
>>> re.findall(r"\w+", "Hello, world!")
['Hello', 'world']
```

`re.sub` replaces every match; `re.findall` returns every match. Those two calls
are most of Steps 1 to 3.

---

### Step 1, Normalize (given)

**Given, already written for you.** Read it in the starter, run its check,
and use it as the pattern for the steps you do write.

**What it does:** `normalize(text)`. Lowercase the text, collapse every run of
whitespace (spaces, tabs, newlines) to a single space, and strip the ends.

One `re.sub` plus `.lower()` and `.strip()` is enough. Note `\s+` matches a
*run* of whitespace, which is why one substitution handles doubled spaces and
newlines together.

**Done when:**

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step1 -q
```

```
.                                                                        [100%]
1 passed, 9 deselected
```

**Check it by hand:**

```python
>>> normalize("  Hello   WORLD\n")
'hello world'
```

**Why it matters:** every counting step in Weeks 2 and 3 assumes `"The"` and
`"the"` are the same word. Normalization is where that becomes true, and it is
also where information is thrown away, so it is a real modeling decision, not
just cleanup.

---

### Step 2, Tokenize

**Write:** `tokenize(text)`. Normalize first, then split into word tokens where
punctuation becomes its own token.

The pattern `r"\w+|[^\w\s]"` reads as "a run of word characters, **or** a single
character that is neither word nor whitespace". `re.findall` with that pattern
does the whole job.

**Done when:**

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step2 -q
```

```
..                                                                       [100%]
2 passed, 8 deselected
```

**Check it by hand:**

```python
>>> tokenize("Hello, world!")
['hello', ',', 'world', '!']
>>> len(tokenize("NLP is fun."))
4
```

Now try the strings from the warm-up worksheet:

```python
>>> tokenize("don't stop")
['don', "'", 't', 'stop']
>>> tokenize("see you :)")
['see', 'you', ':', ')']
```

**Compare these against what you and your partner decided at the board.** The
code splits `don't` into three tokens and tears the `:)` in half. Neither is
obviously right, and neither is what most real tokenizers do. That disagreement
is the point: tokenization is a choice, and Week 2 shows what a
statistically-learned tokenizer (BPE) chooses instead.

---

### Step 3, Extract (given)

**Given, already written for you.** Read it in the starter, run its check,
and use it as the pattern for the steps you do write.

**What it does:** `extract(text)`. Return `{"emails": [...], "urls": [...], "mentions": [...]}`
found in the **raw** text (do not normalize first, URLs are case-sensitive).

Three separate patterns, one per key. Starting points:

- emails: a run of word characters, dots, plus and minus signs, then `@`, then a
  domain with at least one dot
- urls: `https?://` followed by everything up to whitespace
- mentions: `@` followed by word characters

**Done when:**

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step3 -q
```

```
..                                                                       [100%]
2 passed, 8 deselected
```

**Check it by hand:**

```python
>>> extract("Reach me at a.b+x@mail.co or bob@x.io")["emails"]
['a.b+x@mail.co', 'bob@x.io']
>>> out = extract("See https://x.io/p?q=1 from @alice and @bob_99")
>>> out["urls"]
['https://x.io/p?q=1']
>>> out["mentions"]
['alice', 'bob_99']
```

**The trap worth hitting.** A naive mention pattern also fires inside an email
address, since `bob@x.io` contains an `@` followed by word characters:

```python
>>> extract("email bob@x.io and mention @bob")["mentions"]
['bob']
```

If yours returns `['x', 'bob']`, your mention pattern is matching the `@` in the
email too. The fix is a lookbehind asserting the `@` is not preceded by a word
character. Getting this right by staring at failing output is exactly how regex
work goes in practice.

**Why it matters:** this is your first taste of the rule-based end of NLP.
Regexes are precise, fast, and completely brittle, which is the tension the whole
course is about.

---

### Step 4, Edit distance

**Write:** `edit_distance(a, b)`, the Levenshtein distance, using the DP table you
traced on the whiteboard.

Build a `(len(a)+1) x (len(b)+1)` table. Fill row 0 and column 0 with the base
cases, then fill the rest with the min-of-three recurrence. The answer is the
bottom-right cell.

Watch the indexing: `D[i][j]` compares the **first `i`** characters of `a`
against the **first `j`** of `b`, so the characters you compare are `a[i-1]` and
`b[j-1]`, not `a[i]` and `b[j]`. That off-by-one is the single most common bug in
this step.

**Done when:**

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -k step4 -q
```

```
.....                                                                    [100%]
5 passed, 5 deselected
```

(Five, because that test is parametrized over five string pairs, including the
empty-string edge cases.)

**Check it by hand:**

```python
>>> edit_distance("intention", "execution")
5
>>> edit_distance("kitten", "sitting")
3
>>> edit_distance("", "abc")
3
>>> edit_distance("same", "same")
0
```

**Compare the first two against your whiteboard table.** They should match
exactly; if they do not, the table is right and the code is wrong far more often
than the reverse.

**Why it matters:** this is the first dynamic program in the course, and the
pattern (fill a table, each cell a small decision over its neighbors) comes back
for sequence alignment and, in spirit, for the attention and decoding algorithms
later on.

---

### Step 5, Run everything

```bash
lab python -m pytest weeks/week-01/class-02/exercise/test_text_tools.py -q
```

```
..........                                                               [100%]
10 passed
```

Then point your finished tools at something messy, a few lines pasted from a
real page or your own inbox, and see what they get wrong. Bring one failure to
the wrap-up discussion.

## Stretch goals

- Make `tokenize` handle `n't` / `'re` / `'s` as separate tokens (so `don't`
  becomes `do` + `n't`, which is what many real tokenizers do).
- Add `extract` support for hashtags (`#topic`).
- Return the **alignment** (the sequence of edits) from `edit_distance`, not just
  the number. Hint: keep a second table recording which of the three neighbors
  won each cell, then walk it backwards from the bottom-right.

A full reference solution is in `../solutions/text_tools.py`, and the
step-by-step explanation is in `../solutions/WALKTHROUGH.md` (don't peek until
you've tried).
