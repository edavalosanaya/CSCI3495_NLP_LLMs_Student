# W1C2 Walkthrough: Text tools, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `text_tools.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it.

All four functions live on top of one module-level block of compiled patterns:

```python
_TOKEN_RE   = re.compile(r"\w+|[^\w\s]", re.UNICODE)
_EMAIL_RE   = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE     = re.compile(r"https?://\S+")
_MENTION_RE = re.compile(r"(?<!\w)@(\w+)")
```

Compiling once at import is a habit worth teaching: `re` caches patterns anyway,
but naming them documents intent far better than an inline string.

---

## Given, `normalize`

**The idea.** Two cheap transformations that make everything downstream
countable: case-fold so `The` and `the` are one word, and squeeze whitespace so
line breaks and double spaces stop creating phantom tokens.

```python
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()
```

**Why `\s+` and not `" "`.** The `+` makes the pattern match a whole *run* of
whitespace and replace it with one space, so tabs, newlines and runs of spaces
all collapse in a single pass. Replacing `" "` with `" "` would do nothing at
all.

**Order matters, slightly.** `.lower()` before the substitution and `.strip()`
after: the substitution can leave a single leading or trailing space (from
whitespace at the very ends), and `.strip()` cleans that up.

**What you should see:**

```python
>>> normalize("  Hello   WORLD\n")
'hello world'
```

**Worth saying out loud in class:** normalization is lossy. `US` and `us` become
the same token, and for a downstream task that cares about country names that is
a real error you introduced on line one. Every pipeline decision here is a
trade.

---

## Step 1, `tokenize`

**The idea.** Split on the boundary between "word characters" and everything
else, keeping punctuation as its own token rather than discarding it.

```python
def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(normalize(text))
```

**Reading `r"\w+|[^\w\s]"`.** It is an alternation of two alternatives:

- `\w+` matches a run of word characters (letters, digits, underscore).
- `[^\w\s]` matches exactly one character that is neither a word character nor
  whitespace, i.e. a single punctuation mark.

`findall` scans left to right taking the first alternative that matches at each
position, so `"hello,"` yields `hello` then `,`. Whitespace matches neither
alternative and is silently skipped, which is why no explicit split is needed.

**What you should see:**

```python
>>> tokenize("Hello, world!")
['hello', ',', 'world', '!']
>>> tokenize("NLP is fun.")
['nlp', 'is', 'fun', '.']
```

**The interesting failures**, which are the discussion, not a bug to fix:

```python
>>> tokenize("don't stop")
['don', "'", 't', 'stop']
>>> tokenize("see you :)")
['see', 'you', ':', ')']
```

The contraction splits into three tokens because `'` is punctuation under this
rule, and the emoticon is torn in half because `:` and `)` each match
`[^\w\s]` separately. Students will have argued about both at the whiteboard.
Neither answer is "correct"; what matters is that the rule is explicit and
consistent. Contrast with a learned tokenizer (BPE, Week 2), which decides these
boundaries from data instead of from a human's regex, and typically keeps
`don't` together because it is frequent.

---

## Given, `extract`

**The idea.** Three independent patterns run over the **raw** text, one per
category.

```python
def extract(text: str) -> dict[str, list[str]]:
    return {
        "emails": _EMAIL_RE.findall(text),
        "urls": _URL_RE.findall(text),
        "mentions": _MENTION_RE.findall(text),
    }
```

**Note it does not normalize first.** Lowercasing a URL can break it, and
collapsing whitespace would not help. Raw text in, structured data out.

**The three patterns.**

- `_EMAIL_RE = r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"`. The local part allows dots, plus
  and minus (so `a.b+x@mail.co` matches, and plus-addressing survives). The
  domain requires at least one literal dot, which is what stops it matching
  `bob@localhost`. The `\b` anchors keep it from matching a fragment inside a
  longer string.
- `_URL_RE = r"https?://\S+"`. Deliberately crude: scheme, then everything up to
  whitespace. It happily swallows a trailing `)` or `.` at the end of a
  sentence. Real URL matching is genuinely hard, and this is a good moment to
  say that the "obvious" regex is always an approximation.
- `_MENTION_RE = r"(?<!\w)@(\w+)"`. Two subtleties in one short pattern.

**The mention pattern, in detail.** The capturing group `(\w+)` means `findall`
returns `alice`, not `@alice`, which is what the tests expect. The lookbehind
`(?<!\w)` asserts the character before the `@` is not a word character. Without
it, an email address matches too:

```python
>>> re.findall(r"@(\w+)", "email bob@x.io and mention @bob")
['x', 'bob']                       # WRONG: 'x' came from inside the email
>>> extract("email bob@x.io and mention @bob")["mentions"]
['bob']                            # correct
```

This is the single best teaching example in the exercise: the naive pattern is
not *slightly* wrong, it silently invents a user named `x`. Precision problems in
rule-based NLP usually look like this, plausible garbage rather than a crash.

**What you should see:**

```python
>>> extract("Reach me at a.b+x@mail.co or bob@x.io")["emails"]
['a.b+x@mail.co', 'bob@x.io']
>>> out = extract("See https://x.io/p?q=1 from @alice and @bob_99")
>>> out["urls"]
['https://x.io/p?q=1']
>>> out["mentions"]
['alice', 'bob_99']
```

---

## Step 2, `edit_distance`

**The idea.** `D[i][j]` is the edit distance between the first `i` characters of
`a` and the first `j` of `b`. Every cell is decided by three neighbors, so fill
the table top-left to bottom-right and read the answer off the last cell.

```python
def edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    # D[i][j] = edit distance between a[:i] and b[:j]
    d = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        d[i][0] = i
    for j in range(n + 1):
        d[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            d[i][j] = min(
                d[i - 1][j] + 1,        # deletion
                d[i][j - 1] + 1,        # insertion
                d[i - 1][j - 1] + cost, # substitution
            )
    return d[m][n]
```

**The base cases are the whole trick.** `D[i][0] = i` says: to turn the first `i`
characters of `a` into the empty string, delete all `i` of them. `D[0][j] = j` is
the mirror image with insertions. Everything else follows mechanically.

**The off-by-one.** Row `i` corresponds to character `a[i-1]`, because row 0 is
the empty prefix. Comparing `a[i]` and `b[j]` instead of `a[i-1]` and `b[j-1]` is
the mistake almost every student makes, and it usually produces an answer that
is right for some pairs and wrong for others, which makes it hard to spot without
the parametrized test.

**Why cost 1 and not 2.** J&M present a variant where substitution costs 2
(equivalent to a delete plus an insert). We use plain Levenshtein, cost 1, which
is why `intention` to `execution` is 5 here and 8 in the cost-2 convention. If a
student's table disagrees with the code by exactly this pattern, that is the
reason.

**What you should see:**

```python
>>> edit_distance("intention", "execution")
5
>>> edit_distance("kitten", "sitting")
3
>>> edit_distance("", "abc")
3
>>> edit_distance("abc", "")
3
>>> edit_distance("same", "same")
0
```

**Complexity.** Time and space are both `O(len(a) * len(b))`. Worth mentioning
that the space can be reduced to `O(min(m, n))` by keeping only the previous
row, since each cell looks at most one row back. The full table is kept here
because it is what students traced by hand, and because you need it to recover
the alignment in the stretch goal.

---

## Running it

```
..........                                                               [100%]
10 passed
```

Ten, not six: the edit-distance test is parametrized over five string pairs.

**Where to take the discussion.** Ask students to run their finished tools on
real text they paste in. The failures arrive quickly, a URL that swallowed a
closing parenthesis, an email inside a signature block, a contraction split three
ways. That is the honest summary of rule-based NLP: precise, fast, auditable,
and brittle at exactly the edges where real language lives. The rest of the
course is about learning these rules from data instead of writing them.
