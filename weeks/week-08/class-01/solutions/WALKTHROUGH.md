# W8C1 Walkthrough: BPE from scratch, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `bpe.py` in this folder. Every code block below is taken
from it, and every printed value was produced by running it on the demo corpus
`["low low low low low", "lower lower", "newest newest newest", "widest"]`.

---

## Given, `build_vocab`

```python
def build_vocab(corpus: list[str]) -> dict[tuple[str, ...], int]:
    vocab: Counter = Counter()
    for line in corpus:
        for word in line.lower().split():
            symbols = tuple(word) + (END,)
            vocab[symbols] += 1
    return dict(vocab)
```

**Why a tuple of symbols rather than a string.** The whole algorithm is symbols
merging into bigger symbols. After one merge a word is `("l","o","w","er","</w>")`,
where `"er"` occupies one slot. A string cannot express "these two characters are
now one unit", and the pair-counting step would then find pairs that straddle a
merged symbol. Tuples are also hashable, so they work as dict keys.

**Why `</w>` exists.** It marks word ends, which does two jobs. It distinguishes
`er` inside a word from `er` that ends one (English suffixes are a real
distinction). And it lets the tokenizer reconstruct spacing, since without it
`["low", "est"]` and `["lowest"]` would be indistinguishable after decoding.

**Frequency, not a set.** The counts are what makes BPE *statistical*. A pair in a
word that appears 1000 times should be merged before a pair in a word that
appears twice, and that only works if the frequency rides along.

```python
>>> build_vocab(["low low"])
{('l', 'o', 'w', '</w>'): 2}
```

---

## Step 1, `count_pairs`

```python
def count_pairs(vocab: dict[tuple[str, ...], int]) -> Counter:
    pairs: Counter = Counter()
    for symbols, freq in vocab.items():
        for a, b in zip(symbols, symbols[1:]):
            pairs[(a, b)] += freq
    return pairs
```

**`zip(symbols, symbols[1:])`** is the standard adjacent-pairs idiom: pair each
element with the one after it, stopping naturally at the end.

**`+= freq`, not `+= 1`.** This is the single most common bug in this step.
Counting occurrences of pairs *in the vocab* rather than *in the corpus* makes
every distinct word contribute equally regardless of how often it appears, and
BPE stops being frequency-driven. The test uses a word with frequency 2 to catch
exactly this.

---

## Step 2, `merge_pair`

```python
    a, b = pair
    merged = a + b
    new_vocab: dict[tuple[str, ...], int] = {}
    for symbols, freq in vocab.items():
        out: list[str] = []
        i = 0
        n = len(symbols)
        while i < n:
            if i < n - 1 and symbols[i] == a and symbols[i + 1] == b:
                out.append(merged)
                i += 2
            else:
                out.append(symbols[i])
                i += 1
        new_vocab[tuple(out)] = new_vocab.get(tuple(out), 0) + freq
    return new_vocab
```

**The manual index walk is deliberate.** The tempting shortcut is
`" ".join(symbols).replace(a + " " + b, merged)`, which is what the original BPE
paper's reference code does with a regex. It works, but it hides the mechanics
and breaks in interesting ways once symbols contain spaces or regex
metacharacters. The explicit walk is clearer for teaching and has no such
failure modes.

**`i += 2` after a match** prevents overlapping merges. Merging `("a","a")` in
`("a","a","a")` gives `("aa","a")`, not `("aa","aa")`. That is the correct
behavior: each symbol is consumed once.

**`new_vocab.get(tuple(out), 0) + freq`** rather than plain assignment, because
two distinct words can collapse to the same tuple after a merge, and their
frequencies must add. Overwriting silently loses counts, which then quietly
distorts every later merge decision.

**Returns a new dict** rather than mutating in place, so `train_bpe` can reason
about each round independently.

```python
>>> merge_pair(('e', 'r'), {('l', 'o', 'w', 'e', 'r', '</w>'): 1})
{('l', 'o', 'w', 'er', '</w>'): 1}
```

---

## Step 3, `train_bpe`

```python
def train_bpe(corpus: list[str], num_merges: int) -> list[tuple[str, str]]:
    vocab = build_vocab(corpus)
    merges: list[tuple[str, str]] = []
    for _ in range(num_merges):
        pairs = count_pairs(vocab)
        if not pairs:
            break
        # Most frequent pair; deterministic tie-break by the pair itself.
        best = max(pairs.items(), key=lambda kv: (kv[1], kv[0]))[0]
        vocab = merge_pair(best, vocab)
        merges.append(best)
    return merges
```

**The tie-break is not cosmetic.** On a small corpus, ties are the common case,
not the exception. `key=lambda kv: (kv[1], kv[0])` sorts by count and then by the
pair itself, so equal counts resolve alphabetically instead of by dict insertion
order. There is a dedicated test (`test_step4_train_is_deterministic`) because a
tokenizer that changes between runs would make every downstream model
irreproducible.

**`if not pairs: break`** handles the case where everything has merged into
single symbols and there is nothing adjacent left. Without it, `max` on an empty
Counter raises.

**What is returned is the merge list, not a vocabulary.** This surprises students,
and it is worth dwelling on: the *model* of a BPE tokenizer is an ordered
sequence of rewrite rules. The vocabulary is derivable from it, but the order is
the thing that must be preserved, because merge 9 can only fire on symbols that
merges 1 to 8 created.

```python
>>> train_bpe(demo, num_merges=3)
[('o', 'w'), ('l', 'ow'), ('low', '</w>')]
```

Merge 2 consumes merge 1's output; merge 3 consumes merge 2's. The compounding is
the algorithm.

---

## Given, `encode_word`

```python
def encode_word(word: str, merges: list[tuple[str, str]]) -> list[str]:
    symbols: list[str] = list(word.lower()) + [END]
    for a, b in merges:
        merged = a + b
        ...same walk as merge_pair...
    return symbols
```

**Encoding replays training.** Start from characters, then apply every learned
merge **in training order**. Applying them in a different order, or applying only
the ones that "fit", gives a different and generally worse segmentation.

**No vocabulary lookup, and no unknown-token branch.** Any string can be encoded,
because the worst case is that no merge fires and you get characters back. That
is why subword tokenizers eliminated the `<UNK>` token that plagued word-level
models.

```python
>>> encode_word("lowest", merges)
['low', 'est</w>']
```

**"lowest" is not in the training corpus.** It is assembled from `low` (learned
from "low" and "lower") and `est</w>` (learned from "newest" and "widest"). This
one line is the payoff of the entire session; make sure students run it and
notice.

---

## Running it

```
Learned merges (in order):
   1. 'o' + 'w'
   2. 'l' + 'ow'
   3. 'low' + '</w>'
   4. 't' + '</w>'
   5. 's' + 't</w>'
   6. 'e' + 'st</w>'
   7. 'w' + 'est</w>'
   8. 'n' + 'e'
   9. 'ne' + 'west</w>'
  10. 'r' + '</w>'

Encoding 'lowest': ['low', 'est</w>']
```

**Read the merge list aloud as a narrative.** Merges 1 to 3 build the word "low"
because it is the most frequent thing in the corpus. Merges 4 to 6 build
`est</w>` from the end backwards, which is the English superlative suffix,
discovered with no linguistic input whatsoever. Merge 7 then makes `west</w>`,
and merge 9 makes `newest</w>` a single token because it appears three times.

Three points to draw out:

1. **Morphology emerges from frequency.** Nobody encoded that *-est* is a suffix.
   It is a suffix *because* it recurs across different stems, and BPE finds
   exactly the recurring pieces.
2. **Vocabulary size is a dial, not a fact.** `num_merges` decides how coarse the
   tokens are. Few merges means near-character tokens (long sequences, small
   vocab); many merges means whole words (short sequences, large vocab). Real
   models pick a point on that trade-off, typically 32k to 128k merges.
3. **The corpus decides the tokenizer.** This one learned "low" and "est" because
   that is what it saw. A tokenizer trained mostly on English fragments other
   languages into many more tokens, which means those users pay more per word in
   context length and in API cost. That is the fertility issue from lecture, and
   the third stretch goal makes it concrete in about a minute.

**Connecting forward:** every model in the rest of the course sits on top of a
tokenizer built exactly this way. When W9C2 discusses why a model cannot count
the letters in a word, or why it is bad at arithmetic on long numbers, the answer
is usually visible in the token boundaries students just learned to compute.
