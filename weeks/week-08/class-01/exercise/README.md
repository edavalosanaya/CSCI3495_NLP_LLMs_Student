# W8C1 Lab: Byte Pair Encoding

## 1. Learning objective

Build the tokenizer every modern LLM uses: start from characters and
repeatedly fuse the most frequent adjacent pair, until common words are single
symbols and rare ones survive as pieces.

You write three functions in `bpe.py`. The initial vocabulary and the encoder
are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-08/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `count_pairs`

Counts are weighted by word frequency, so a word appearing five times pushes
its pairs five times as hard.

Walk each word's adjacent pairs and add that word's frequency, not 1.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 7 deselected
```

## 4. Implement `merge_pair`

A merge replaces every adjacent occurrence of the chosen pair with one new
symbol:

$$\text{replace every adjacent } (a,b) \text{ with the single symbol } ab$$

Scan left to right, emitting the joined symbol and skipping both halves.
Add frequencies when two words collapse to the same symbols.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 7 deselected
```

## 5. Implement `train_bpe`

![BPE merges: start from characters, repeatedly fuse the most frequent adjacent pair](../lecture/visuals/bpe-merges.png)

Each round picks the most frequent adjacent pair across the whole corpus and
merges it everywhere:

$$(a, b)^* = \arg\max_{(a,b)} \; \mathrm{count}(a, b)$$

The merges are an ordered list, and encoding replays them from the start.

Count, pick the winner with a deterministic tie-break, merge, record. Stop
early when no pairs remain.

```bash
pytest -k step3 -q
```

```
...                                                                      [100%]
3 passed, 5 deselected
```

## 6. Run it, then break it

```bash
python bpe.py
```

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

`lowest` never appeared in the training corpus, yet it encodes into two clean
pieces. Each experiment below is a one-line edit; undo it before the next.

1. Count pairs the wrong way. In `count_pairs`, add `1` instead of the word's
   frequency. The top pairs go from `[('l','o'), 7), (('o','w'), 7)]` to counts
   of 2, and the merge order changes. Which idea about language does weighting
   by frequency encode?
2. Starve it of merges. Encode `lowest` after 0, 2 and 10 merges:
   `['l','o','w','e','s','t','</w>']`, then `['low','e','s','t','</w>']`, then
   `['low','est</w>']`. Sketch the curve of "tokens per word" against merges.
   Where would it flatten out, and why?
3. Feed it a word from another language: `encode_word("zebra", merges)` gives
   `['z','e','b','r','a','</w>']`, every character separate. The tokenizer did
   not fail, so what did it actually do, and what does that cost at run time?
4. Compare `newest` and `lower`: one encodes to a single symbol
   `['newest</w>']`, the other to three. Both are real English words. What
   decided the difference, and is that a property of the language or of the
   corpus?
