# W2C1 Lab: The N-gram Bard

## 1. Learning objective

Turn a corpus into an n-gram language model, use it to babble new sentences,
and score how surprised it is by text it has never seen.

You write two functions in `ngram_lm.py`: the smoothed probability, and
perplexity. The counting, padding and sampling are already written for you.

## 2. Understanding the math

![Worked bigram example: three padded sentences and the probabilities you get by counting](../lecture/visuals/worked-example.png)

A bigram model estimates the next word from the one before it, by counting:

$$P(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t)}{\text{count}(w_{t-1})}$$

![Add-one smoothing: pretend every n-gram was seen one extra time](../lecture/visuals/add-one.png)

Any pair the corpus never happened to contain would get probability 0, and one
zero wipes out a whole sentence. Add-one smoothing pretends every n-gram was
seen once more than it was, with $V$ the vocabulary size:

$$P_{\text{lap}}(w_t \mid w_{t-1}) = \frac{\text{count}(w_{t-1}, w_t) + 1}{\text{count}(w_{t-1}) + V}$$

Perplexity is the exponentiated average negative log-probability over $N$
predicted tokens. Read it as "how many words the model was effectively choosing
between at each step", so lower is better:

$$\text{PPL} = \exp\Big(-\frac{1}{N}\sum_{t=1}^{N} \log P_{\text{lap}}(w_t \mid \text{context}_t)\Big)$$

## 3. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-02/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 4. Implement `prob`

Return the smoothed $P(\text{word} \mid \text{context})$. One line of
arithmetic. Both counts live in `model`, and a tuple that was never seen is
missing from those dicts rather than stored as zero.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 4 deselected
```

## 5. Implement `perplexity`

Sum the log of `prob` over every prediction, then exponentiate minus the
average. `iter_predictions(n, sentences)` hands you the padded
`(context, word)` pairs, so there is no windowing to do.

```bash
pytest -k step2 -q
```

```
..                                                                       [100%]
2 passed, 4 deselected
```

## 6. Run it, then break it

```bash
python ngram_lm.py
```

```
============================================================
N-gram Bard, train, babble, and score with perplexity
============================================================

[unigram]  perplexity on held-out =   15.39
   babble: eggs am not

[bigram]  perplexity on held-out =    8.64
   babble: ham am mat

[trigram]  perplexity on held-out =   10.63
   babble: ham and mat a in green a in

Lower perplexity = better.
```

The bigram wins, and the trigram is worse than the bigram. Each experiment
below is a one-line edit; undo it before the next.

1. Keep going up. Add `4` to the `for n in (1, 2, 3)` loop. Perplexity climbs
   again, to 12.52. With ten short sentences, why does a longer context make
   the model worse rather than better?
2. Watch smoothing do its job. Print `prob(model, ("the",), "elephant")` for
   the bigram model. It is 0.0278 even though "elephant" never follows "the",
   and never appears at all. Where did that probability come from, and which
   word paid for it?
3. Score a sentence from another planet. Change `held_out` to
   `["colorless green ideas sleep furiously"]`. Unigram perplexity is 60.30 and
   bigram 27.79, both far worse than on the in-domain sentence. Is the bigram
   really understanding this sentence better than the unigram, or is something
   else going on?
4. Read the babble. The trigram output repeats `a in green a in`. Look at how
   `generate` picks each word, and explain why a higher-order model on this
   corpus loops.
