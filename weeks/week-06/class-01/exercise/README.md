# W6C1 Lab: Static vs. Contextual Embeddings

## 1. Learning objective

Show that a word's vector changes with the sentence around it. Pull the same
word out of two sentences with a real BERT, and compare against the static
lookup that cannot tell those sentences apart.

You write two functions in `contextual_embeddings.py`. Cosine similarity, the
model loader and the token-locating helpers are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-06/class-01/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `contextual_vector`

![Contextual idea: run a deep LM over the sentence; its hidden states ARE the representations](../lecture/visuals/contextual-idea.png)

A word's vector averages over the $m$ word-pieces it was split into. The
contextual one averages the model's LAST hidden layer $\mathbf{h}^{(L)}$, after
the whole sentence has been read:

$$\text{contextual\_vector}(s, w) = \frac{1}{m}\sum_{i=1}^{m} \mathbf{h}^{(L)}_i$$

Run the sentence through the model, take the last hidden layer, and average
the positions the word occupies.

```bash
pytest -k step1 -q
```

```
.                                                                        [100%]
1 passed, 5 deselected
```

## 4. Implement `static_vector`

![Static vs. contextual embeddings: both senses of "bank" collapse to one static vector, but contextual vectors differ](../lecture/visuals/static-vs-contextual.png)

The static one averages the same $m$ pieces, but from the input embedding table
$E$, which no sentence has touched:

$$\text{static\_vector}(w) = \frac{1}{m}\sum_{i=1}^{m} E[t_i]$$

Both are compared with the usual cosine, which is given:

$$\mathrm{cosine}(\mathbf{u}, \mathbf{v}) = \frac{\mathbf{u} \cdot \mathbf{v}}{\lVert\mathbf{u}\rVert \, \lVert\mathbf{v}\rVert}$$

Look the word's sub-tokens up in the input embedding table and average them.
No sentence, no forward pass.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 5 deselected
```

## 5. Run it, then question it

```bash
python contextual_embeddings.py
```

```
Contextual cosine('bank' river vs. money): 0.809
Static     cosine('bank' river vs. money): 1.000
Expect: static == 1.000 (identical), contextual < static (sense-dependent).
```

The static number is 1.000 by construction: the same word, looked up the same
way, twice.

1. Try a different ambiguous word. Use `bat` in "The bat flew out of the cave
   at dusk." and "He swung the bat and hit a home run.": the contextual cosine
   is 0.860, higher than `bank`'s 0.809. Which of the two words do you think
   this model separates better, and does that match your intuition?
2. Try a word with no senses at all. Run `the` through the two `bank`
   sentences: 0.905, the highest of the three. Why should a function word move
   less than an ambiguous noun, and what would a cosine of exactly 1.0 mean?
3. Read a different layer. Take `out.hidden_states[0]` instead of `[-1]`. The
   cosine becomes 0.818 rather than 0.809, and layer 1 gives 0.901. Layer 0 is
   the embedding layer, so why is it not exactly 1.000 like `static_vector`?
4. This model has 2 layers and 128 hidden dimensions. The `bank` senses are
   only 0.809 apart. What would you expect from a 12-layer BERT, and what
   experiment on this file would settle it?
