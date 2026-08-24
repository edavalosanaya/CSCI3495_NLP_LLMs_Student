# W4C1 Lab: Build an MLP Text Classifier in PyTorch

**Goal:** train a small **multi-layer perceptron** to classify short reviews as
positive/negative, using **averaged word embeddings** as input. You will see the
universal training loop (zero_grad, forward, loss, backward, step) end to end.

Everything runs **CPU-only** in a few seconds; the dataset and model are tiny on
purpose.

> **In-class warm-up first (pairs, whiteboard, 10 min).** Before any code,
> hand-draw a 2-layer MLP (2 inputs, a 2-neuron hidden layer, 1 output), label
> every weight and bias, then do **one forward pass** and **one gradient step** by
> hand on a tiny toy input. Once you have done the math by hand, Step 3 below is
> just that math in PyTorch.

## Before you code: the picture and the math

The whole pipeline you are about to build, from a review to one vector to a prediction (this is the deep-averaging slide from lecture):

![tokenize, look up embeddings, average into one vector, classify](../lecture/visuals/deep-averaging.png)

And the training loop you will fill in, exactly as shown in lecture:

![the five-line PyTorch training loop: zero_grad, forward, loss, backward, step](../lecture/visuals/training-loop.png)

In slide notation, `embed_document` computes the averaged input vector, and `MLP.forward` computes the next two lines (here $\sigma$ = ReLU):

$$x = \frac{1}{n}\sum_{i=1}^{n} E[w_i] \qquad h = \sigma(Wx + b) \qquad z = Uh$$

Training minimizes cross-entropy on the logits, $L = -\sum_i y_i \log \hat{y}_i$ with $\hat{y} = \mathrm{softmax}(z)$, by gradient descent: $\theta \leftarrow \theta - \eta\,\nabla L$.

In plain words: the finished code squashes each review into ONE $d$-dimensional vector by averaging its word embeddings, pushes that vector through `Linear → ReLU → Linear` to get 2 logits (negative vs positive), and loops zero_grad, forward, loss, backward, step until the loss stops falling. Word order is thrown away by the averaging; only the "average meaning" survives. **Check yourself before coding:** if embeddings have dimension $d$ and a batch holds $B$ documents, what is the shape of the tensor entering `MLP.forward`? ($B \times d$: averaging collapses each document's many token vectors into a single $d$-dimensional vector.)

## How this lab works

Each step tells you **what to write**, then exactly **how to check it**. The
steps are strictly sequential here: Step 3 trains the model you build in Steps 1
and 2.

Set a shortcut for the long docker command first:

```bash
alias lab='docker compose -f docker/docker-compose.yml run --rm --no-deps course'
```

Check **one step**:

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step1 -q
```

Check **everything**:

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -q
```

Stuck for more than a few minutes? Open the walkthrough released after class at the
matching step.

---

### Step 0, Orientation (nothing to write)

`build_vocab` is already written. Read it, then confirm it behaves:

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step0 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

```python
>>> import sys; sys.path.insert(0, "weeks/week-04/class-01/exercise")
>>> from mlp_classifier import TRAIN, build_vocab, tokenize
>>> vocab = build_vocab([t for t, _ in TRAIN])
>>> len(vocab)
25
>>> vocab["<unk>"]
0
```

**Notice:** id 0 is reserved for `<unk>`, and `embed_document` maps any unknown
token to it with `vocab.get(tok, 0)`. Twenty-five words is the entire world this
model will ever know.

---

### Step 1, Average the embeddings

**Write:** the last line of `embed_document`. `vectors` already holds the
embedding rows for this document, shape `(num_tokens, embedding_dim)`. Collapse
it to a single vector by averaging **over the token dimension**.

One call: `vectors.mean(dim=0)`.

`dim=0` is the axis to get right. Averaging over `dim=1` would average each
token's 16 numbers into one scalar, giving you a vector of length `num_tokens`,
which is a different (and useless) thing that will not error until later.

**Done when:**

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step1 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

**Check it by hand:**

```python
>>> import torch, torch.nn as nn
>>> torch.manual_seed(0)
>>> emb = nn.Embedding(len(vocab), 16)
>>> d = embed_document("i loved this movie it was great", vocab, emb)
>>> tuple(d.shape)
(16,)
```

Seven tokens went in, one 16-dimensional vector came out, regardless of how long
the review was.

**Why it matters:** this is the whole "deep averaging" idea, and its cost. A
fixed-size vector means the classifier can take any-length input, but averaging
destroys word order: "good not bad" and "bad not good" produce the identical
vector. Hold that thought until Week 5.

---

### Step 2, The forward pass

**Write:** `MLP.forward`. Apply `fc1`, then the ReLU in `self.act`, then `fc2`,
and return the result.

Return the **logits**, raw scores, not probabilities. Do not add a softmax:
`nn.CrossEntropyLoss` applies its own internally, and adding a second one is a
classic bug that quietly slows training to a crawl.

**Done when:**

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step2 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

**Check it by hand:**

```python
>>> model = MLP(in_dim=16, hidden=8)
>>> tuple(model(torch.stack([d, d])).shape)
(2, 2)
```

Two documents in, two logits each (negative and positive).

**Why it matters:** `Linear → ReLU → Linear` is the smallest network that is not
just linear regression. Remove the ReLU and the two Linears collapse into a
single matrix, so the model could only ever draw a straight line. The
nonlinearity is what makes it a *deep* network.

---

### Step 3, The training loop

**Write:** the five lines inside `train`'s loop, in this order:

```
optimizer.zero_grad()
logits = model(X)
loss = loss_fn(logits, y)
loss.backward()
optimizer.step()
```

Leave `history.append(float(loss))` at the end; it is already there.

**The order is not negotiable.** Forgetting `zero_grad()` makes PyTorch
*accumulate* gradients across epochs, so your updates grow without bound. Calling
`optimizer.step()` before `loss.backward()` steps on gradients that do not exist
yet.

**Done when:**

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -k step3 -q
```

```
.                                                                        [100%]
1 passed, 3 deselected
```

**Why it matters:** this exact five-line loop trains everything else in this
course, including the Transformer in Week 5 and the LoRA fine-tune in Week 9.
Only the model and the data get bigger.

---

### Step 4, Run the whole thing

```bash
lab python weeks/week-04/class-01/exercise/mlp_classifier.py
```

```
vocab size: 25
epoch   1 loss: 0.7414
epoch 200 loss: 0.0000
train accuracy: 1.00
```

And the full suite:

```bash
lab python -m pytest weeks/week-04/class-01/exercise/test_mlp_classifier.py -q
```

```
....                                                                     [100%]
4 passed
```

**A loss of 0.0000 and 100% accuracy is not a success story.** Eight training
documents, no held-out set, and a model with enough parameters to memorize all of
them. The model has not learned sentiment, it has learned these eight sentences.
That is the honest reading, and it is why every later week insists on a held-out
split.

Notice also where the loss started: **0.7414**, very close to `ln(2) = 0.693`,
which is exactly the cross-entropy of a model guessing 50/50 between two classes.
Untrained networks start at chance, and that number is a useful sanity check any
time you train a binary classifier.

## Stretch goals

- Add **dropout** between the layers. Does it change anything on this tiny set?
- Add a held-out review and print the predicted probabilities (apply `softmax` to
  the logits yourself, since the model does not).
- Feed the model "good not bad" and "bad not good" and confirm they get identical
  predictions. That is the averaging limitation, made concrete.

## Why this matters

This is the smallest complete neural NLP classifier. The same loop trains
Transformers later in the course: only the model and data get bigger.

A full reference solution is in the reference solution released after class, and the
step-by-step explanation is in the walkthrough released after class (don't peek until
you've tried).
