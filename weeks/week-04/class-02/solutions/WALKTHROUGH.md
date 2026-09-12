# W4C2 Walkthrough: from reviews to a trained model, step by step

The horror version, complete, so you have the method while you write the comedy
one yourself. **Read only the step you are stuck on.**

The complete file is `horror_scariness.py` in this folder, and every code block
below is copied from it. Every printed number was produced by running it on the
shipped `horror_reviews.csv`: 120 films, three word lists, 200 steps.

Your job in the lab is the same seven steps on `comedy_reviews.csv`, predicting
`funniness` from `funny.txt`, `crude.txt` and `flat.txt`. Do not copy this file
across. Read a step, close it, and type the comedy version yourself.

---

## Step 1, Load the films

```python
df = pd.read_csv(DATA / "horror_reviews.csv")
```

One row per film. The two columns that matter sit at opposite ends of the
problem: `reviews` is a string, `scariness` is a number out of ten. Nothing in
PyTorch can multiply a string, so the whole job is crossing that gap.

`df.shape` is `(120, 5)`.

---

## Step 2, Read the rule in

```python
{name: (DATA / f"{name}.txt").read_text().split()
 for name in ("fear", "gore", "dull")}
```

The three lists are the rule, and a person wrote them. `.read().split()` on a
file of one word per line gives a list of words; that is the whole of the
parsing.

`8 5 5` words. Open `fear.txt` and argue with it: there is a word missing, and
probably one that should not be there.

---

## Step 3, Count the words

```python
low = df["reviews"].str.lower()
for name, words in lists.items():
    df[name] = sum(low.str.count(word) for word in words)
```

This is **rule-based feature extraction**, and it is what people built before
anything was learned from data. Note what the loop is over: three *categories*,
not 120 films. `.str.count` already reaches every row at once.

Note also what it throws away, which is nearly everything: word order, who said
it, and negation. "Not remotely terrifying" counts as one fear word.

The first film, Red House, comes out as `1  0  0`.

---

## Step 4, Make them tensors

```python
X = torch.tensor(df[features].values, dtype=torch.float32)
y = torch.tensor(df[target].values, dtype=torch.float32)
X = X - X.mean(0)
```

`float32` always. The error you get from the wrong dtype never contains the word
"dtype", which is why it costs everyone an afternoon once.

`X.shape` is `(120, 3)`: 120 films with three features each.

**The last line is the one people skip.** Centring subtracts each column's own
average so the average film sits at zero. Without it the fear weight lands in
about fifty steps and the bias is still crawling thousands of steps later,
because the bias has a much smaller gradient than a feature that averages 1.7.

---

## Step 5, The model is four numbers

```python
w = torch.zeros(X.shape[1], requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

No class, no `nn.Linear`, no framework. One number per word list, plus a height.

`requires_grad` is the switch: it tells PyTorch to record every operation these
take part in, and that recording is what makes them trainable. A tensor without
it is just an array.

---

## Step 6, One loss, one backward

```python
loss = ((X @ w + b - y) ** 2).mean()
loss.backward()
```

Three beats: **predict, score, blame.** `X @ w + b` predicts all 120 films at
once, `- y` gives 120 errors, `** 2` makes a big miss hurt far more than a small
one, and `.mean()` collapses the lot to a single number.

Then `backward()` walks that one number back to everything with `requires_grad`
on and leaves a slope in `.grad`.

From all zeros: loss `32.041`, `w.grad` `[-4.113, -0.392, 1.125]`, `b.grad`
`-10.733`. All three gradients are negative except the dull one. Work out what
that says about which way each number has to move before you read on.

**Nothing has moved yet.** `backward()` fills in `.grad` and updates nothing.

---

## Step 7, Measure, blame, step, clear

```python
for _ in range(steps):
    loss = ((X @ w + b - y) ** 2).mean()      # measure
    loss.backward()                           # blame
    with torch.no_grad():                     # step
        w -= lr * w.grad
        b -= lr * b.grad
    w.grad.zero_()                            # clear
    b.grad.zero_()
```

`no_grad` because moving a parameter is not part of the model and must not be
recorded. `zero_()` because gradients **accumulate by default**, so a loop
without it trains on a running total. Everyone writes that bug exactly once.

After 200 steps at `lr = 0.05`:

| | weight |
|---|---|
| fear words | **+1.283** |
| gore words | **+0.334** |
| dull words | **-0.587** |
| bias | +5.367 |

Read them as three sentences. A fear word is worth about a point and a quarter
of scariness. A gore word is worth a quarter of that, because blood is not fear,
and nobody told the model that. A dull word takes scariness *away*, and that
negative weight is the clearest evidence the model is reading words rather than
just counting them.

---

## The check that matters

```python
float(((y - y.mean()) ** 2).mean())
```

Loss `0.2055`, against **`3.2402`** for ignoring the reviews entirely and
guessing the average every time.

Always compute that second number. A model always returns an answer, and a
weight always looks like a finding. The do-nothing baseline is how you ask
whether either is worth anything.

---

## Doing it on the comedies

Same seven steps, and two things will be different:

1. **The loss will be much higher**, and that is the data, not your code. The
   comedy ratings are noisier because comedy is more divisive. Judge the model
   against its own baseline, never against the horror number.
2. **The weights will be slightly less exact.** Noise costs precision, not
   correctness, which is the honest reason to trust the method at all.
