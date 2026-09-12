# W4C2 Lab: how scary is it, and how funny is it?

**There is no notebook in this folder. Making one is the first thing you do.**

Open Jupyter, create a new notebook here, and call it whatever you like. Then we
write the whole thing together: it goes up on screen, you type it, and by the end
you have built a model that predicts how frightening a film is from what people
wrote about it. Nothing is pre-written, so nothing is skipped past.

After the break you do it again on a different problem, on your own.

## Before the break: the horror films

120 horror films. For each one, a pile of audience reviews and a scariness score
out of ten. **We have the words and we want the number.** Nothing in PyTorch can
multiply a string, so the whole session is about crossing that gap: three
hand-written lists of adjectives turn each film into three counts, and four
trainable numbers turn those counts into a prediction.

What you should see as we go, so you can tell you are on track without anyone
looking at your screen:

| after | you should see |
|---|---|
| loading the csv | `(120, 5)` |
| reading the word lists | `8 5 5` |
| counting the words | the first film, Red House, is `1  0  0` |
| the tensors | `torch.Size([120, 3]) torch.float32` |
| the first `backward()` | loss `32.041` |
| 200 steps of training | fear `+1.283`, gore `+0.334`, dull `-0.587`, loss `0.2055` |

That last row is the whole session in four numbers. A fear word is worth about a
point and a quarter of scariness; a gore word a quarter of that, because blood is
not fear; and a dull word takes scariness away. Nobody told the model any of that.

## After the break: the comedies, on your own

**Your turn, and there is no template.** Same method, new problem:

> **140 comedies, their audience reviews, and how funny people found them out of
> ten. Predict `funniness` from the reviews.**

Everything you need is in `data/`: the corpus, and three word lists a person
wrote. Write all seven steps yourself, in the same notebook you made. You watched
the whole method an hour ago; this is where you find out whether you can do it.

**Expect a worse fit than the horror one, and do not read that as a bug.**
Comedy is more divisive than horror, so the ratings are noisier and no model can
do as well. Judge it against the do-nothing baseline, never against `0.2055`.

> When it is right: roughly funny `+0.98`, crude `+0.26`, flat `-0.80`, bias
> `5.71`, and a loss near **0.803** against **3.135** for guessing the average
> every time. Four times better than doing nothing, and four times worse than the
> horror model. Both of those facts are worth a sentence.

**Stuck?** `../solutions/horror_scariness.py` is the complete horror code, and
`../solutions/WALKTHROUGH.md` takes it one step at a time. Read the one step you
are stuck on, close it, and type the comedy version yourself. Copying the file
across teaches you nothing and takes about as long.

## Getting started

Once, from the repository root on your own machine:

```
uv sync
```

Then, from this folder:

```
uv run jupyter lab
```

Make a new notebook, pick the project's **`.venv`** kernel, and put your hands on
the keyboard.

## What is in this folder

| File | What it is |
|---|---|
| `data/horror_reviews.csv` | 120 horror films: title, year, reviews, scariness. |
| `data/fear.txt` `gore.txt` `dull.txt` | The horror rule: 8, 5 and 5 adjectives. |
| `data/comedy_reviews.csv` | 140 comedies: title, year, reviews, funniness. |
| `data/funny.txt` `crude.txt` `flat.txt` | The comedy rule: 8, 5 and 5 adjectives. |

The word lists are the rule, and a person wrote them. Open one. Argue with it:
find a word that is missing, and one that has no business being there. Changing a
list and retraining is the fastest interesting experiment in this lab.

Nothing to submit.
