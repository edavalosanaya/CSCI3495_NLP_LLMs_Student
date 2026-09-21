# W5C1: The network reads the review

**There is no code in this folder, and the cells in `lab.ipynb` are empty.** We
write the whole thing together: it goes up on screen, you type it, and by the end
you have trained two neural networks on raw text. Nothing is pre-written, so
nothing is skipped past.

## The setup

`data/horror_reviews.csv` holds 1200 horror films, each with a written review and
a `scariness` score out of ten. The job is to predict the score from the review.

Last class you did something like this by hand: somebody gave you three word
lists, you counted how often those words appeared, and you fitted four numbers.

**This time nobody gives you a word list.** The network gets the review itself.
Whatever is worth noticing in the text, it has to find.

## Getting started

Once, from the repository root:

```
uv sync
```

Then, from this folder:

```
uv run jupyter lab
```

Open `lab.ipynb` and start typing in the first cell.

## What we build, in eight stages

1. Load the reviews.
2. Give every distinct word an integer id.
3. Pack the reviews into one rectangle of integers, padded on the **left**.
   Then **look at the data**: what we are predicting, how long reviews are, and
   what the single word *terrifying* does to a score depending on whether there
   is a "not" in front of it.
4. Hold 200 films back as a test set, and work out the score you get by guessing.
5. **Model one:** look up a vector per word, average them, predict a number.
6. Write the training loop once. Train model one.
7. **Model two:** one line different. An RNN reads the words in order first.
   Then **plot both training curves** side by side, which shows something the
   two final numbers do not.
8. Ask *why* model two wins. The answer is the point of the day.

## The numbers to expect

| model | test MSE |
|---|---|
| guess the average every time | 1.972 |
| average the word embeddings | 1.416 |
| read the words in order with an RNN | 0.635 |

Lower is better. Training both models takes about twelve seconds in total on a
laptop; you do not need a GPU and you do not need to leave it running.

If your numbers are a little different, that is fine and expected. If your RNN is
not clearly better than your bag of embeddings, something is wrong, and the most
likely culprit is stage 3.

## After the break: the project proposal

**No second dataset.** The rest of the period is the first working session on
the semester project, and the aim is that nobody leaves without a filled-in
skeleton.

Open `project/proposal/CSCI3495_Project_Proposal_Template.docx`. It is a form:
click into a box and type. It is **one page**, and it is due **Wednesday
October 7 (Week 7, Class 2)**, submission only, no pitch.

### What to get done in the room

- **Teams.** Solo, two, or three. Scope scales with size: solo is a baseline
  plus one comparison, two adds a second axis, three is roughly two to three
  times a solo project.
- **Section 1, the question.** Write it as an actual question, with a question
  mark. If it does not fit in one sentence, keep working on it; that is the
  session doing its job.
- **Section 3, the plan.** The hard part is **evaluation**. "We'll see if it
  works" is not a metric. Name the number, and name the baseline you compare it
  against. Today's lab is the shape of it: 1.972 was the baseline, 1.416 and
  0.635 were the numbers, and none of the three means anything on its own.
- **Section 5, responsibilities.** Every member named against a specific
  deliverable, not a role. The confidential peer evaluation at the end of the
  semester is checked against this table, so it is worth ten honest minutes now.

Ask the instructor to look at your section 3 before you leave. The most common
problem at this stage is a project with no way to tell whether it worked.

### Optional, if you want the repetition

`data/comedy_reviews.csv` is 1200 films with a `funniness` score, built exactly
the same way. Redoing all eight stages on it from an empty cell changes two
things: the filename and the column name.

Not assigned and not collected. If you do it, expect every number a little
worse (guessing **2.026**, the bag model **1.518**, the RNN **0.745**), because
comedy ratings are noisier. The ordering of the three models does not change,
which is the part that matters.

If you get stuck on a stage, open that one stage in
`../solutions/instructor-lab.ipynb`, read it, close it, and type it yourself.

## Nothing to submit today

The lab is not collected. The proposal is, on October 7.
