# W5C2: Build a translator

An English-to-Spanish translator, built together in class.

## Files

| file | what |
|---|---|
| `lab.ipynb` | Mostly written already. **Stages 4 and 7 are empty**: we type those two together. |
| `data/eng-spa.tsv` | 10,949 real sentence pairs, split into train, develop and test. |
| `data/DATA-SOURCE.md` | Where the data comes from, and its licence. |

## Getting started

Once, from the repository root: `uv sync`. Then from this folder:

```
uv run jupyter lab
```

Open `lab.ipynb` and run the cells from the top. Stages 4 and 7 are blank and
will fail until we write them.

## The two stages you write

**Stage 4, the model.** An encoder LSTM reads the English and keeps its final
state. A decoder LSTM starts from that state and writes Spanish. That shared
state is the only connection between the two halves.

**Stage 7, greedy decoding.** Generating is not the training loop. There is no
correct answer to feed in, so the decoder runs one step at a time, and each
step's output becomes the next step's input.

## The data

Real sentences from Tatoeba, written and translated by volunteers. Four tokens
at most on each side, drawn from the most frequent 964 English and 1496 Spanish
words.

Two columns hold the right answers. A sentence often has several correct
translations, so `accepted_es` lists every Spanish translation of an English
row and `accepted_en` every English translation of a Spanish one. Scoring with
plain `==` would mark correct translations wrong.

## What to expect

About one held-out sentence in three comes back exactly right. Many of the
other two thirds are not wrong so much as different: `okay lets go` becomes
`pues vamonos` where the stored answer is `vale vamonos`. Both are Spanish
people say.

That is a real result for a model this size trained in thirty seconds, and it
is why machine translation is not scored by exact match.

## After the break: the semester project

We stop working on the translator and turn to the project. Bring up
`project/proposal/CSCI3495_Project_Proposal_Template.docx` and keep filling it
in with your team.

The proposal is due **Wednesday October 7**, one page, submission only.

The two things worth having before you leave:

- **A question with a question mark in it.** If it does not fit in one
  sentence, it is not a project yet.
- **A metric and a baseline.** "We will see if it works" is not a metric. Name
  the number and name what you compare it against.

## If you want more of the translator

Not assigned, and nothing to hand in:

- Flip it. Train Spanish to English instead, which is two columns changed, and
  score against `accepted_en`. It comes out better than this direction; work
  out why.
- Translate a sentence of your own. Anything outside the 964-word English
  vocabulary comes back as `<no id for: ...>`.
