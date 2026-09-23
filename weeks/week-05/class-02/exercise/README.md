# W5C2: Flip the translator

Before the break you built an English-to-Spanish translator by typing it out
together. Now you turn it around and use it to read Spanish.

## The task

Train the same model to go Spanish to English, then translate five phrases.

You do not need a new model, a new loop, or a new idea. The encoder does not
know or care which language it is reading. Two columns change.

Work in your team. Copy the cells you need into new ones and edit them.

## Step 1. Swap the direction

Everywhere English is the input and Spanish the output, make it the other way
round. Two places matter:

- the vocabularies: which column builds `src_stoi` and which builds `tgt_stoi`
- the tensors: which column fills `X` and which fills `Y`

`SRC_LEN` and `TGT_LEN` swap with them.

## Step 2. Check it works

Retrain, then run stage 8 again. It should print Spanish going in and English
coming out, and get roughly one of the five exactly right.

This direction is the easier one. English has one article where Spanish has
four, no gender on adjectives, and one verb form where Spanish has five. Going
this way the model throws information away; going the other way it has to
invent it.

## Step 3. Read the five phrases

Translate these. All five are held-out sentences your model has never trained
on.

```
1.  ella me hace feliz
2.  no puedo hacer nada
3.  hace mucho frío
4.  llama a tu hermano
5.  él empezó a cantar
```

Write down the five English sentences. That is the deliverable.

Two of them are worth a second look:

- one is **idiomatic**: a word-for-word reading gives you nonsense, and Spanish
  says it a completely different way from English
- one contains a word that English has no equivalent for at all

Say which, and why.

## Hand in at the end of the period

One message per team:

```
team:        names
phrases:     1. ...   2. ...   3. ...   4. ...   5. ...
the odd ones: <which two, and what is going on>
```

## If you get stuck

- **Shapes disagree.** `SRC_LEN` and `TGT_LEN` are the other way round now.
- **`KeyError`.** A word is being looked up in the other language's vocabulary.
  Check which column built which `stoi`.
- **Output is Spanish.** The tensors swapped but `tgt_itos` did not.
- **Everything ends immediately.** The `y_in` / `y_out` shift went missing.

## Going further

- Try your own sentence. Anything outside the 964-word English vocabulary comes
  back as `<no id for: ...>`.
- Score yourself. Write a loop that translates every test row and counts how
  many land in `accepted_en`. Expect about 0.45.
- Count how many of your wrong answers are actually correct English that simply
  differs from the stored translation. That number is why translation is not
  scored with exact match.
