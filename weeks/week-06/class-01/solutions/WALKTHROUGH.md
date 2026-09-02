# W6C1 Walkthrough: Static vs contextual embeddings, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `contextual_embeddings.py` in this folder. Every code block
below is taken from it, and every printed value was produced by running it with
`prajjwal1/bert-tiny`.

---

## Orientation

`load_model()` ships written, and its comment explains a real packaging wrinkle:

```python
        from transformers import BertModel, BertTokenizerFast
```

`prajjwal1/bert-tiny` ships only a `vocab.txt` (no `tokenizer.json`) and a config
without `model_type`, so the `Auto*` classes cannot dispatch under transformers
5.x. Naming the concrete `Bert*` classes sidesteps it. Students who go looking
for `AutoModel` in the docs and try to "modernize" this will break it; the
comment is there to stop that.

**The load warning is expected.** The checkpoint carries masked-LM head weights
that a bare `BertModel` has no home for, so transformers reports unexpected keys.
Nothing is wrong. Say so before a student spends ten minutes on it.

**What the tokenizer does**, which everything else depends on:

```python
>>> tok("I sat by the river bank.")["input_ids"]
[101, 1045, 2938, 2011, 1996, 2314, 2924, 1012, 102]
>>> tok("bank", add_special_tokens=False)["input_ids"]
[2924]
```

101 and 102 are `[CLS]` and `[SEP]`. "bank" is id 2924, sitting at index 6.

---

## Given, `cosine_similarity`

```python
def cosine_similarity(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return dot / (nu * nv)
```

Third time students have written this (W3C1 on sparse dicts, W3C2 on numpy
arrays, now on plain lists). Worth naming that repetition out loud: the same
similarity measure keeps reappearing because "compare two vectors by angle" is
the operation the whole field runs on.

```python
>>> cosine_similarity([1.0, 0.0], [1.0, 0.0])
1.0
>>> cosine_similarity([1.0, 1.0], [3.0, 3.0])
1.0
```

---

## Step 1, `contextual_vector`

```python
def contextual_vector(sentence: str, word: str):
    import torch

    tok, model = load_model()
    enc = tok(sentence, return_tensors="pt")
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    last = out.hidden_states[-1][0]  # (seq_len, hidden)

    ids = enc["input_ids"][0].tolist()
    word_ids = _word_token_ids(tok, word)
    positions = _find_positions(ids, word_ids)
    if not positions:
        # Fallback: average everything except special tokens.
        positions = list(range(1, len(ids) - 1))
    vec = last[positions].mean(dim=0)
    return vec.tolist()
```

**The whole sentence goes through the model**, and only then do we pick out the
word's position. That ordering *is* the definition of contextual: the vector at
position 6 was computed with every other token visible (BERT is bidirectional, no
causal mask), so it encodes "bank, in this sentence".

**Locating the word** is the fiddly part, and it is why two helpers exist:

```python
def _word_token_ids(tok, word: str) -> list[int]:
    """Sub-token ids for `word` alone, without special tokens."""
    return tok(word, add_special_tokens=False)["input_ids"]


def _find_positions(haystack: list[int], needle: list[int]) -> list[int]:
    """Return the indices in `haystack` covered by the first match of `needle`."""
```

Tokenize the word by itself, then search for that run of ids in the sentence.
This handles multi-word-piece words (a rarer word might split into three pieces)
and words anywhere in the sentence. `add_special_tokens=False` is essential: with
the default, the "word" would come back wrapped in `[CLS]`/`[SEP]` and would
never match.

**`hidden_states[-1][0]`**: `-1` is the last layer, `[0]` is the first (and only)
item in the batch. Note that `hidden_states[0]` is the *embedding* layer output,
not layer 1, which matters for the stretch goal comparing layers.

**`torch.no_grad()`** because nothing here is being trained; it saves memory and
time.

**The fallback** (average all non-special tokens when the word is not found)
keeps the function from returning `None` and producing a confusing crash later.
It is a defensive choice, not a correct one; if a student's word never matches,
the fallback silently returns a sentence-average and they will chase a wrong
number. Worth mentioning as a debugging trap.

**What you should see:**

```python
>>> a = contextual_vector("I sat by the river bank and watched the water.", "bank")
>>> b = contextual_vector("I deposited my paycheck at the bank downtown.", "bank")
>>> len(a)
128
>>> round(cosine_similarity(a, b), 3)
0.809
```

128 is `bert-tiny`'s hidden size. Two different vectors for the same string, which
is the result the whole week is built on.

---

## Step 2, `static_vector`

```python
def static_vector(word: str):
    import torch

    tok, model = load_model()
    emb = model.get_input_embeddings()  # nn.Embedding: id -> vector
    word_ids = _word_token_ids(tok, word)
    with torch.no_grad():
        vecs = emb(torch.tensor(word_ids))
    return vecs.mean(dim=0).tolist()
```

**Look at the signature: there is no sentence parameter.** That is the entire
concept. `get_input_embeddings()` returns the lookup table that maps a token id
to a vector *before any layer runs*, so it is context-free by construction. It is
the same kind of object as the word2vec table from Week 3, living inside a
contextual model as its input layer.

```python
>>> round(cosine_similarity(static_vector("bank"), static_vector("bank")), 3)
1.0
```

Exactly 1.0, necessarily. The same table lookup twice returns the same row, and
any vector has cosine 1 with itself. If a student's answer is not 1.0, they are
accidentally reading a hidden state rather than the embedding table.

---

## Running it

```
Contextual cosine('bank' river vs. money): 0.809
Static     cosine('bank' river vs. money): 1.000
```

**The two numbers are the week's thesis.** Static: 1.000, the model literally
cannot distinguish the senses, because the representation never saw the sentence.
Contextual: 0.809, meaningfully lower, because it did.

**Now the caveat, which matters more than the headline.** Do not let students
read 0.809 as "the model understands that these are different meanings". Two
honest qualifications:

1. **0.809 is still high.** The vectors are more alike than different. A larger
   model separates the senses further, which is a good stretch-goal experiment
   (`bert-base-uncased` typically lands lower).
2. **`bert-tiny` is 2 layers and about 4M parameters**, chosen so this runs on a
   laptop in seconds, not because it is good. The demonstration is that the
   representation is *a function of the sentence*; how well it separates senses is
   a separate question about model quality.

**Connecting back and forward.** W3C2 ended on exactly this gap: static
embeddings give `bank` one vector regardless of context. This session closes it
and, with the ELMo layer finding from lecture (lower layers syntax, higher layers
sense), sets up why the pretrain-then-finetune paradigm took over in 2018.

**Best live follow-up if the jigsaw finishes early:** compare
`hidden_states[1]` against `hidden_states[-1]`. The contextual cosine should be
*higher* at the lower layer, since less contextualization has happened. That is
ELMo's Table 4 result reproduced in three lines on a student's laptop.
