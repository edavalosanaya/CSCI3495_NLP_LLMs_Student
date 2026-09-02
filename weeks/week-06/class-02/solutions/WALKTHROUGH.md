# W6C2 Walkthrough: Masked LM and fine-tuning, step by step

Instructor reference and student rescue hatch. **Read only the step you are
stuck on.**

The complete file is `bert_mlm.py` in this folder. Every code block below is
taken from it, and every printed value was produced by running it.

---

## Orientation

Eight training sentences, two test sentences. That ratio is the honest framing
for everything below: this session demonstrates *mechanisms*, and any accuracy
number it produces is a smoke test rather than a measurement.

---

## Step 1, `top_mask_predictions`

```python
    from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

    tok = BertTokenizerFast.from_pretrained(MLM_MODEL)
    model = BertForMaskedLM.from_pretrained(MLM_MODEL)
    fill = pipeline("fill-mask", model=model, tokenizer=tok)
    results = fill(sentence_with_mask, top_k=k)
    return [r["token_str"].strip() for r in results]
```

**The explicit tokenizer/model construction is a workaround, and the code comment
says why:**

> bert-tiny ships only vocab.txt (no tokenizer.json) and no `model_type` in its
> config, so the Auto* dispatch that `pipeline(model=...)` uses fails under
> transformers >= 5.

Passing `model="prajjwal1/bert-tiny"` as a string is the documented one-liner and
it will not work here. A student following a blog post will hit this; point at
the comment rather than letting them conclude their environment is broken.

**`BertForMaskedLM`, not `BertModel`.** The MLM head is the vocabulary-sized
output layer that turns a hidden state into a distribution over word-pieces. The
bare `BertModel` from W6C1 does not have it, which is why *that* file got the
"unexpected keys" warning (it was discarding this head) and this one does not.

**`.strip()`** because the pipeline can return tokens with leading whitespace
depending on the tokenizer; the tests compare exact strings.

**What you should see:**

```python
>>> top_mask_predictions("The capital of France is [MASK].")
['france', 'spain', 'germany', 'algeria', 'canada']
```

**This is the single best teaching output in the session, precisely because it is
wrong.** The correct answer is Paris. The model returns five countries, with
*france* first.

What that tells you, and what to draw out of the class:

1. **It learned the shape of the sentence, not the fact.** The blank sits where a
   proper noun belongs, in a sentence about France, so it proposes salient
   country tokens. Distributional structure is cheap to learn; facts are
   expensive.
2. **It is copying a nearby token.** "france" appears in the input. Small models
   lean heavily on local repetition, which is a failure mode worth naming before
   Week 9's hallucination material.
3. **4M parameters is genuinely tiny.** `bert-base` (110M) answers *paris* here.
   The gap is a concrete argument for scale that students can verify in the
   stretch goal.

**Why the bidirectionality matters.** Ask the class to try
`"The [MASK] of France is Paris."` The model conditions on tokens *after* the
blank, which a causal model structurally cannot do. That difference is exactly
what separates BERT-style encoders from GPT-style decoders, and it is why masked
LM was invented rather than just using a left-to-right LM.

**Running MASK roulette.** The scoring rewards stumping your partner, which
pushes students toward right-context-only sentences and rare words, both of which
expose the model's reliance on local statistics. If a pair is stuck, suggest they
try a sentence where the answer is obvious to a human but requires world
knowledge.

---

## Step 2, `finetune_and_eval`

```python
    torch.manual_seed(seed)

    tok = BertTokenizerFast.from_pretrained(MLM_MODEL)
    model = BertForSequenceClassification.from_pretrained(MLM_MODEL, num_labels=2)
    model.train()

    texts = [t for t, _ in TRAIN_DATA]
    labels = torch.tensor([y for _, y in TRAIN_DATA])
    enc = tok(texts, padding=True, truncation=True, return_tensors="pt")

    opt = torch.optim.AdamW(model.parameters(), lr=5e-4)
    for _ in range(epochs):
        opt.zero_grad()
        out = model(**enc, labels=labels)
        out.loss.backward()
        opt.step()

    model.eval()
```

**The "MISSING" warning is the lesson, not a problem.** Loading prints:

```
classifier.weight                          | MISSING
classifier.bias                            | MISSING
```

The pretrained checkpoint has no 2-class head, so transformers creates one with
random weights. **Everything else transfers.** That is Devlin et al. Fig. 1's
right-hand side stated as a log line: pretrain once, then swap a tiny output
layer per task. Point at it explicitly, because students read "MISSING" as an
error.

**It is the same five-line loop from W4C1**, just with the loss computed inside
the model. Passing `labels=` makes `BertForSequenceClassification` return
`out.loss` directly, so there is no explicit `CrossEntropyLoss` object. Worth
connecting: the universal training loop has not changed since Week 4, only the
model got bigger and pretrained.

**Whole-batch training, no DataLoader.** Eight sentences fit in one batch, so each
"epoch" is a single gradient step. Fine at this scale, and it keeps the loop
readable; a real fine-tune would shuffle and batch.

**`padding=True`** pads to the longest sentence in the batch so the tensors are
rectangular. The attention mask that the tokenizer also returns (and that
`**enc` passes through) is what stops the model attending to the padding.

**`lr=5e-4` is high for fine-tuning** (2e-5 to 5e-5 is typical). It is turned up
because there are only eight steps; with a normal learning rate this would not
move. Another artifact of the toy scale, worth flagging so nobody copies the
hyperparameter into a real project.

**What you should see:**

```
Fine-tuned tiny BERT test accuracy: 1.00
```

---

## Running it

```
...                                                                      [100%]
3 passed
```

**Handle the 1.00 accuracy carefully.** The test set has **two sentences**, so
the achievable scores are 0.00, 0.50, and 1.00. A perfect score means the model
got two easy, in-vocabulary examples right. The test is deliberately named
`beats_chance`, not `is_accurate`.

**The result that actually matters** is not the accuracy, it is how little it
took: eight sentences, eight gradient steps, a few seconds of CPU, and a working
sentiment classifier. That is only possible because the encoder already knew
English. This is the payoff of the whole pretrain-then-finetune paradigm the week
has been building toward.

**The demonstration that makes it land** (first stretch goal, worth doing live if
there is time): build the same architecture from a fresh `BertConfig` instead of
`from_pretrained`, so the weights are random, and run the identical loop. It does
not learn. Same architecture, same data, same optimizer, and the only difference
is whether the encoder was pretrained. Students who see both runs stop thinking
of pretraining as a preliminary step and start thinking of it as where the
knowledge is.

**Connecting to the rest of the course.** This is the last week where fine-tuning
means "update every parameter". W9C1 asks what to do when the model is 7B
parameters and full fine-tuning no longer fits on the machine, which is where
LoRA and quantization come in. The contrast is much sharper if students remember
that here they updated all 4M parameters without thinking about it.
