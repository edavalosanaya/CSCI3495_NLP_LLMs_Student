# W6C2 Lab: Masked LM & Fine-tuning

## 1. Learning objective

Use BERT two ways: run its pretraining objective directly by filling in a
`[MASK]`, then fine-tune the same weights into a sentiment classifier and watch
eight sentences be enough.

You write two functions in `bert_mlm.py`. The data and the demo are given.

## 2. Getting started

From the repository root on your own machine, once per session:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps -w /workspace/weeks/week-06/class-02/exercise course bash
```

A step you have not written yet reports `skipped`, not a failure. If you get
stuck, `../solutions/WALKTHROUGH.md` works out every step, and these labs are
not graded.

## 3. Implement `top_mask_predictions`

![Masked LM training signal: loss flows only through the selected tokens (SLP3 Fig. 9.3)](../lecture/visuals/assets/slp3-fig-9-3.png)

Masked language modelling scores every vocabulary word at the masked position
from that position's final hidden state, and the top-$k$ are its best guesses:

$$P(w \mid \text{context}) = \mathrm{softmax}\big(\mathbf{W}\,\mathbf{h}^{(L)}_i\big), \qquad \text{top-}k = \operatorname{arg\,top}_k \; P(w \mid \text{context})$$

Build the tokenizer and masked-LM model explicitly, hand them to a `fill-mask`
pipeline, and return the `token_str` of each result, stripped.

```bash
pytest -k step1 -q
```

```
..                                                                       [100%]
2 passed, 1 deselected
```

## 4. Implement `finetune_and_eval`

![BERT pretrain then fine-tune: everything transfers except the tiny output layer (Devlin et al. 2019, Fig. 1)](../lecture/visuals/assets/bert-2019-fig-1.png)

Fine-tuning throws away the masked-LM head, puts a fresh classifier on the
`[CLS]` position, and trains the whole thing on labels. Everything under that
new head is already pretrained, which is why so little data is needed:

$$\hat{\mathbf{y}} = \mathrm{softmax}\big(\mathbf{W}_c\,\mathbf{h}_{\text{[CLS]}} + \mathbf{b}\big), \qquad \mathcal{L} = -\log \hat{y}_{\text{true}}$$

Load a sequence-classification model, train it for a few epochs with
`labels=`, and return test accuracy.

```bash
pytest -k step2 -q
```

```
.                                                                        [100%]
1 passed, 2 deselected
```

## 5. Run it, then question it

```bash
python bert_mlm.py
```

```
Top [MASK] predictions: ['france', 'spain', 'germany', 'algeria', 'canada']
Fine-tuned tiny BERT test accuracy: 1.00
```

Read that first line carefully: asked for the capital of France, the model
answered with five countries and never said Paris.

1. Give it an easier frame. Try `"Paris is the capital of [MASK]."` instead:
   now it answers `['france', 'paris', 'spain', 'madrid', 'brussels']` and gets
   it right. The fact is the same in both sentences. What changed?
2. Probe it for bias. Compare `"The doctor said [MASK] would be late."` with
   `"The nurse said [MASK] would be late."`. The first ranks `he` first, the
   second ranks `she` first. Nothing in either sentence gives the gender away,
   so where did the model get it, and how does this relate to W3C2?
3. Untrain the classifier. Call `finetune_and_eval(epochs=0)`: accuracy is
   0.50, exactly chance. At `epochs=1` it is still 0.50, and by 8 it is 1.00.
   What is the freshly initialized classification head doing at epoch 0, given
   everything below it is already pretrained?
4. That 1.00 is on two test sentences. Write down the smallest change to
   `TEST_DATA` you believe would break it, then make that change and find out
   whether you were right.
