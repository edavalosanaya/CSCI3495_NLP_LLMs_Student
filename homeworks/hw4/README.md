# HW4: Fine-tuning & Adapting LLMs with LoRA / PEFT

**Out:** Week 8, Class 1 · **Due:** Week 9, Class 1 · **100 points** · **Weight:** 2.5% of the course grade
**Estimated time:** 4-6 hours

## Learning goals
By completing this homework you will be able to:
1. Explain **why** full fine-tuning of large models is expensive and how
   **parameter-efficient fine-tuning (PEFT)** addresses it.
2. Implement **LoRA** (Low-Rank Adaptation) from scratch in PyTorch, including
   the correct initialization that makes the adapted model start identical to
   the pretrained one.
3. **Inject** LoRA adapters into an existing model, freeze the backbone, and
   account for the (small) number of trainable parameters.
4. Fine-tune **only** the adapters on a task and verify that the loss
   decreases while the frozen weights stay fixed.
5. Connect your implementation to the **Hugging Face `peft`** library.

## Background: LoRA (Hu et al., 2021)
Reading: **LoRA: Low-Rank Adaptation of Large Language Models**, Hu et al.,
2021, [arXiv:2106.09685](https://arxiv.org/abs/2106.09685). (Week 9 reading.)

Fine-tuning all weights of a large model means storing and updating a full copy
of every parameter, billions of them, which is costly in memory and disk.
LoRA's insight is that the *update* you need for a new task has **low intrinsic
rank**. Instead of learning a full `ΔW ∈ R^{d×k}`, LoRA freezes the pretrained
weight `W0` and learns a low-rank factorization:

```
ΔW = B · A,      A ∈ R^{r×k},   B ∈ R^{d×r},   r ≪ min(d, k)
h  = W0·x + (alpha / r) · (B · A) · x
```

Key details you must reproduce:
- **Initialization:** `A` is a small random Gaussian and `B` is **zero**, so
  `ΔW = 0` at the start, the adapted model is *exactly* the pretrained model on
  step 0, and training only nudges it from there.
- **Scaling:** the update is scaled by `alpha / r`.
- **Frozen backbone:** `W0` (and any bias) has `requires_grad = False`; only
  `A` and `B` are trained. For a 1000×1000 layer with `r = 8`, you train
  `8·(1000+1000) = 16k` numbers instead of `1,000,000`, a ~60× reduction.

## Files

```
hw4/
  lora.py              # <- YOU implement the TODOs here
  tests/test_lora.py   # the tests each step below refers to
  ANSWERS.md           # <- YOU write the short answers here
  README.md            # this handout
```

## How this homework works

This handout is a sequence of steps. Each step is one function, and **each step
ends with a test you can run**, so you always know whether you are done before
you move on. Work them in order: later steps import earlier ones.

From the repository root, inside the course image:

```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw4/tests -q
```

`hw` is a shortcut for the long docker command. Set it up once per
terminal session, using the line for **your** shell:

```
# macOS / Linux (bash, zsh)
alias hw='docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw4/tests -q'

# Windows, PowerShell
function hw { docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw4/tests -q @args }

# Windows, Command Prompt
doskey hw=docker compose -f docker/docker-compose.yml run --rm --no-deps course python -m pytest homeworks/hw4/tests -q $*
```

Then:

```bash
hw -k step3      # check ONLY step 3
hw               # run every step
```

If you already work inside the container (`... run --rm --no-deps course bash`),
drop the docker prefix and just use `python -m pytest homeworks/hw4/tests -q`.

**Before you write anything, every test skips.** That is expected: the suite
detects the unfinished starter and skips rather than drowning you in failures.
The moment step 1 is implemented the tests start running for real.

**Total when you are finished: `11 passed`.**

### Step 0, Orientation (0 pts)

Nothing to write yet.

Read `lora.py` top to bottom. `TinyClassifier` is already written for you: it is a
plain two-layer MLP with `fc1`, `fc2` and `head`, and it stands in for a frozen
pretrained backbone. You are not training a language model here; you are building
the adapter mechanism and proving it does what the paper claims. Then:

```bash
hw
```

You should get `11 skipped`.

### Step 1, `LoRALinear` (30 pts)

**Write** `__init__` and `forward`. Store the base layer, `r`, `alpha` and `scaling = alpha / r`. Create `A` of shape `(r, in_features)` initialized `N(0, 0.02)` and `B` of shape `(out_features, r)` initialized to **zeros**. Freeze `base.weight` and `base.bias`. The forward pass is `base(x) + scaling * (x @ A.T @ B.T)`.

**Done when** `hw -k step1` prints `4 passed, 7 deselected`.

**Check it by hand**

```python
>>> base = nn.Linear(4, 3)
>>> wrapped = LoRALinear(base, r=2, alpha=4.0)
>>> wrapped.scaling
2.0
>>> x = torch.randn(1, 4)
>>> torch.allclose(wrapped(x), base(x))    # B is zero, so the adapter starts invisible
True
>>> base.weight.requires_grad
False
```

**Why it matters.** That `allclose` is the whole design. Because `B` starts at zero the wrapped model is *exactly* the pretrained model at step 0, so fine-tuning starts from the pretrained behavior instead of a random perturbation of it. Initialize both matrices randomly and you have quietly damaged the model before training begins.

### Step 2, `inject_lora` (12 pts)

**Write** `inject_lora(model, target, r, alpha)`: walk the module tree and replace every child **named** `target` that is an `nn.Linear` with a `LoRALinear` wrapping it. Recurse into submodules; the target is often nested.

**Done when** `hw -k step2` prints `2 passed, 9 deselected`.

**Check it by hand**

```python
>>> model = TinyClassifier()
>>> inject_lora(model, target="fc1", r=4, alpha=8.0)
>>> type(model.fc1).__name__
'LoRALinear'
>>> type(model.head).__name__              # untouched
'Linear'
```

**Why it matters.** Real PEFT targets layers by name inside a large model (`q_proj`, `v_proj` and so on), which is why the walk has to recurse rather than look only at top-level attributes.

### Step 3, `count_trainable_parameters` (8 pts)

**Write** the counter: return `(trainable, total)` where trainable sums `numel()` over parameters with `requires_grad == True` and total sums over all of them.

**Done when** `hw -k step3` prints `2 passed, 9 deselected`.

**Check it by hand**

```python
>>> base = nn.Linear(4, 3)                 # 4*3 weights + 3 biases
>>> count_trainable_parameters(base)
(15, 15)
>>> count_trainable_parameters(LoRALinear(base, r=2, alpha=4.0))
(14, 29)                                   # A is 2x4=8, B is 3x2=6; the base is frozen
```

**Why it matters.** These are small enough to check on paper, which is the point: do it once here and the headline LoRA number stops being a slogan. On the full `TinyClassifier` with `fc1` and `fc2` adapted you should see **514 of 2114 parameters trainable, about 24%** on a model this tiny. The saving grows with model size, which is why the paper reports 10,000x.

### Step 4, `make_toy_dataset` (10 pts)

**Write** the data generator. Use `torch.Generator().manual_seed(seed)` so two calls with the same seed return identical tensors. The label is 1 when the sum of the first half of the features exceeds the sum of the second half.

**Done when** `hw -k step4` prints `1 passed, 10 deselected`.

**Check it by hand**

```python
>>> X, y = make_toy_dataset(n=8, in_dim=4, seed=0)
>>> X.shape, y.shape
(torch.Size([8, 4]), torch.Size([8]))
>>> y.tolist()
[0, 1, 0, 1, 0, 0, 1, 1]
>>> torch.equal(make_toy_dataset(8, 4, 0)[0], make_toy_dataset(8, 4, 0)[0])
True
```

**Why it matters.** Using the global RNG instead of an explicit generator makes your loss curve depend on whatever ran before it. Seeding properly is what lets step 5's 'the loss went down' claim mean anything.

### Step 5, `train_lora` (20 pts)

**Write** the training loop: optimize **only** the parameters with `requires_grad == True`, run cross-entropy for `epochs` passes, and return the list of per-epoch losses.

**Done when** `hw -k step5` prints `2 passed, 9 deselected`.

**Check it by hand**

```python
>>> model = TinyClassifier()
>>> inject_lora(model, target="fc1", r=4, alpha=8.0)
>>> inject_lora(model, target="fc2", r=4, alpha=8.0)
>>> X, y = make_toy_dataset(n=256, in_dim=16, seed=0)
>>> [round(v, 4) for v in train_lora(model, X, y, epochs=5, lr=0.05)]
[0.6793, 0.6426, 0.5166, 0.2764, 0.1052]
```

**Why it matters.** The loss falls from chance (`log 2 = 0.693`) to 0.11 while the frozen base weights never move: the test checks that too. That is the claim of the paper reproduced on a model small enough to inspect.

### Step 6, Run the whole thing (0 pts)

```bash
hw
```

Every step green means `11 passed`. If a step you finished earlier has gone red,
you broke it with a later change; fix that before you submit.

## Written reflection (20 pts)

Worth 20 points: 15 for the answers, 5 for an honest AI-use note.

Answer in `ANSWERS.md`, 2-4 sentences each:

- **Q1.** Why is `B` initialized to zero rather than `A`? What would go wrong if *both*
  were random at init?
- **Q2.** A LoRA adapter for an `n x n` linear layer at rank `r` has how many trainable
  parameters, in terms of `n` and `r`? At what `r` does LoRA stop saving anything?
- **Q3.** Name one advantage LoRA gives at deployment time when you have many
  task-specific adapters for the same base model.

## What to submit

- `lora.py` with every TODO filled in and `hw` fully green.
- `ANSWERS.md` with Q1-Q3 answered.
- The `AI-USE:` note described below (worth 5 of the reflection points).

Partial credit follows the tests: each step is worth the points listed above, and a
step whose tests pass earns them. Code that does not import earns at most the written
points, so submit something that runs even if it is incomplete.

## AI-use disclosure (required)

Per the syllabus, you may use LLM tools as coding assistants, but you must
**disclose** it (which tool, for what), be able to **explain every line** you
submit, and write the reflection in your own words. Put a short `AI-USE:` note
in your file header. Undisclosed AI use is an academic-integrity violation.
