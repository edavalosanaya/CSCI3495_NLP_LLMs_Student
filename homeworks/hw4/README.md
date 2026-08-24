# HW4: Fine-tuning & Adapting LLMs with LoRA / PEFT

**Out:** Week 8, Class 1 · **Due:** Week 9, Class 1 · **Weight:** 5% of course grade
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

## Tasks
Edit **`homeworks/hw4/lora.py`** and implement each `# TODO`. Do **not** edit
the test files.

1. **`LoRALinear`** (Task 1): wrap a frozen `nn.Linear` with trainable `A`/`B`
   low-rank matrices. Initialize `A ~ N(0, 0.02)`, `B = 0`, scale by
   `alpha / r`, and freeze the base layer.
2. **`inject_lora` / `count_trainable_parameters`** (Task 2): recursively
   replace child `nn.Linear` layers named `target` with `LoRALinear`, and count
   trainable vs. total parameters.
3. **`make_toy_dataset` / `train_lora`** (Task 3): build a deterministic toy
   binary-classification dataset and train **only** the LoRA parameters with
   Adam + cross-entropy, returning the per-epoch loss history.

### Short written questions (put answers in `ANSWERS.md`)
Answer briefly (2-4 sentences each):
- **Q1.** Why is `B` initialized to zero rather than `A`? What would go wrong if
  *both* were random at init?
- **Q2.** A LoRA adapter for an `n×n` linear layer at rank `r` has how many
  trainable parameters (in terms of `n` and `r`)? At what `r` does LoRA stop
  saving parameters?
- **Q3.** Name one advantage LoRA gives at *deployment/serving* time when you
  have many task-specific adapters for the same base model.

## Deliverables
- Completed `homeworks/hw4/lora.py` (all TODOs implemented; tests pass).
- `homeworks/hw4/ANSWERS.md` with Q1-Q3 and your **AI-use disclosure**.

## How to run & test
From the repo root, inside the course Docker image:
```bash
docker compose -f docker/docker-compose.yml run --rm course \
    python -m pytest homeworks/hw4/tests -q
```
All tests must pass. (One test cross-checks your mental model against the real
`peft` library; it runs fully offline, no model download.)

## Grading rubric (100 pts)
| Item | Pts |
|------|----:|
| `LoRALinear` correct (identity at init, scaling, frozen base) | 30 |
| `inject_lora` + `count_trainable_parameters` correct | 20 |
| `make_toy_dataset` deterministic; `train_lora` reduces loss; frozen weights unchanged | 30 |
| Written Q1-Q3 | 15 |
| AI-use disclosure present & honest | 5 |

## AI-use disclosure reminder
You may use LLM coding assistants, **but** you must (a) disclose any AI
assistance in `ANSWERS.md`, (b) be able to explain every line you submit, and
(c) never present AI-generated prose as your own. See the syllabus AI-use
policy.
