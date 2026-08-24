"""HW4 starter, Fine-tuning & adapting LLMs with LoRA / PEFT.

You will implement Low-Rank Adaptation (LoRA) *from scratch* and then verify
your understanding against the Hugging Face `peft` library.

Reference: Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models"
(2021), arXiv:2106.09685.

Core idea (Eq. 3 in the paper): freeze the pretrained weight W0 and learn a
low-rank update  ΔW = B @ A  with  A ∈ R^{r×k}, B ∈ R^{d×r}, r ≪ min(d, k).
The adapted layer computes
        h = W0 @ x  +  (alpha / r) * (B @ A) @ x
A is initialized from a small random Gaussian and B is initialized to ZERO, so
that ΔW = 0 at the start of training (the model behaves exactly like the
pretrained model on step 0). Only A and B are trained; W0 is frozen.

Run the tests with:
    docker compose -f docker/docker-compose.yml run --rm course \
        python -m pytest homeworks/hw4/tests -q
"""
# Each TODO below names its README step. Check one step with:
#     python -m pytest homeworks/hw4/tests -q -k step3      (or step1, step2, ...)
# and the whole assignment with:
#     python -m pytest homeworks/hw4/tests -q

from __future__ import annotations

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Step 1, A LoRA-adapted linear layer (implement the math yourself).
# ---------------------------------------------------------------------------
class LoRALinear(nn.Module):
    """A frozen `nn.Linear` wrapped with a trainable low-rank LoRA update.

    Args:
        base: an existing, pretrained ``nn.Linear`` (its weight/bias are FROZEN).
        r: LoRA rank (r >= 1).
        alpha: LoRA scaling factor; the update is scaled by ``alpha / r``.

    Forward:
        y = base(x) + (alpha / r) * (x @ A^T @ B^T)

    where A has shape (r, in_features) and B has shape (out_features, r).
    Initialize A ~ N(0, 0.02) and B = 0 so the update starts at zero.
    The base layer's parameters must have ``requires_grad == False``.
    """

    def __init__(self, base: nn.Linear, r: int = 4, alpha: float = 8.0) -> None:
        super().__init__()
        # TODO (STEP 1): store base (frozen), r, alpha, scaling = alpha / r.
        # TODO (STEP 1): create nn.Parameter A of shape (r, in_features), init N(0, 0.02).
        # TODO (STEP 1): create nn.Parameter B of shape (out_features, r), init zeros.
        # TODO (STEP 1): freeze base.weight and base.bias (requires_grad = False).
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO (STEP 1): return base(x) + scaling * (x @ A.T @ B.T)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2, Inject LoRA into an existing model and report trainable params.
# ---------------------------------------------------------------------------
def inject_lora(model: nn.Module, target: str, r: int = 4, alpha: float = 8.0) -> nn.Module:
    """Replace every direct child module named ``target`` that is an nn.Linear
    with a ``LoRALinear`` wrapping it. Recurse into submodules.

    Returns the (mutated) model.
    Hint: iterate ``model.named_children()``; if a child is an nn.Linear and its
    attribute name == target, ``setattr`` a LoRALinear; otherwise recurse.
    """
    # TODO (STEP 2): implement recursive replacement.
    raise NotImplementedError


def count_trainable_parameters(model: nn.Module) -> tuple[int, int]:
    """Return (trainable, total) parameter counts for ``model``.

    trainable = sum of numel() for params with requires_grad == True.
    total     = sum of numel() over all params.
    """
    # TODO (STEP 3): implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 3, A tiny model + toy task you can fine-tune on a CPU in seconds.
# ---------------------------------------------------------------------------
class TinyClassifier(nn.Module):
    """A minimal 2-layer MLP "pretrained" backbone + classification head.

    The point is to have real ``nn.Linear`` layers to attach LoRA to, NOT to
    be a good model. ``fc1``/``fc2`` stand in for frozen pretrained weights.
    """

    def __init__(self, in_dim: int = 16, hidden: int = 32, n_classes: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return self.head(x)


def make_toy_dataset(n: int = 256, in_dim: int = 16, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministic linearly-separable-ish binary classification data.

    Label = 1 if the sum of the first half of features > sum of the second half.
    Returns (X float32 [n, in_dim], y long [n]).
    """
    # TODO (STEP 4): use a torch.Generator(seed) so the data is deterministic.
    raise NotImplementedError


def train_lora(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 30,
    lr: float = 0.05,
    seed: int = 0,
) -> list[float]:
    """Train ONLY the trainable (LoRA) parameters with Adam + cross-entropy.

    Steps:
      * torch.manual_seed(seed)
      * optimizer over [p for p in model.parameters() if p.requires_grad]
      * full-batch gradient descent for ``epochs`` steps
      * record the loss (float) BEFORE each step's update
    Returns the list of per-epoch loss values (length == epochs).
    """
    # TODO (STEP 5): implement the training loop and return the loss history.
    raise NotImplementedError
