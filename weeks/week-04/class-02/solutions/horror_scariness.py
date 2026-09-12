"""W4C2 reference solution: predicting scariness from horror reviews.

This is the code we wrote together at the front of the room, complete and in
order, so you have it while you do the comedy version yourself.

The method, in one sentence: three hand-written lists of adjectives turn each
film's reviews into three counts, and four trainable numbers turn those counts
into a prediction.

Run it:

    uv run python weeks/week-04/class-02/solutions/horror_scariness.py

`weeks/week-04/class-02/solutions/WALKTHROUGH.md`, beside this file, takes it a
step at a time. Read only the step you are stuck on.
"""
from pathlib import Path

import pandas as pd
import torch

DATA = Path(__file__).resolve().parent.parent / "exercise" / "data"

STEPS = 200
LEARNING_RATE = 0.05


def load() -> pd.DataFrame:
    """Stage 1. One row per film; `reviews` is text and `scariness` is a number."""
    return pd.read_csv(DATA / "horror_reviews.csv")


def word_lists() -> dict[str, list[str]]:
    """Stage 2. The rule, read from the three text files.

    `.read().split()` on a file of one word per line gives a list of words.
    """
    return {name: (DATA / f"{name}.txt").read_text().split()
            for name in ("fear", "gore", "dull")}


def count_words(df: pd.DataFrame, lists: dict[str, list[str]]) -> pd.DataFrame:
    """Stage 3. Text in, numbers out.

    The loop is over the three CATEGORIES, not over the films: `.str.count`
    already reaches every row at once, which is why there is no loop over rows
    anywhere in this file.
    """
    low = df["reviews"].str.lower()
    for name, words in lists.items():
        df[name] = sum(low.str.count(word) for word in words)
    return df


def to_tensors(df: pd.DataFrame, features: list[str], target: str):
    """Stage 4. float32, and centre each feature column.

    Centring subtracts each column's own average so the average film sits at
    zero. Without it the first weight lands quickly and the bias crawls for
    thousands of steps, because the bias has a much smaller gradient.
    """
    X = torch.tensor(df[features].values, dtype=torch.float32)
    y = torch.tensor(df[target].values, dtype=torch.float32)
    return X - X.mean(0), y


def train(X, y, steps: int = STEPS, lr: float = LEARNING_RATE):
    """Stages 5 to 7. The model is four numbers, and this is how they move.

    `requires_grad` is the switch that makes a number trainable. `backward()`
    fills in `.grad` and moves nothing at all; the two lines under `no_grad`
    are what move the model. `zero_()` is there because gradients accumulate by
    default, so a loop without it trains on a running total.
    """
    w = torch.zeros(X.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)

    for _ in range(steps):
        loss = ((X @ w + b - y) ** 2).mean()      # measure
        loss.backward()                           # blame
        with torch.no_grad():                     # step
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()                            # clear
        b.grad.zero_()

    with torch.no_grad():
        final = ((X @ w + b - y) ** 2).mean().item()
    return w.detach(), b.detach(), final


def baseline(y) -> float:
    """What you score by ignoring the reviews and guessing the average.

    Always compare against this. A model always returns an answer; whether the
    answer is worth anything is a separate question.
    """
    return float(((y - y.mean()) ** 2).mean())


def main() -> None:
    features = ["fear", "gore", "dull"]
    df = count_words(load(), word_lists())
    X, y = to_tensors(df, features, "scariness")
    w, b, loss = train(X, y)

    print(f"{len(df)} films")
    for name, value in zip(features, w.tolist()):
        print(f"  {name:>5} words  {value:+.3f}")
    print(f"  {'bias':>5}        {b.item():+.3f}")
    print(f"\nloss {loss:.4f}   against {baseline(y):.4f} for guessing the average")

    # Self-check: the numbers this prints are the numbers on the slides.
    assert abs(w[0].item() - 1.283) < 0.01, w
    assert abs(w[2].item() + 0.587) < 0.01, w
    assert abs(loss - 0.2055) < 0.001, loss
    print("\nmatches the slides: fear +1.283, gore +0.334, dull -0.587, loss 0.2055")


if __name__ == "__main__":
    main()
