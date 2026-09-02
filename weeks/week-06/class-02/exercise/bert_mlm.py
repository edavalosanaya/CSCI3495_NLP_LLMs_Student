"""W6C2 starter, fill in the [MASK] + fine-tune a tiny BERT."""
from __future__ import annotations

MLM_MODEL = "prajjwal1/bert-tiny"

# A tiny, deterministic sentiment dataset (label 1 = positive, 0 = negative).
TRAIN_DATA = [
    ("I loved this movie, it was fantastic", 1),
    ("An amazing and wonderful experience", 1),
    ("Truly great, I enjoyed every minute", 1),
    ("Brilliant acting and a beautiful story", 1),
    ("This was terrible and boring", 0),
    ("I hated it, a complete waste of time", 0),
    ("Awful, the worst film I have seen", 0),
    ("Dull, slow, and deeply disappointing", 0),
]
TEST_DATA = [
    ("What a wonderful and great film", 1),
    ("A boring and awful waste", 0),
]


def top_mask_predictions(sentence_with_mask: str, k: int = 5) -> list[str]:
    """Ask a pretrained BERT what belongs in the [MASK].

    Args:
        sentence_with_mask: a sentence containing exactly one "[MASK]".
        k: how many candidates to return, best first.

    Returns:
        k word strings, whitespace stripped, most likely first. These are BERT's
        own pretraining objective answering, with no fine-tuning involved.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    #
    #   The formula is the first half of README section 2.
    #
    #   build the tokenizer and the masked-LM model from MLM_MODEL. bert-tiny
    #       ships only a vocab.txt, so use BertTokenizerFast and
    #       BertForMaskedLM explicitly rather than the Auto* classes
    #   hand both to a "fill-mask" pipeline
    #   ask that pipeline for the top k fillings of the sentence
    #   each result is a dict; the word you want is under "token_str", and it
    #       needs stripping
    #
    raise NotImplementedError


def finetune_and_eval(epochs: int = 8, seed: int = 0) -> float:
    """Fine-tune the same pretrained BERT as a sentiment classifier.

    Args:
        epochs: how many full passes over TRAIN_DATA. The dataset is eight
            sentences, so an epoch is one batch.
        seed: seeds torch, so the result is reproducible.

    Returns:
        Accuracy on TEST_DATA as a float in [0, 1].
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
    #
    #   The formula is the second half of README section 2.
    #
    #   seed torch, then load the tokenizer and a SEQUENCE CLASSIFICATION model
    #       from MLM_MODEL with two labels. The classification head starts
    #       random; everything beneath it is pretrained
    #   encode all the training texts in one go, padded and truncated, and put
    #       their labels in a tensor
    #   make an AdamW optimizer over the model's parameters
    #   loop for `epochs`: clear the gradients, run the model WITH labels so it
    #       returns a loss, backpropagate, and step
    #   switch to eval mode, predict on the test texts, and return the fraction
    #       the model got right
    #
    #   Passing labels= is what makes the model compute the loss for you.
    #
    raise NotImplementedError


def _demo() -> None:
    try:
        preds = top_mask_predictions("The capital of France is [MASK].")
        print("Top [MASK] predictions:", preds)
        acc = finetune_and_eval()
        print(f"Fine-tuned tiny BERT test accuracy: {acc:.2f}")
    except NotImplementedError:
        print("Implement the TODOs in bert_mlm.py first.")


if __name__ == "__main__":
    _demo()
