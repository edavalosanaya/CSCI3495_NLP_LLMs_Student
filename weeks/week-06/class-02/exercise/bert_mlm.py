"""W6C2 starter, fill in the [MASK] + fine-tune a tiny BERT.

Part A, Masked LM: feed a sentence with a [MASK] to a pretrained masked-LM and
read off the top predicted words. This is BERT's pretraining objective in action.

Part B, Fine-tuning: train a tiny BERT classifier on a tiny sentiment dataset
(a handful of sentences) and watch accuracy rise above chance. CPU-only, seconds.

Work through the lab in `README.md`. Each STEP below has its own check:
    python -m pytest weeks/week-06/class-02/exercise/test_bert_mlm.py -k step1 -q

Models (tiny, CPU): prajjwal1/bert-tiny. First run downloads a few MB, then cached.
"""
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
    """Return the top-k predicted words for the [MASK] in `sentence_with_mask`.

    Hint:
      1. Use a fill-mask pipeline. bert-tiny ships only a vocab.txt, so build the
         fast tokenizer + model explicitly and pass them to the pipeline:
           from transformers import BertForMaskedLM, BertTokenizerFast, pipeline
           tok = BertTokenizerFast.from_pretrained(MLM_MODEL)
           model = BertForMaskedLM.from_pretrained(MLM_MODEL)
           fill = pipeline("fill-mask", model=model, tokenizer=tok)
         (the tokenizer's mask token is "[MASK]" for BERT).
      2. Call fill(sentence_with_mask, top_k=k).
      3. Return the predicted token strings (the "token_str" field), stripped.
    """
    # TODO (STEP 1): implement. Check with: pytest -k step1
    raise NotImplementedError


def finetune_and_eval(epochs: int = 8, seed: int = 0) -> float:
    """Fine-tune a tiny BERT sequence classifier on TRAIN_DATA and return the
    accuracy on TEST_DATA (a float in [0, 1]).

    Hint (a minimal training loop):
      1. Seed torch for determinism (torch.manual_seed(seed)).
      2. Load BertTokenizerFast + BertForSequenceClassification(MLM_MODEL,
         num_labels=2).  (Auto* dispatch fails for bert-tiny under transformers 5.)
      3. Tokenize TRAIN_DATA (padding=True, truncation=True, return_tensors="pt").
      4. Train with AdamW for a few epochs on the cross-entropy loss
         (the model returns loss when you pass `labels=`).
      5. Put the model in eval(), tokenize TEST_DATA, take argmax over logits,
         and return accuracy.
    """
    # TODO (STEP 2): implement. Check with: pytest -k step2
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
