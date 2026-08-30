"""W6C2 reference solution, fill in the [MASK] + fine-tune a tiny BERT."""
from __future__ import annotations

MLM_MODEL = "prajjwal1/bert-tiny"

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
    # bert-tiny ships only vocab.txt (no tokenizer.json) and no `model_type` in
    # its config, so the Auto* dispatch that `pipeline(model=...)` uses fails
    # under transformers >= 5. Build the fast tokenizer + model explicitly and
    # hand them to the pipeline.
    from transformers import BertForMaskedLM, BertTokenizerFast, pipeline

    tok = BertTokenizerFast.from_pretrained(MLM_MODEL)
    model = BertForMaskedLM.from_pretrained(MLM_MODEL)
    fill = pipeline("fill-mask", model=model, tokenizer=tok)
    results = fill(sentence_with_mask, top_k=k)
    return [r["token_str"].strip() for r in results]


def finetune_and_eval(epochs: int = 8, seed: int = 0) -> float:
    import torch
    from transformers import BertForSequenceClassification, BertTokenizerFast

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
    test_texts = [t for t, _ in TEST_DATA]
    test_labels = torch.tensor([y for _, y in TEST_DATA])
    tenc = tok(test_texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        logits = model(**tenc).logits
    preds = logits.argmax(dim=-1)
    return (preds == test_labels).float().mean().item()


def _demo() -> None:
    preds = top_mask_predictions("The capital of France is [MASK].")
    print("Top [MASK] predictions:", preds)
    acc = finetune_and_eval()
    print(f"Fine-tuned tiny BERT test accuracy: {acc:.2f}")


if __name__ == "__main__":
    _demo()
