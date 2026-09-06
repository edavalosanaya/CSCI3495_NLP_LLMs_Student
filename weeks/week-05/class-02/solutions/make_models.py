#!/usr/bin/env python3
"""Train the three small LSTMs the W5C2 walkthrough loads.

One LSTM, three heads: generate the next character, label every token, or
classify the whole sequence. The three forward() methods differ by one line,
which is the point of the class. Headlines are the sarcasm corpus (Misra &
Arora); the tagger is trained on NLTK's Penn Treebank sample.
"""
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

DATA = Path(__file__).resolve().parents[1] / "exercise" / "data"
SEED = 0


class Generator(nn.Module):
    """Job 1. One output per step, scoring every possible next word."""

    def __init__(self, n_words, hidden=160):
        super().__init__()
        self.embedding = nn.Embedding(n_words, hidden)
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_words)
        # Tie the head to the embedding: the same matrix reads words in and
        # scores them on the way out, which halves the file we have to ship.
        self.head.weight = self.embedding.weight

    def forward(self, ids, state=None):
        outputs, state = self.lstm(self.embedding(ids), state)
        return self.head(outputs), state


class Tagger(nn.Module):
    """Job 2. One output per step, scoring every possible tag."""

    def __init__(self, n_words, n_tags, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(n_words, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden, n_tags)

    def forward(self, ids):
        outputs, _ = self.lstm(self.embedding(ids))
        return self.head(outputs)


class Classifier(nn.Module):
    """Job 3. ONE output for the whole sequence, from the final hidden state."""

    def __init__(self, n_words, hidden=128):
        super().__init__()
        self.embedding = nn.Embedding(n_words, 64, padding_idx=0)
        self.lstm = nn.LSTM(64, hidden, batch_first=True)
        self.head = nn.Linear(hidden, 2)

    def forward(self, ids):
        outputs, (last_hidden, last_cell) = self.lstm(self.embedding(ids))
        return self.head(last_hidden[-1])


def train_generator():
    frame = pd.read_csv(DATA / "headlines.csv")
    lines = frame[frame["is_sarcastic"] == 1]["headline"].astype(str).tolist()

    counts = Counter(w for line in lines for w in line.split())
    words = {"<pad>": 0, "<unk>": 1, "<end>": 2}
    for w, n in counts.most_common(4000):
        words[w] = len(words)
    index = {i: w for w, i in words.items()}

    stream = []
    for line in lines:
        stream += [words.get(w, 1) for w in line.split()] + [2]
    ids = torch.tensor(stream, dtype=torch.long)
    span = 24
    usable = (len(ids) - 1) // span * span
    x = ids[:usable].view(-1, span)
    y = ids[1:usable + 1].view(-1, span)

    torch.manual_seed(SEED)
    model = Generator(len(words))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.003)
    schedule = torch.optim.lr_scheduler.StepLR(optimiser, step_size=6, gamma=0.4)
    loss_function = nn.CrossEntropyLoss()
    for epoch in range(15):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), 64):
            batch = order[start:start + 64]
            optimiser.zero_grad()
            scores, _ = model(x[batch])
            loss = loss_function(scores.reshape(-1, len(words)), y[batch].reshape(-1))
            loss.backward()
            optimiser.step()
            total += loss.item()
        schedule.step()
        print(f"  generator epoch {epoch + 1}  loss {total / (len(x) / 64):.3f}")
    # The corpus is real published satire, so it swears. The rows below are
    # masked out at sampling time; the list travels in the checkpoint rather
    # than sitting in the student notebook.
    blocked = {"<unk>", "<pad>", "fuck", "fucking", "fucked", "shit", "shitty",
               "bullshit", "ass", "asshole", "dick", "bitch", "damn", "goddamn",
               "piss", "pissed", "cunt", "slut", "whore", "sex", "sexual",
               "sexy", "porn", "penis", "vagina", "masturbating", "rape",
               "raped", "nigger", "fag", "faggot"}
    banned = sorted({words[w] for w in blocked if w in words})
    torch.save({"state": model.state_dict(), "words": words, "index": index,
                "banned": banned}, DATA / "generator.pt")


def train_tagger():
    import nltk
    nltk.download("treebank", quiet=True)
    from nltk.corpus import treebank
    sentences = [[(w.lower(), t) for w, t in s] for s in treebank.tagged_sents()]
    counts = Counter(w for s in sentences for w, _ in s)
    words = {"<pad>": 0, "<unk>": 1}
    for w, _ in counts.most_common(8000):
        words[w] = len(words)
    tags = sorted({t for s in sentences for _, t in s})
    tag_index = {t: i for i, t in enumerate(tags)}

    span = 40
    ids = np.zeros((len(sentences), span), dtype=np.int64)
    labels = np.full((len(sentences), span), -100, dtype=np.int64)
    for i, s in enumerate(sentences):
        for j, (w, t) in enumerate(s[:span]):
            ids[i, j] = words.get(w, 1)
            labels[i, j] = tag_index[t]

    torch.manual_seed(SEED)
    model = Tagger(len(words), len(tags))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.003)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)
    x, y = torch.from_numpy(ids), torch.from_numpy(labels)
    for epoch in range(8):
        order = torch.randperm(len(x))
        total = 0.0
        for start in range(0, len(x), 32):
            batch = order[start:start + 32]
            optimiser.zero_grad()
            loss = loss_function(model(x[batch]).reshape(-1, len(tags)), y[batch].reshape(-1))
            loss.backward()
            optimiser.step()
            total += loss.item()
        with torch.no_grad():
            predicted = model(x).argmax(-1)
        real = y != -100
        accuracy = (predicted[real] == y[real]).float().mean().item()
        print(f"  tagger epoch {epoch + 1}  loss {total / (len(x) / 32):.3f}  train acc {accuracy:.3f}")
    torch.save({"state": model.state_dict(), "words": words, "tags": tags},
               DATA / "tagger.pt")


def train_classifier():
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    frame = pd.read_csv(DATA / "headlines.csv")
    texts = frame["headline"].astype(str).tolist()
    labels = frame["is_sarcastic"].tolist()
    tr, te, ytr, yte = train_test_split(texts, labels, test_size=0.2,
                                        random_state=SEED, stratify=labels)
    counts = Counter(w for t in tr for w in t.split())
    words = {"<pad>": 0, "<unk>": 1}
    for w, _ in counts.most_common(10000):
        words[w] = len(words)

    span = 25

    def encode(rows):
        out = np.zeros((len(rows), span), dtype=np.int64)
        for i, t in enumerate(rows):
            for j, w in enumerate(t.split()[:span]):
                out[i, j] = words.get(w, 1)
        return torch.from_numpy(out)

    torch.manual_seed(SEED)
    model = Classifier(len(words))
    optimiser = torch.optim.Adam(model.parameters(), lr=0.002)
    loss_function = nn.CrossEntropyLoss()
    x, y = encode(tr), torch.tensor(ytr)
    xt = encode(te)
    for epoch in range(5):
        order = torch.randperm(len(x))
        for start in range(0, len(x), 64):
            batch = order[start:start + 64]
            optimiser.zero_grad()
            loss_function(model(x[batch]), y[batch]).backward()
            optimiser.step()
        with torch.no_grad():
            accuracy = accuracy_score(yte, model(xt).argmax(1).numpy())
        print(f"  classifier epoch {epoch + 1}  test acc {accuracy:.3f}")
    torch.save({"state": model.state_dict(), "words": words}, DATA / "classifier.pt")


def main() -> None:
    train_tagger()
    train_classifier()
    train_generator()
    for f in ("generator.pt", "tagger.pt", "classifier.pt"):
        print(f"  {f}  {(DATA / f).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
