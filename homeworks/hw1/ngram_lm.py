"""HW1 starter, n-gram language models.

Implement the functions/methods below, then run the tests:

    docker compose -f docker/docker-compose.yml run --rm course \
        python -m pytest homeworks/hw1 -q

Each stub raises NotImplementedError; replace the body with your code.
Do NOT change the public function/method signatures, the tests rely on them.

You may use only the Python standard library (re, math, collections, random).
"""
# Each TODO below names its README step. Check one step with:
#     python -m pytest homeworks/hw1 -q -k step3      (or step1, step2, ...)
# and the whole assignment with:
#     python -m pytest homeworks/hw1 -q

from __future__ import annotations

import math
import random
import re
from collections import Counter, defaultdict

# Sentence boundary markers. Every sentence is padded so the model can
# condition on the start of a sentence and predict its end.
BOS = "<s>"   # beginning-of-sentence
EOS = "</s>"  # end-of-sentence
UNK = "<unk>"  # unknown / out-of-vocabulary token


# ---------------------------------------------------------------------------
# Step 1, Preprocessing
# ---------------------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    """Lowercase `text` and split it into word/punctuation tokens.

    Rules:
      - Lowercase everything.
      - A token is either a run of word characters (``\\w+``) OR a single
        non-word, non-space character (e.g. punctuation).
      - Collapsing of whitespace is implicit in the regex.

    Example:
        tokenize("Hi, NLP!") -> ["hi", ",", "nlp", "!"]
    """
    # TODO (STEP 1): implement using re.findall with the pattern r"\w+|[^\w\s]"
    raise NotImplementedError


def sentences(text: str) -> list[list[str]]:
    """Split `text` into sentences, then tokenize each sentence.

    Split on the sentence-final punctuation characters ``.``, ``!`` and ``?``
    (one or more of them in a row count as a single boundary). The punctuation
    itself is NOT kept as a token here. Empty sentences are dropped.

    Example:
        sentences("Hi there. NLP rocks!") -> [["hi", "there"], ["nlp", "rocks"]]
    """
    # TODO (STEP 2): split on [.!?]+ , tokenize each piece, drop empties
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Step 2, The n-gram model
# ---------------------------------------------------------------------------
class NGramLM:
    """An n-gram language model with add-k (Laplace) smoothing.

    Attributes set by ``fit``:
      - self.n               : the order (1=unigram, 2=bigram, ...)
      - self.k               : smoothing constant
      - self.vocab           : set[str] of known tokens, including EOS and UNK
      - self.context_counts  : Counter mapping (n-1)-gram context tuple -> total count
      - self.ngram_counts    : Counter mapping full n-gram tuple -> count
    """

    def __init__(self, n: int = 2, k: float = 1.0):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.k = k
        self.vocab: set[str] = set()
        self.context_counts: Counter = Counter()
        self.ngram_counts: Counter = Counter()

    # -- helpers ---------------------------------------------------------
    def pad(self, tokens: list[str]) -> list[str]:
        """Pad a token list with (n-1) BOS markers and one EOS marker.

        For a bigram model (n=2): ["a","b"] -> ["<s>", "a", "b", "</s>"].
        For a trigram model (n=3): ["a"] -> ["<s>", "<s>", "a", "</s>"].
        A unigram model (n=1) adds no BOS, only the trailing EOS.
        """
        # TODO (STEP 3): return [BOS]*(self.n - 1) + tokens + [EOS]
        raise NotImplementedError

    def ngrams(self, tokens: list[str]) -> list[tuple[str, ...]]:
        """Return the list of n-grams (as tuples) from an already-padded list.

        Example (n=2): ["<s>","a","</s>"] -> [("<s>","a"), ("a","</s>")]
        """
        # TODO (STEP 3): slide a window of length self.n over tokens
        raise NotImplementedError

    # -- training --------------------------------------------------------
    def fit(self, corpus: list[list[str]]) -> "NGramLM":
        """Estimate counts from a corpus (list of tokenized sentences).

        Steps:
          1. Build the vocabulary as the set of all tokens that appear, PLUS
             the special tokens EOS and UNK. (BOS is a context-only marker and
             is NOT a predictable outcome, so it is excluded from `vocab`.)
          2. For each sentence: pad it, slide n-grams, and increment
             ``ngram_counts[ngram]`` and ``context_counts[ngram[:-1]]``.

        Returns self (so you can chain ``NGramLM(2).fit(corpus)``).
        """
        # TODO (STEP 4): implement
        raise NotImplementedError

    # -- probabilities ---------------------------------------------------
    def _map(self, token: str) -> str:
        """Map an out-of-vocabulary token to UNK; otherwise return it as-is."""
        # TODO (STEP 4): return token if token in self.vocab else UNK
        raise NotImplementedError

    def prob(self, token: str, context: tuple[str, ...]) -> float:
        """Add-k smoothed probability P(token | context).

        Formula:
            P(w | context) = (count(context, w) + k)
                             / (count(context) + k * |V|)

        where |V| = len(self.vocab). The `context` tuple must have length
        ``n-1``; tokens (both in context and the target) are mapped through
        ``_map`` so unseen words become UNK. For a unigram model the context
        is the empty tuple and ``context_counts[()]`` holds the total number
        of tokens observed.
        """
        # TODO (STEP 5): implement add-k smoothing
        raise NotImplementedError

    def sentence_logprob(self, tokens: list[str]) -> float:
        """Natural-log probability of one sentence (a list of raw tokens).

        Pad the tokens, then sum log P(w_i | preceding n-1 tokens) over every
        position that has a full context (i.e. every n-gram produced by
        ``ngrams`` on the padded sequence). Use math.log (natural log).
        """
        # TODO (STEP 6): implement
        raise NotImplementedError

    # -- evaluation ------------------------------------------------------
    def perplexity(self, corpus: list[list[str]]) -> float:
        """Perplexity of the model over a corpus of tokenized sentences.

        Perplexity = exp( - (sum of sentence log-probs) / N )
        where N is the total number of PREDICTED tokens across the corpus,
        i.e. the number of n-grams scored (each non-BOS position, including
        the EOS marker). Use natural log with math.exp to invert.
        """
        # TODO (STEP 7): accumulate total log-prob and total predicted-token count
        raise NotImplementedError

    # -- generation ------------------------------------------------------
    def generate(self, max_len: int = 20, seed: int | None = None) -> list[str]:
        """Sample a sentence from the model.

        Start from the context (BOS,)*(n-1). Repeatedly sample the next token
        from the smoothed distribution P(w | context) over ``self.vocab``,
        append it, and slide the context. Stop when EOS is produced or
        ``max_len`` tokens have been generated. Do NOT include BOS or EOS in
        the returned list.

        Use a local ``random.Random(seed)`` instance for reproducibility.
        """
        # TODO (STEP 8): implement weighted sampling with random.Random(seed)
        raise NotImplementedError
