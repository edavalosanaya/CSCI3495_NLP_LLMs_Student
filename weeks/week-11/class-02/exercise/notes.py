"""A tiny corpus of CSCI 3495 course notes for the mini-RAG exercise.

Each document is a (source, text) pair; documents contain several passages
separated by blank lines (the chunker splits on those).
"""

NOTES: list[tuple[str, str]] = [
    ("week01-text-processing.md",
     "Tokenization splits text into discrete units called tokens, which may be "
     "words, subwords, or characters. The choice of granularity affects "
     "vocabulary size and sequence length.\n\n"
     "Minimum edit distance measures how different two strings are: the fewest "
     "insert, delete, and substitute operations to turn one into the other. It "
     "is computed with dynamic programming."),

    ("week05-transformer.md",
     "The Transformer architecture, introduced in Attention Is All You Need "
     "(Vaswani et al., 2017), relies entirely on self-attention and has no "
     "recurrence. Multi-head attention lets the model attend to different "
     "positions in parallel.\n\n"
     "Positional encodings inject information about token order because "
     "self-attention is otherwise permutation-invariant."),

    ("week07-decoding.md",
     "Decoding strategies control how an autoregressive model selects the next "
     "token. Greedy decoding always takes the most likely token. Temperature "
     "scales the logits: higher temperature increases randomness, lower "
     "temperature makes output more deterministic.\n\n"
     "Top-k and top-p (nucleus) sampling restrict the choice to a subset of the "
     "most probable tokens, balancing quality and diversity."),

    ("week10-prompting.md",
     "In-context learning lets a large model perform a new task from examples in "
     "the prompt, with no weight updates. Zero-shot uses only an instruction; "
     "few-shot adds demonstrations.\n\n"
     "Chain-of-thought prompting adds intermediate reasoning steps to the "
     "demonstrations, which improves performance on multi-step reasoning tasks, "
     "especially for large models."),

    ("week11-rag.md",
     "Retrieval-Augmented Generation (Lewis et al., 2020) combines a retriever "
     "with a generator: relevant passages are fetched and placed in the prompt so "
     "the model answers from evidence rather than memory.\n\n"
     "RAG reduces hallucination, lets a model use fresh or private knowledge "
     "without retraining, and supports citing sources. Retrieval quality caps the "
     "overall system."),
]
