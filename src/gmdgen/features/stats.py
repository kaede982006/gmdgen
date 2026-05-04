from __future__ import annotations

from collections import Counter
from statistics import mean

from gmdgen.features.tokenizer import EOS_TOKEN


def summarize_sequences(sequences: list[list[str]]) -> dict[str, float | int]:
    if not sequences:
        return {
            "num_sequences": 0,
            "avg_tokens": 0.0,
            "min_tokens": 0,
            "max_tokens": 0,
            "vocab_size": 0,
        }

    lengths = [max(0, len(seq) - 1) for seq in sequences]
    vocab = {token for seq in sequences for token in seq if token != EOS_TOKEN}

    return {
        "num_sequences": len(sequences),
        "avg_tokens": round(mean(lengths), 3),
        "min_tokens": min(lengths),
        "max_tokens": max(lengths),
        "vocab_size": len(vocab),
    }


def object_token_frequencies(sequences: list[list[str]], top_n: int = 50) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for sequence in sequences:
        for token in sequence:
            if token != EOS_TOKEN:
                counter[token] += 1
    return dict(counter.most_common(top_n))
