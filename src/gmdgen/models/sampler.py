# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import random
from collections import Counter


def sample_from_counter(
    counter: Counter[str],
    *,
    rng: random.Random,
    temperature: float = 1.0,
    top_k: int = 0,
) -> str:
    if not counter:
        raise ValueError("Cannot sample from an empty counter.")

    if temperature <= 0:
        raise ValueError("temperature must be greater than 0.")

    candidates = counter.most_common()
    if top_k > 0:
        candidates = candidates[:top_k]

    weighted: list[tuple[str, float]] = []
    inv_temp = 1.0 / temperature
    for token, count in candidates:
        weighted.append((token, float(count) ** inv_temp))

    total = sum(weight for _, weight in weighted)
    pivot = rng.random() * total

    cumulative = 0.0
    for token, weight in weighted:
        cumulative += weight
        if cumulative >= pivot:
            return token

    return weighted[-1][0]
