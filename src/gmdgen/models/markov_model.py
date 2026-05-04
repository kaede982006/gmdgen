from __future__ import annotations

import json
import random
from collections import Counter, defaultdict

from gmdgen.models.sampler import sample_from_counter

BOS_TOKEN = "<BOS>"


class MarkovModel:
    def __init__(self, order: int = 2) -> None:
        if order < 1:
            raise ValueError("order must be >= 1")
        self.order = order
        self.transitions: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)
        self.unigram: Counter[str] = Counter()

    def fit(self, sequences: list[list[str]]) -> None:
        for sequence in sequences:
            if not sequence:
                continue

            padded = [BOS_TOKEN] * self.order + sequence
            for idx in range(self.order, len(padded)):
                state = tuple(padded[idx - self.order : idx])
                next_token = padded[idx]
                self.transitions[state][next_token] += 1
                self.unigram[next_token] += 1

    def sample(
        self,
        *,
        max_steps: int,
        eos_token: str,
        rng: random.Random,
        temperature: float = 1.0,
        top_k: int = 0,
    ) -> list[str]:
        state = [BOS_TOKEN] * self.order
        sampled: list[str] = []

        for _ in range(max_steps):
            counter = self.transitions.get(tuple(state))
            if not counter:
                counter = self.unigram
            if not counter:
                break

            next_token = sample_from_counter(
                counter,
                rng=rng,
                temperature=temperature,
                top_k=top_k,
            )
            if next_token == eos_token:
                break

            sampled.append(next_token)
            state = state[1:] + [next_token]

        return sampled

    def to_dict(self) -> dict:
        transitions = []
        for state, counter in self.transitions.items():
            transitions.append(
                {
                    "state": list(state),
                    "next": dict(counter),
                }
            )
        return {
            "order": self.order,
            "transitions": transitions,
            "unigram": dict(self.unigram),
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "MarkovModel":
        model = cls(order=int(payload["order"]))
        for item in payload.get("transitions", []):
            state = tuple(item["state"])
            model.transitions[state] = Counter(item["next"])
        model.unigram = Counter(payload.get("unigram", {}))
        return model

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)
