# SPDX-License-Identifier: GPL-3.0-or-later
"""Representation learning utilities for GD level objects.

This package supersedes the legacy gmdgen.features package.
Both packages remain importable for backward compatibility:

  - gmdgen.features.tokenizer   → legacy OBJ:{id} tokens (still used by MarkovModel)
  - gmdgen.representation.tokenizer → feature tokens with CLS/DX/Y/SEC fields

  - gmdgen.features.stats       → sequence-level stats
  - gmdgen.representation.stats → adds class_entropy, dx_entropy, sec_count
"""

from gmdgen.representation.object_classifier import (
    ObjectClass,
    class_short,
    classify,
    is_structural,
    is_visible,
)
from gmdgen.representation.tokenizer import (
    level_data_to_feature_tokens,
    records_to_feature_sequences,
    to_feature_token,
)
from gmdgen.representation.stats import summarize_feature_sequences

__all__ = [
    "ObjectClass",
    "class_short",
    "classify",
    "is_structural",
    "is_visible",
    "level_data_to_feature_tokens",
    "records_to_feature_sequences",
    "to_feature_token",
    "summarize_feature_sequences",
]
