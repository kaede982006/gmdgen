"""Feature extraction utilities.

Legacy package kept for backward compatibility.
New code should import from gmdgen.representation instead.
"""

from gmdgen.features.tokenizer import (
    EOS_TOKEN,
    extract_object_field,
    extract_object_id,
    extract_object_number,
    level_data_to_tokens,
    parse_object_pairs,
    records_to_token_sequences,
    rewrite_object_xy,
    tokens_to_object_ids,
)
from gmdgen.features.stats import (
    object_token_frequencies,
    summarize_sequences,
)

__all__ = [
    "EOS_TOKEN",
    "extract_object_field",
    "extract_object_id",
    "extract_object_number",
    "level_data_to_tokens",
    "object_token_frequencies",
    "parse_object_pairs",
    "records_to_token_sequences",
    "rewrite_object_xy",
    "summarize_sequences",
    "tokens_to_object_ids",
]
