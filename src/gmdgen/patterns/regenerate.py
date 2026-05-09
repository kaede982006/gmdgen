# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from gmdgen.patterns.builder import PATTERNS_INDEX_PATH, build_index


def main() -> None:
    index = build_index(PATTERNS_INDEX_PATH, write_pattern_files=True)
    print(
        "regenerated "
        f"{len(index.get('patterns', {}))} patterns across {len(index.get('cells', {}))} cells"
    )


if __name__ == "__main__":
    main()
