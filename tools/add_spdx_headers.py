# SPDX-License-Identifier: GPL-3.0-or-later
"""Idempotently add SPDX license headers to all Python source files.

Skips files that already contain `SPDX-License-Identifier`.
Preserves shebang lines and the standard `from __future__ import annotations`
ordering by inserting the header at the top while keeping the shebang first.
"""
from __future__ import annotations

import sys
from pathlib import Path

HEADER = "# SPDX-License-Identifier: GPL-3.0-or-later"


def already_tagged(text: str) -> bool:
    return "SPDX-License-Identifier" in text[:512]


def add_header(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    if already_tagged(text):
        return False
    lines = text.split("\n")
    insert_at = 0
    if lines and lines[0].startswith("#!"):
        insert_at = 1
    new_lines = lines[:insert_at] + [HEADER] + lines[insert_at:]
    path.write_text("\n".join(new_lines), encoding="utf-8")
    return True


def main(roots: list[str]) -> int:
    total = 0
    changed = 0
    for root in roots:
        for path in Path(root).rglob("*.py"):
            total += 1
            if add_header(path):
                changed += 1
    print(f"scanned={total} updated={changed}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["src", "tests"]))
