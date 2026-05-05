# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Lightweight progress bar with no external dependency."""
from __future__ import annotations

import os
import sys
import time
from contextlib import contextmanager
from typing import Iterator


class ProgressBar:
    """Tiny stderr progress bar.

    Usage:
        with ProgressBar(total=100, desc="layout") as bar:
            for _ in range(100):
                bar.update(1)
    """

    def __init__(self, total: int, desc: str = "", unit: str = "it", width: int = 24) -> None:
        self.total = max(1, total)
        self.desc = desc
        self.unit = unit
        self.width = width
        self.n = 0
        self.start = time.time()
        self._silent = (
            os.environ.get("GMDGEN_LOG_LEVEL", "1") == "0"
            or os.environ.get("GMDGEN_NO_PROGRESS") == "1"
            or not sys.stderr.isatty()
        )

    def update(self, k: int = 1) -> None:
        self.n = min(self.total, self.n + k)
        if not self._silent:
            self._render(end="\r")

    def _render(self, end: str = "\r") -> None:
        pct = self.n / self.total
        filled = int(pct * self.width)
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = time.time() - self.start
        rate = self.n / elapsed if elapsed > 0 else 0
        sys.stderr.write(
            f"  {self.desc:<10} {pct*100:5.1f}% [{bar}] {self.n}/{self.total} {self.unit} ({rate:.0f}/s){end}"
        )
        sys.stderr.flush()

    def close(self) -> None:
        if not self._silent:
            self._render(end="\n")

    def __enter__(self) -> "ProgressBar":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


@contextmanager
def progress(total: int, desc: str = "", unit: str = "it") -> Iterator[ProgressBar]:
    bar = ProgressBar(total=total, desc=desc, unit=unit)
    try:
        yield bar
    finally:
        bar.close()
