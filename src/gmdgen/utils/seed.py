# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import random


def set_global_seed(seed: int) -> None:
    random.seed(seed)
