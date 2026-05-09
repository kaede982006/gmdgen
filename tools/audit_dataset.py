# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
"""Audit the GMD training dataset.

Produces a small JSON report covering:
  * total files / loaded records
  * per-mode object frequency (cube/ship/.../spider) — coarse heuristic
  * per-class distribution (structure/decoration/trigger/portal/special)
  * length distribution (object count percentiles)
  * top-K object id frequencies

This addresses Onuoha + Smith&Rustagi's point that an unbalanced training set
silently bakes in distributional bias. Run before / after augmentation.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from gmdgen.data.loader import load_dataset_with_report
from gmdgen.data.preprocess import split_level_objects
from gmdgen.features.tokenizer import extract_object_id
from gmdgen.representation.object_classifier import classify

_MODE_PORTAL = {
    "12": "cube", "13": "ship", "47": "ball", "111": "ufo",
    "660": "wave", "745": "robot", "1331": "spider",
}


def audit(dataset_dir: Path) -> dict:
    res = load_dataset_with_report(dataset_dir)
    records = res.records

    total_objects = 0
    object_lengths: list[int] = []
    id_counter: Counter[str] = Counter()
    cls_counter: Counter[str] = Counter()
    mode_first_counter: Counter[str] = Counter()
    mode_present_counter: Counter[str] = Counter()

    for r in records:
        objs = split_level_objects(r.decoded_level_data)
        n = 0
        first_mode = "cube"
        modes_seen: set[str] = set()
        for obj in objs:
            oid = extract_object_id(obj)
            if not oid:
                continue
            n += 1
            id_counter[oid] += 1
            cls_counter[classify(oid).value] += 1
            if oid in _MODE_PORTAL:
                modes_seen.add(_MODE_PORTAL[oid])
                if first_mode == "cube" and not modes_seen.intersection({"ship",
                        "ball", "ufo", "wave", "robot", "spider"}):
                    first_mode = _MODE_PORTAL[oid]
        if n > 0:
            object_lengths.append(n)
            total_objects += n
            mode_first_counter[first_mode] += 1
            for m in (modes_seen or {"cube"}):
                mode_present_counter[m] += 1

    def _percentiles(xs: list[int]) -> dict:
        if not xs:
            return {"min": 0, "p50": 0, "p90": 0, "max": 0, "mean": 0}
        xs = sorted(xs)
        return {
            "min": xs[0],
            "p50": xs[len(xs) // 2],
            "p90": xs[max(0, int(len(xs) * 0.9) - 1)],
            "max": xs[-1],
            "mean": float(statistics.fmean(xs)),
        }

    return {
        "dataset_dir": str(dataset_dir),
        "files_scanned": res.report.files_scanned,
        "loaded_records": res.report.loaded_records,
        "skipped": {
            "missing_k4": res.report.skipped_missing_k4,
            "parse_failed": res.report.skipped_parse_failed,
            "decode_failed": res.report.skipped_decode_failed,
        },
        "total_objects": total_objects,
        "object_count_per_level": _percentiles(object_lengths),
        "top_ids": id_counter.most_common(30),
        "class_distribution": dict(cls_counter),
        "mode_first_distribution": dict(mode_first_counter),
        "mode_present_distribution": dict(mode_present_counter),
    }


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Audit the gmdgen .gmd dataset.")
    p.add_argument("--in", dest="in_dir", required=True, type=Path)
    p.add_argument("--out", type=Path, default=Path("reports/dataset_audit.json"))
    return p


def main(argv: list[str] | None = None) -> int:
    args = _argparser().parse_args(argv)
    report = audit(args.in_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
