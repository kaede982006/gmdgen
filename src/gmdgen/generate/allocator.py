# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field

from gmdgen.generate.ir import ColorSymbol, GroupSymbol, LevelIR


@dataclass(slots=True)
class SymbolAllocationReport:
    group_ids: dict[str, int] = field(default_factory=dict)
    color_channel_ids: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict[str, object]:
        return {
            "group_ids": dict(self.group_ids),
            "color_channel_ids": dict(self.color_channel_ids),
            "errors": list(self.errors),
            "passed": self.passed,
        }


def allocate_symbols(
    level_ir: LevelIR,
    *,
    first_group_id: int = 1,
    first_color_channel_id: int = 1,
    max_group_id: int = 9999,
    max_color_channel_id: int = 999,
) -> SymbolAllocationReport:
    """Assign concrete ids to symbolic references inside local IR."""
    report = SymbolAllocationReport()
    next_group_id = first_group_id
    next_color_id = first_color_channel_id

    for group in _iter_group_symbols(level_ir):
        if group.name not in report.group_ids:
            if next_group_id > max_group_id:
                report.errors.append("group_id_budget_exceeded")
                break
            report.group_ids[group.name] = next_group_id
            next_group_id += 1

    for color in _iter_color_symbols(level_ir):
        if color.name not in report.color_channel_ids:
            if next_color_id > max_color_channel_id:
                report.errors.append("color_channel_budget_exceeded")
                break
            report.color_channel_ids[color.name] = next_color_id
            next_color_id += 1

    for section in level_ir.sections:
        for obj in section.objects:
            obj.group_ids = [
                report.group_ids[symbol.name]
                for symbol in obj.group_symbols
                if symbol.name in report.group_ids
            ]
            if obj.color_symbol is not None:
                obj.color_channel_id = report.color_channel_ids.get(obj.color_symbol.name)
        for trigger in section.triggers:
            if trigger.target_group_symbol is not None:
                trigger.target_group_id = report.group_ids.get(trigger.target_group_symbol.name)
            if trigger.color_symbol is not None:
                trigger.color_channel_id = report.color_channel_ids.get(trigger.color_symbol.name)

    return report


def _iter_group_symbols(level_ir: LevelIR) -> list[GroupSymbol]:
    symbols: list[GroupSymbol] = []
    for section in level_ir.sections:
        symbols.extend(section.group_symbols)
        for obj in section.objects:
            symbols.extend(obj.group_symbols)
        for trigger in section.triggers:
            if trigger.target_group_symbol is not None:
                symbols.append(trigger.target_group_symbol)
    return symbols


def _iter_color_symbols(level_ir: LevelIR) -> list[ColorSymbol]:
    symbols: list[ColorSymbol] = []
    for section in level_ir.sections:
        symbols.extend(section.color_symbols)
        for obj in section.objects:
            if obj.color_symbol is not None:
                symbols.append(obj.color_symbol)
        for trigger in section.triggers:
            if trigger.color_symbol is not None:
                symbols.append(trigger.color_symbol)
    return symbols
