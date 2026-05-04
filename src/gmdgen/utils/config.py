# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import Any


def config_to_dict_safe(config: Any) -> dict[str, Any]:
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    if dataclasses.is_dataclass(config):
        try:
            return dataclasses.asdict(config)
        except Exception:
            return {field.name: getattr(config, field.name) for field in dataclasses.fields(config)}
    if hasattr(config, "model_dump"):
        return dict(config.model_dump())
    if hasattr(config, "dict"):
        return dict(config.dict())
    if isinstance(config, tuple) and hasattr(config, "_fields") and hasattr(config, "_asdict"):
        return dict(config._asdict())
    if isinstance(config, SimpleNamespace):
        return {key: value for key, value in vars(config).items() if not key.startswith("_")}
    values: dict[str, Any] = {}
    for name in dir(config):
        if name.startswith("_"):
            continue
        try:
            value = getattr(config, name)
        except Exception:
            continue
        if callable(value):
            continue
        values[name] = value
    return values


def clone_config_with_updates(config: Any, **updates: Any) -> Any:
    if config is None:
        return dict(updates)
    if dataclasses.is_dataclass(config):
        try:
            return dataclasses.replace(config, **updates)
        except Exception:
            data = config_to_dict_safe(config)
            data.update(updates)
            return type(config)(**data)
    if hasattr(config, "model_copy"):
        return config.model_copy(update=updates)
    if hasattr(config, "copy"):
        try:
            return config.copy(update=updates)
        except TypeError:
            pass
    if isinstance(config, tuple) and hasattr(config, "_replace"):
        return config._replace(**updates)
    if isinstance(config, dict):
        data = dict(config)
        data.update(updates)
        return data
    data = config_to_dict_safe(config)
    data.update(updates)
    try:
        return type(config)(**data)
    except Exception:
        return data
