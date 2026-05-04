from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

from gmdgen.utils.config import clone_config_with_updates, config_to_dict_safe


@dataclass(slots=True)
class SlotsConfig:
    name: str
    rebuild: bool = False


@dataclass
class NormalConfig:
    name: str
    rebuild: bool = False


class TupleConfig(NamedTuple):
    name: str
    rebuild: bool = False


def test_clone_config_with_slots_dataclass() -> None:
    config = SlotsConfig(name="context")

    cloned = clone_config_with_updates(config, rebuild=True)

    assert isinstance(cloned, SlotsConfig)
    assert cloned.name == "context"
    assert cloned.rebuild is True
    assert config.rebuild is False


def test_clone_config_with_normal_dataclass() -> None:
    config = NormalConfig(name="context")

    cloned = clone_config_with_updates(config, rebuild=True)

    assert isinstance(cloned, NormalConfig)
    assert cloned.rebuild is True


def test_clone_config_with_dict() -> None:
    cloned = clone_config_with_updates({"name": "context", "rebuild": False}, rebuild=True)

    assert cloned == {"name": "context", "rebuild": True}


def test_clone_config_with_namedtuple() -> None:
    config = TupleConfig(name="context")

    cloned = clone_config_with_updates(config, rebuild=True)

    assert isinstance(cloned, TupleConfig)
    assert cloned.rebuild is True


def test_config_to_dict_safe_does_not_require_dunder_dict() -> None:
    config = SlotsConfig(name="context", rebuild=True)

    payload = config_to_dict_safe(config)

    assert payload == {"name": "context", "rebuild": True}


def test_config_to_dict_safe_simple_namespace() -> None:
    payload = config_to_dict_safe(SimpleNamespace(name="context", rebuild=False))

    assert payload == {"name": "context", "rebuild": False}
