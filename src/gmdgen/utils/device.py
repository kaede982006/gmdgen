# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 kaede982006
from __future__ import annotations

import functools
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    import torch
except ImportError:
    torch = None


def dataclass_slots_if_supported(cls):
    from dataclasses import dataclass
    try:
        return dataclass(slots=True)(cls)
    except TypeError:
        return dataclass(cls)


@dataclass_slots_if_supported
class DeviceInfo:
    compute_device: str
    gpu_available: bool
    gpu_name: str | None
    gpu_backend: str | None
    torch_available: bool
    cuda_available: bool
    mps_available: bool
    fallback_reason: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "compute_device": self.compute_device,
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "gpu_backend": self.gpu_backend,
            "torch_available": self.torch_available,
            "cuda_available": self.cuda_available,
            "mps_available": self.mps_available,
            "gpu_fallback_reason": self.fallback_reason,
        }


@functools.lru_cache(maxsize=1)
def get_device_info() -> DeviceInfo:
    """Detect the best available compute device and return detailed info."""
    torch_available = torch is not None
    cuda_available = False
    mps_available = False
    gpu_available = False
    gpu_name = None
    gpu_backend = None
    compute_device = "cpu"
    fallback_reason = None

    if not torch_available:
        return DeviceInfo(
            compute_device="cpu",
            gpu_available=False,
            gpu_name=None,
            gpu_backend=None,
            torch_available=False,
            cuda_available=False,
            mps_available=False,
            fallback_reason="torch_not_installed",
        )

    if torch.cuda.is_available():
        cuda_available = True
        gpu_available = True
        compute_device = "cuda"
        gpu_backend = "cuda"
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "unknown_cuda_device"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        mps_available = True
        gpu_available = True
        compute_device = "mps"
        gpu_backend = "mps"
        gpu_name = "Apple Silicon GPU"
    else:
        fallback_reason = "no_gpu_detected"

    return DeviceInfo(
        compute_device=compute_device,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_backend=gpu_backend,
        torch_available=True,
        cuda_available=cuda_available,
        mps_available=mps_available,
        fallback_reason=fallback_reason,
    )


def get_best_device() -> Any:
    """Return the torch device object for the best available hardware."""
    if torch is None:
        return "cpu"
    info = get_device_info()
    return torch.device(info.compute_device)


def apply_device_info_to_report(report: Any) -> None:
    """Populate a report object (or dict) with current device information."""
    info = get_device_info()
    data = info.to_dict()
    if isinstance(report, dict):
        report.update(data)
    else:
        for k, v in data.items():
            if hasattr(report, k):
                setattr(report, k, v)
