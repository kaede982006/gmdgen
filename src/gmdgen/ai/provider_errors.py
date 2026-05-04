from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

@dataclass(slots=True)
class AIProviderErrorInfo:
    provider_name: str
    code: str
    message: str
    recoverable: bool = False
    provider_exhausted: bool = False
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

class AIProviderError(RuntimeError):
    def __init__(self, info: AIProviderErrorInfo) -> None:
        self.info = info
        super().__init__(f"{info.code}: {info.message}")

def sanitize_provider_error(text: str) -> str:
    value = str(text)
    value = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-[REDACTED]", value)
    value = re.sub(r"AIza[A-Za-z0-9_\-]{12,}", "AIza[REDACTED]", value)
    return value
