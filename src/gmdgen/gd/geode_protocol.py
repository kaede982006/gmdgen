from dataclasses import dataclass, field
from typing import Any

@dataclass
class GeodeBridgeConfig:
    enabled: bool = False
    helper_executable_path: str = ""
    protocol_version: str = "1.0"
    timeout_seconds: float = 5.0
    working_dir: str = ""
    use_temp_files: bool = False
    log_path: str = ""

@dataclass
class GeodeIPCRequest:
    command: str
    args: dict[str, Any] = field(default_factory=dict)
    request_id: str = "0"

    def to_dict(self) -> dict[str, Any]:
        return {"command": self.command, "args": self.args, "request_id": self.request_id}

@dataclass
class GeodeIPCResponse:
    ok: bool
    result: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    geode_version: str = ""
    protocol_version: str = ""
    request_id: str = "0"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GeodeIPCResponse":
        return cls(
            ok=data.get("ok", False),
            result=data.get("result", {}),
            warnings=data.get("warnings", []),
            errors=data.get("errors", []),
            geode_version=data.get("geode_version", ""),
            protocol_version=data.get("protocol_version", ""),
            request_id=data.get("request_id", "0")
        )
