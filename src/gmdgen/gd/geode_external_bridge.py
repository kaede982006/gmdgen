import json
import subprocess
from typing import Any

from gmdgen.gd.geode_protocol import GeodeBridgeConfig, GeodeIPCRequest, GeodeIPCResponse

class ExternalGeodeBridge:
    def __init__(self, config: GeodeBridgeConfig):
        self.config = config

    def _send_request(self, command: str, args: dict[str, Any]) -> GeodeIPCResponse:
        if not self.config.enabled or not self.config.helper_executable_path:
            return GeodeIPCResponse(ok=False, errors=["Geode helper not configured or disabled"])
            
        req = GeodeIPCRequest(command=command, args=args)
        req_json = json.dumps(req.to_dict())
        
        try:
            # We mock the subprocess if testing
            proc = subprocess.run(
                [self.config.helper_executable_path],
                input=req_json,
                capture_output=True,
                text=True,
                timeout=self.config.timeout_seconds,
                cwd=self.config.working_dir or None
            )
            
            if proc.returncode != 0:
                return GeodeIPCResponse(ok=False, errors=[f"Helper exited with code {proc.returncode}", proc.stderr])
                
            res_data = json.loads(proc.stdout)
            return GeodeIPCResponse.from_dict(res_data)
            
        except subprocess.TimeoutExpired:
            return GeodeIPCResponse(ok=False, errors=[f"Timeout after {self.config.timeout_seconds}s"])
        except json.JSONDecodeError:
            return GeodeIPCResponse(ok=False, errors=["Invalid JSON response from helper"])
        except Exception as e:
            return GeodeIPCResponse(ok=False, errors=[f"Unexpected error: {str(e)}"])

    def get_version(self) -> str:
        res = self._send_request("get_version", {})
        return res.geode_version if res.ok else "unknown"

    def pos_for_time(self, time: float) -> float:
        res = self._send_request("pos_for_time", {"time": time})
        return res.result.get("pos", 0.0) if res.ok else 0.0
