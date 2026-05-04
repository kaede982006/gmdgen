import json
from gmdgen.gd.geode_protocol import GeodeIPCRequest, GeodeIPCResponse, GeodeBridgeConfig
from gmdgen.gd.geode_external_bridge import ExternalGeodeBridge

def test_geode_protocol_request_serializes():
    req = GeodeIPCRequest(command="pos_for_time", args={"time": 10.0})
    data = req.to_dict()
    assert data["command"] == "pos_for_time"
    assert data["args"]["time"] == 10.0

def test_geode_protocol_response_parses():
    data = {"ok": True, "result": {"pos": 500.0}, "geode_version": "v3.0.0"}
    res = GeodeIPCResponse.from_dict(data)
    assert res.ok is True
    assert res.result["pos"] == 500.0
    assert res.geode_version == "v3.0.0"

def test_external_geode_bridge_disabled():
    config = GeodeBridgeConfig(enabled=False)
    bridge = ExternalGeodeBridge(config)
    assert bridge.get_version() == "unknown"

def test_external_geode_bridge_invalid_json_reported(monkeypatch):
    import subprocess
    
    class MockCompletedProcess:
        returncode = 0
        stdout = "invalid json"
        stderr = ""

    def mock_run(*args, **kwargs):
        return MockCompletedProcess()
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    config = GeodeBridgeConfig(enabled=True, helper_executable_path="dummy")
    bridge = ExternalGeodeBridge(config)
    res = bridge._send_request("get_version", {})
    assert res.ok is False
    assert "Invalid JSON" in res.errors[0]

def test_external_geode_bridge_timeout_reported(monkeypatch):
    import subprocess
    
    def mock_run(*args, **kwargs):
        raise subprocess.TimeoutExpired("dummy", 1.0)
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    config = GeodeBridgeConfig(enabled=True, helper_executable_path="dummy", timeout_seconds=1.0)
    bridge = ExternalGeodeBridge(config)
    res = bridge._send_request("get_version", {})
    assert res.ok is False
    assert "Timeout" in res.errors[0]

def test_external_geode_bridge_mock_success(monkeypatch):
    import subprocess
    
    class MockCompletedProcess:
        returncode = 0
        stdout = '{"ok": true, "result": {"pos": 100.0}, "geode_version": "1.0"}'
        stderr = ""

    def mock_run(*args, **kwargs):
        return MockCompletedProcess()
        
    monkeypatch.setattr(subprocess, "run", mock_run)
    config = GeodeBridgeConfig(enabled=True, helper_executable_path="dummy")
    bridge = ExternalGeodeBridge(config)
    
    assert bridge.get_version() == "1.0"
    assert bridge.pos_for_time(1.0) == 100.0
