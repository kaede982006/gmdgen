from __future__ import annotations

import ast
import inspect
from pathlib import Path
from dataclasses import fields, is_dataclass

from gmdgen.gui.app import GuiGenerationConfig


def test_gui_generation_config_constructor_contract() -> None:
    """
    Verifies that all GuiGenerationConfig(...) calls in app.py match the actual fields.
    Also ensures use_environment_key and v_use_environment_key are gone.
    """
    # 1. Get accepted fields
    if is_dataclass(GuiGenerationConfig):
        accepted_fields = {f.name for f in fields(GuiGenerationConfig)}
    else:
        sig = inspect.signature(GuiGenerationConfig.__init__)
        accepted_fields = {p.name for p in sig.parameters.values() if p.name != "self"}

    app_path = Path("src/gmdgen/gui/app.py")
    tree = ast.parse(app_path.read_text(encoding="utf-8"))

    found_calls = 0
    for node in ast.walk(tree):
        # Check for GuiGenerationConfig(...)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "GuiGenerationConfig":
            found_calls += 1
            for keyword in node.keywords:
                arg_name = keyword.arg
                assert arg_name is not None, "Implicit keyword expansion (**) is not allowed in GuiGenerationConfig calls in app.py for this audit."
                
                # Check if the argument exists in the dataclass
                assert arg_name in accepted_fields, f"GuiGenerationConfig() call uses unknown keyword: {arg_name}"
                
                # Specifically check for banned keys
                assert arg_name != "use_environment_key", "Found banned keyword 'use_environment_key' in GuiGenerationConfig call"
                assert arg_name != "v_use_environment_key", "Found banned keyword 'v_use_environment_key' in GuiGenerationConfig call"

    assert found_calls > 0, "Could not find any GuiGenerationConfig calls in app.py"

def test_no_legacy_variable_usage_in_app_py() -> None:
    """
    Ensures v_use_environment_key is not defined or used in app.py.
    """
    app_path = Path("src/gmdgen/gui/app.py")
    content = app_path.read_text(encoding="utf-8")
    
    assert "v_use_environment_key" not in content, "Found 'v_use_environment_key' in app.py"
    assert "use_environment_key" not in content or "use_ollama_environment_key" in content, "Found potential 'use_environment_key' without 'ollama' prefix"
    
    # More specific check for the keyword argument
    assert "use_environment_key=" not in content, "Found 'use_environment_key=' keyword argument in app.py"

def test_ollama_base_url_consistency() -> None:
    """
    Ensures ollama_base_url is used instead of ollama_api_key in GuiGenerationConfig.
    """
    if is_dataclass(GuiGenerationConfig):
        accepted_fields = {f.name for f in fields(GuiGenerationConfig)}
    else:
        sig = inspect.signature(GuiGenerationConfig.__init__)
        accepted_fields = {p.name for p in sig.parameters.values() if p.name != "self"}

    assert "ollama_base_url" in accepted_fields
    assert "ollama_api_key" not in accepted_fields
