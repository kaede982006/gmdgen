# SPDX-License-Identifier: GPL-3.0-or-later
import os
from pathlib import Path

def test_release_scripts():
    scripts = [
        "tools/local_ai/verify_final_release.ps1",
        "tools/local_ai/run_local_editor.ps1",
        "tools/local_ai/create_ollama_model_alias.ps1",
        "tools/local_ai/package_release.ps1",
        "tools/local_ai/smoke_ollama_provider.ps1",
    ]
    
    for script in scripts:
        assert Path(script).exists(), f"Missing required script {script}"

    verify = Path("tools/local_ai/verify_final_release.ps1").read_text(encoding="utf-8").lower()
    assert "compileall" in verify
    assert "flake8" in verify
    assert "test_gui_ollama_only_surface.py" in verify
    assert "test_release_tree_hygiene.py" in verify
    assert "pytest" in verify
    
    alias = Path("tools/local_ai/create_ollama_model_alias.ps1").read_text(encoding="utf-8").lower()
    assert "ollama create" in alias
    assert "ollama alias create" not in alias
    
    package = Path("tools/local_ai/package_release.ps1").read_text(encoding="utf-8").lower()
    assert "gmdgen_ai_editor_source.zip" in package
    
    run_script = Path("tools/local_ai/run_local_editor.ps1").read_text(encoding="utf-8").lower()
    assert "ollama serve" in run_script
    assert "python -m gmdgen" in run_script
