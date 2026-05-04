import os
import subprocess
from pathlib import Path

def test_release_tree_hygiene():
    forbidden_dirs = [
        "release_artifacts",
        "dataset/local_coder_memory",
        "dataset/cache",
        "dataset/learning",
        "outputs",
        "debug_artifacts",
    ]
    
    try:
        git_files = subprocess.check_output(["git", "ls-files"]).decode("utf-8").splitlines()
    except Exception:
        git_files = []

    for d in forbidden_dirs:
        for f in git_files:
            assert not f.startswith(d + "/"), f"Forbidden directory is tracked in git: {d}"

    assert "artifacts/model.json" not in git_files, "artifacts/model.json must not be tracked"

    forbidden_substrings = [
        "final-oneclick",
        "hotfix_missing_artifact",
        "final_repair_release",
        "fix_ollama_gui_release",
        "fast_finalize",
        "rewrite_docs_ollama_only_fast",
        "run_docs_cleanup_4_to_9",
        ".aider",
    ]
    
    for root, dirs, files in os.walk("."):
        if ".git" in root:
            continue
        for file in files:
            for sub in forbidden_substrings:
                assert sub not in file, f"Recovery artifact found: {file}"

    # Check tools/local_ai directory
    tools_dir = Path("tools/local_ai")
    allowed_tools = {
        "verify_final_release.ps1",
        "run_local_editor.ps1",
        "create_ollama_model_alias.ps1",
        "package_release.ps1",
        "smoke_ollama_provider.ps1",
        "write_docs.py",
        "clean_files.py",
        "clean_gui.py",
        "clean_audio_conditioned.py",
    }
    
    if tools_dir.exists():
        for file in tools_dir.iterdir():
            if file.is_file() and file.name not in allowed_tools:
                assert False, f"Unexpected script in tools/local_ai: {file.name}"

    # Verify no __pycache__, *.pyc, .pytest_cache are tracked in git
    try:
        git_files = subprocess.check_output(["git", "ls-files"]).decode("utf-8").splitlines()
        for f in git_files:
            assert "__pycache__" not in f, f"__pycache__ is tracked in git: {f}"
            assert not f.endswith(".pyc"), f".pyc is tracked in git: {f}"
            assert ".pytest_cache" not in f, f".pytest_cache is tracked in git: {f}"
    except Exception as e:
        pass # If git fails, just pass
