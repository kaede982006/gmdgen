python -m compileall .\src\gmdgen .\tests
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m flake8 --select=E9,F821,F823,F831,F406,F407,F701,F702,F704,F706 --show-source --isolated src tests
if ($LASTEXITCODE -ne 0) { exit 1 }

python -c "import gmdgen; print('import ok')"
if ($LASTEXITCODE -ne 0) { exit 1 }

python -c "import gmdgen.gui.app; print('gui import ok')"
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m pytest -q tests\test_gui_ollama_only_surface.py tests\test_docs_ollama_final.py tests\test_release_scripts.py tests\test_release_tree_hygiene.py
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m pytest -q tests\test_audio_file_inputs.py::test_empty_audio_file_uses_style_only tests\test_audio_regression.py::test_audio_conditioned_pipeline_uses_audio_mode tests\test_generation_report.py::test_generation_report_contains_mode
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m pytest -q tests\test_ollama_provider.py tests\test_ollama_only_policy.py
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m pytest -q tests\test_gui_worker.py tests\test_quality_gate.py tests\test_output_bundle.py
if ($LASTEXITCODE -ne 0) { exit 1 }

python -m pytest -q
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "All verification steps passed!"
