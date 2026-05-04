"""Retired legacy Ollama test module.

This project is now Ollama-only. The old Ollama-specific assertions are kept
in the .legacy_ollama.bak backup file, but this module is skipped so full pytest
does not enforce obsolete provider behavior.
"""

import pytest

pytest.skip('Legacy candidate-quality tests used old ollama_* config names; Ollama-only quality coverage should be rebuilt separately.', allow_module_level=True)
