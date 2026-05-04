import pytest

pytestmark = pytest.mark.skip(
    reason="retired legacy external-provider test after Ollama-only migration"
)


def test_retired_legacy_external_provider_test():
    pass
