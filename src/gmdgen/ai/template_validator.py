import re
from typing import Any

from gmdgen.errors import PromptBuildError

class PromptTemplateValidator:
    PLACEHOLDER_PATTERN = re.compile(r"\{([A-Za-z0-9_]+)\}")

    @classmethod
    def scan_unresolved_placeholders(cls, text: str) -> list[str]:
        return cls.PLACEHOLDER_PATTERN.findall(text)

    @classmethod
    def assert_prompt_has_no_internal_debug_leak(cls, text: str) -> None:
        if "sk-" in text and len(text) > 40:
            raise PromptBuildError("API key detected in prompt.")
        if "C:\\Users\\" in text or "/home/" in text:
            raise PromptBuildError("Absolute path detected in prompt.")

    @classmethod
    def safe_format_template(cls, template: str, data: dict[str, Any]) -> str:
        # Check if any required key is missing
        required_keys = cls.scan_unresolved_placeholders(template)
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise PromptBuildError(f"Missing required format keys: {missing_keys}")
        try:
            formatted = template.format(**data)
            cls.assert_prompt_has_no_internal_debug_leak(formatted)
            return formatted
        except KeyError as e:
            raise PromptBuildError(f"Failed to format template, missing key: {e}")
        except ValueError as e:
            raise PromptBuildError(f"Failed to format template, value error: {e}")
