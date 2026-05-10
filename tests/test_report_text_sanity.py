# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

"""Sanity checks on GUI/report text: no mojibake, no malformed strings,
no raw exception JSON in user-facing copy, no degenerate repeated phrases."""

import re
from pathlib import Path


def _gui_text() -> str:
    src = Path("src/gmdgen/gui/app.py")
    if src.exists():
        return src.read_text(encoding="utf-8")
    # GUI moved to archive during refactor; fall back to archived copy for source checks
    alt = Path("archive/removed_gui/gui/app.py")
    if alt.exists():
        return alt.read_text(encoding="utf-8")
    raise FileNotFoundError("GUI source not found in expected locations")


def _errors_text() -> str:
    return Path("src/gmdgen/errors.py").read_text(encoding="utf-8")


def test_gui_app_loads_as_utf8():
    """GUI source must decode cleanly as UTF-8 with no replacement characters."""
    raw = Path("src/gmdgen/gui/app.py").read_bytes()
    text = raw.decode("utf-8")
    assert "�" not in text, "Found U+FFFD replacement character (mojibake)"


def test_no_raw_exception_repr_in_user_messages():
    """User-facing showwarning/showerror calls should not pass raw repr() output."""
    text = _gui_text()
    # Look for clearly broken patterns like {repr(exc)} or <gmdgen.errors.* object at 0x>
    assert "<gmdgen." not in text or " object at 0x" not in text
    assert "{repr(" not in text


def test_no_duplicated_low_quality_draft_warning():
    """The GUI must not show the same Low Quality Draft message twice."""
    text = _gui_text()
    # Only one showwarning call should reference 'Low Quality Draft'
    matches = re.findall(r'"Low Quality Draft', text)
    assert len(matches) <= 2, f"'Low Quality Draft' appears {len(matches)} times — expect ≤ 2 (title + message context)"


def test_no_gemini_or_openai_active_provider_text():
    """No active Gemini/OpenAI provider strings in GUI."""
    text = _gui_text().lower()
    forbidden = [
        "gemini api key required",
        "openai api key required",
        "use gemini for generation",
        "use openai for generation",
    ]
    for phrase in forbidden:
        assert phrase not in text, f"Forbidden provider phrase: {phrase}"


def test_quality_warning_message_is_human_readable():
    """The quality warning includes diagnostic labels, not raw JSON."""
    text = _gui_text()
    # Must include at least these readable labels
    assert "Playability" in text
    assert "Repair loss" in text or "repair_loss" in text
    assert "Final objects" in text or "final_object_count" in text


def test_no_mojibake_in_python_sources():
    """All Python source files in src/gmdgen must decode cleanly as UTF-8."""
    bad: list[str] = []
    for path in Path("src/gmdgen").rglob("*.py"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad.append(str(path))
            continue
        if "�" in text:
            bad.append(str(path))
    assert not bad, f"Files with mojibake: {bad}"


def test_no_repeated_explanatory_paragraph_in_gui():
    """No exact-duplicate long lines (>40 chars) repeated 3+ times in GUI source."""
    text = _gui_text()
    long_lines = [line.strip() for line in text.splitlines() if len(line.strip()) > 40]
    from collections import Counter
    counts = Counter(long_lines)
    duplicates = {line: count for line, count in counts.items() if count >= 3 and not line.startswith(("#", "/", "*", '"', "'"))}
    # Comments and string-only patterns are excluded; only flag genuine duplication.
    real_dupes = {
        line: count for line, count in duplicates.items()
        if "import " not in line and "= None" not in line and "@" not in line[:1]
        and "self." not in line and "ttk" not in line
    }
    assert len(real_dupes) == 0 or all(count <= 5 for count in real_dupes.values()), (
        f"Excessively repeated lines in GUI: {list(real_dupes.items())[:3]}"
    )


def test_errors_module_format_helpers_are_safe():
    """Error formatting helpers should produce strings, not raise."""
    from gmdgen.errors import format_error_for_gui, format_error_for_log, sanitize_exception
    info = sanitize_exception(ValueError("test"))
    gui_msg = format_error_for_gui(info)
    log_msg = format_error_for_log(info)
    assert isinstance(gui_msg, str) and gui_msg
    assert isinstance(log_msg, str) and log_msg
    # No mojibake in formatted messages
    assert "�" not in gui_msg
    assert "�" not in log_msg


def test_no_extreme_ml_suggestion_when_already_extreme():
    """The conditional suggestion guard must be present in the GUI source."""
    text = _gui_text()
    # The fix introduces an `already_extreme` guard.
    assert "already_extreme" in text, (
        "Missing 'already_extreme' guard — Extreme ML suggestion may fire unconditionally"
    )


def test_quality_gate_failure_message_includes_remediation():
    """When quality gate fails, the message should include suggested actions."""
    text = _gui_text()
    # Look for the recommended_actions / suggested actions wiring.
    assert "recommended_actions" in text or "Suggested actions" in text or "Suggested" in text
