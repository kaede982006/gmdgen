from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmdgen.errors import GmdgenError
from gmdgen.io.gmd_writer import write_gmd_file

logger = logging.getLogger(__name__)


class SaveError(GmdgenError):
    """Raised when level saving fails."""
    pass


@dataclass(slots=True)
class SaveResult:
    requested_output_path: str
    resolved_output_path: str = ""
    output_dir: str = ""
    file_name: str = ""
    file_exists: bool = False
    file_size_bytes: int = 0
    bytes_written: int = 0
    level_string_length: int = 0
    validation_report_path: str = ""
    quality_report_path: str = ""
    output_bundle_path: str = ""
    saved_as_draft: bool = False
    quality_gate_passed: bool = True
    success: bool = False
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_output_path": self.requested_output_path,
            "resolved_output_path": self.resolved_output_path,
            "output_dir": self.output_dir,
            "file_name": self.file_name,
            "file_exists": self.file_exists,
            "file_size_bytes": self.file_size_bytes,
            "bytes_written": self.bytes_written,
            "level_string_length": self.level_string_length,
            "validation_report_path": self.validation_report_path,
            "quality_report_path": self.quality_report_path,
            "output_bundle_path": self.output_bundle_path,
            "saved_as_draft": self.saved_as_draft,
            "quality_gate_passed": self.quality_gate_passed,
            "success": self.success,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def sanitize_windows_filename(name: str) -> str:
    """Remove or replace characters that are invalid in Windows filenames."""
    if not name:
        return "unnamed_level"
    name = re.sub(r'[<>:"/\\|?*]', "_", name)
    name = name.strip(" .")
    if not name:
        return "unnamed_level"
    return name


def resolve_output_path(output_path: str | Path | None, default_name: str = "generated_level", extension: str = ".gmd") -> Path:
    """Resolve the requested output path to a safe absolute Path."""
    if not output_path:
        return Path("outputs") / f"{sanitize_windows_filename(default_name)}{extension}"
    
    path = Path(str(output_path).strip())
    if str(path) == "." or path.is_dir() or str(path).endswith("/") or str(path).endswith("\\"):
        return path / f"{sanitize_windows_filename(default_name)}{extension}"
        
    if not path.suffix:
        path = path.with_suffix(extension)
        
    # Sanitize the filename part of the path
    safe_name = sanitize_windows_filename(path.stem)
    return path.parent / f"{safe_name}{path.suffix}"


def verify_written_file(path: Path, min_size: int = 1) -> bool:
    try:
        return path.exists() and path.stat().st_size >= min_size
    except Exception:
        return False


def save_level_output(
    encoded_level_data: str,
    output_path: str | Path | None,
    tags: dict[str, tuple[str, str]],
    *,
    is_draft: bool = False,
    quality_gate_passed: bool = True,
    default_name: str = "generated_level",
) -> SaveResult:
    """
    Unified save function for generated levels.
    """
    result = SaveResult(
        requested_output_path=str(output_path) if output_path is not None else "",
        saved_as_draft=is_draft,
        quality_gate_passed=quality_gate_passed,
    )
    
    if encoded_level_data is None:
        result.errors.append("Encoded level data is None")
        return result
        
    level_string = encoded_level_data.strip()
    result.level_string_length = len(level_string)
    
    if not level_string:
        result.errors.append("Encoded level data is empty")
        return result
        
    try:
        resolved_path = resolve_output_path(output_path, default_name=default_name)
        resolved_path = resolved_path.resolve()
        
        result.resolved_output_path = str(resolved_path)
        result.output_dir = str(resolved_path.parent)
        result.file_name = resolved_path.name
        
        # Ensure output directory exists
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write file
        write_gmd_file(resolved_path, tags)
        
        # Verify file
        if verify_written_file(resolved_path):
            result.file_exists = True
            result.file_size_bytes = resolved_path.stat().st_size
            result.bytes_written = result.file_size_bytes
            result.success = True
        else:
            result.errors.append(f"Verification failed: file {resolved_path} may not exist or is empty")
            
    except Exception as e:
        result.errors.append(f"Write error: {str(e)}")
        logger.error(f"Failed to save level output: {e}", exc_info=True)
        
    return result
