from __future__ import annotations

from pathlib import Path
from typing import Any

FALLBACK_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav"})
EXTENDED_AUDIO_EXTENSIONS: frozenset[str] = frozenset({".wav", ".mp3", ".ogg", ".flac"})


def normalize_audio_file_path(audio_file: str | Path | None) -> Path | None:
    """Return a concrete audio path or None when the user did not request audio.

    Empty strings are treated as "not provided" so the existing style-only path
    remains intact. Non-empty invalid paths fail loudly; audio requests must not
    silently fall back to style-only generation.
    """

    if audio_file is None:
        return None
    raw = str(audio_file).strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"audio_file does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"audio_file must be a file, not a directory: {path}")
    return path


def librosa_or_audio_backend_available() -> bool:
    try:
        import librosa  # type: ignore[import-not-found]  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def supported_audio_extensions(backend: str | None = None) -> frozenset[str]:
    backend_name = str(backend or "auto").strip().lower()
    if backend_name in {"fallback", "fallback_wav", "wav", "builtin"}:
        return FALLBACK_AUDIO_EXTENSIONS
    if backend_name == "librosa":
        return EXTENDED_AUDIO_EXTENSIONS
    if librosa_or_audio_backend_available():
        return EXTENDED_AUDIO_EXTENSIONS
    return FALLBACK_AUDIO_EXTENSIONS


def validate_audio_file_extension(path: Path, backend: str | None = None) -> None:
    suffix = path.suffix.lower()
    supported = supported_audio_extensions(backend)
    if suffix not in supported:
        raise ValueError(
            f"unsupported audio file extension: {suffix or '<none>'}. "
            f"supported extensions: {sorted(supported)}"
        )


def resolve_audio_file_from_config(config: dict[str, Any]) -> Path | None:
    audio_path = normalize_audio_file_path(config.get("audio_file"))
    if audio_path is None:
        return None
    validate_audio_file_extension(audio_path, config.get("audio_backend", "auto"))
    return audio_path
