# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import importlib.util
import json
import os
import re
import shutil
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from gmdgen.ai.dataset_index import resolve_dataset_dir
from gmdgen.ai.training import AutoTrainingConfig, AutoTrainingResult, run_auto_training
from gmdgen.ai.training import rebuild_auto_training_config
from gmdgen.errors import (
    QualityGateFailure,
    format_error_for_gui,
    format_error_for_log,
    sanitize_exception,
)
from gmdgen.generate.generator import generate_from_config
from gmdgen.learning.store import (
    build_learning_example_from_result,
    save_learning_example,
    summarize_learning_examples_for_context,
    update_learning_example_feedback,
)
from gmdgen.learning.dataset_memory import (
    dataset_learned_data_store_dir,
    dataset_learning_examples_dir,
    save_learning_example_to_dataset,
    update_dataset_feedback,
)
from gmdgen.learning.feature_extractor import (
    clear_learned_data_store,
    export_finetune_jsonl_from_learning_store,
    learn_from_directory,
    learn_from_file,
    learned_data_store_path,
    load_learned_data_store,
    save_learned_data_store,
    update_learned_data_store,
)
from gmdgen.validation.code_validation import run_code_validation_suite


AI_PROVIDER_REQUIRED_LABEL = "Ollama does not require an API key for real generation."
AI_GENERATOR_NOTICE = "This generator creates Geometry Dash level plans using local Ollama."
AI_FAILURE_NOTICE = "Local generation is disabled for real level generation."
LOCAL_TEST_NOTICE = "Low Cost mode limits API calls and uses cache."


def summarize_generation_status(result: dict[str, Any]) -> dict[str, str]:
    report = result.get("validation_report", {})
    if not isinstance(report, dict):
        report = {}
    planner_fallback = bool(result.get("planner_fallback_used", report.get("planner_fallback_used", False)))
    low_quality = bool(result.get("low_quality_draft_saved", report.get("low_quality_draft_saved", False)))
    quality_passed = bool(result.get("quality_gate_passed", report.get("quality_gate_passed", True)))
    final_success = bool(result.get("final_success", report.get("final_success", False)))
    if final_success and quality_passed and not planner_fallback and not low_quality:
        return {
            "state": "final_success",
            "status": "Generation completed",
            "title": "Generation completed",
            "summary": "Final output passed syntax, semantic, playability, and report checks.",
        }
    if planner_fallback:
        return {
            "state": "fallback_draft",
            "status": "Fallback draft saved",
            "title": "Fallback Draft Saved",
            "summary": "Ollama planner failed or was invalid; a deterministic fallback draft was saved for inspection.",
        }
    if low_quality or not quality_passed:
        return {
            "state": "low_quality_draft",
            "status": "Validation failed (draft saved)",
            "title": "Low Quality Draft Saved",
            "summary": "The generated draft was saved, but it did not pass the final quality gate.",
        }
    return {
        "state": "incomplete",
        "status": "Generation produced non-final output",
        "title": "Generation Not Final",
        "summary": "The run produced output, but the report did not mark it as final success.",
    }

@dataclass(slots=True)
class GuiGenerationConfig:
    audio_file: str
    output_path: str
    prompt: str = ""
    primary_provider: str = "ollama"
    ollama_model: str = "qwen2.5-coder:7b"
    ollama_base_url: str = ""
    use_ollama_environment_key: bool = True
    level_name: str = "generated_level"
    creator_name: str = "gmdgen"
    difficulty: str = "normal"
    target_duration: float = 120.0
    object_budget: int = 1200
    object_multiplier: float = 1.0
    target_object_count: int | None = None
    seed: int = 42
    safe_mode: bool = True
    high_detail_allowed: bool = False
    two_player_mode: bool = False
    song_offset: float = 0.0
    sync_strength: float = 0.75
    beat_snap_tolerance: float = 0.08
    onset_event_threshold: float = 0.35
    energy_density_scale: float = 1.0
    section_change_sensitivity: float = 0.5
    drop_emphasis: float = 1.0
    max_events_per_beat: int = 2
    start_speed: str = "normal"
    allow_speed_portals: bool = True
    speed_portal_policy: str = "musical"
    allow_triggers: bool = True
    trigger_safety_level: str = "safe"
    group_id_policy: str = "sequential"
    reference_level_file: str = ""
    reference_levels_dir: str = "tests/fixtures/levels"
    context_dir: str = "docs"
    style_strength: float = 0.5
    decoration_intensity: float = 0.6
    structure_density: float = 0.6
    gameplay_density: float = 0.6
    motif_reuse_strength: float = 0.5
    ai_temperature: float = 0.2
    ai_timeout_seconds: int = 60
    ai_retry_count: int = 1
    ollama_num_ctx: int = 4096
    ai_candidate_count: int = 3
    ai_max_regeneration_attempts: int = 2
    ai_quality_retry_enabled: bool = True
    quality_mode: str = "Low Cost"
    low_cost_mode: bool = True
    max_ai_calls_per_generation: int = 2
    cache_ai_responses: bool = True
    max_prompt_chars: int = 6000
    max_output_tokens: int = 4096
    candidates_per_section: int = 3
    section_retry_count: int = 1
    global_refinement_passes: int = 0
    enable_critic: bool = False
    min_acceptable_score: float = 0.35
    min_final_object_count: int = 8
    max_repair_loss_ratio: float = 0.45
    min_drop_impact_score: float = 0.25
    enforce_quality_gate: bool = True
    require_ai_planning: bool = False
    fail_on_ai_planning_error: bool = False
    max_generation_seconds: int = 600
    max_extreme_ml_seconds: int = 300
    fast_materialization: bool = True
    save_debug_bundle: bool = False
    output_format: str = ".gmd"
    save_validation_report: bool = True
    open_output_folder_after_generation: bool = False
    enable_local_test_provider: bool = False
    use_learning_memory: bool = True
    save_learning_data: bool = True
    learning_store_dir: str = ""
    use_learned_data: bool = True
    learned_data_path: str = ""
    learned_data_store_dir: str = ""
    include_debug_artifacts_in_learning: bool = False
    dataset_dir: str = "dataset"
    use_dataset_context: bool = True
    save_generation_to_dataset_learning: bool = True
    use_dataset_learning_memory: bool = True

    @property
    def ai_provider(self) -> str:
        return "local_test_only" if self.enable_local_test_provider else self.primary_provider.strip().lower()

    def to_generation_config(self) -> dict[str, Any]:
        qm = self.quality_mode.lower()
        low_cost_mode = (self.low_cost_mode and qm in {"low cost", "low_cost"}) or qm == "fast"
        if qm == "extreme":
            qg_max_repair = min(self.max_repair_loss_ratio, 0.35)
            qg_min_play = 0.6
            effective_candidates_per_section = max(5, self.candidates_per_section)
            effective_ai_candidate_count = max(5, self.ai_candidate_count)
            effective_regen_attempts = max(4, self.ai_max_regeneration_attempts)
            section_generation_enabled = True
            weak_section_regeneration = True
            enable_critic = True
        elif qm == "draft":
            qg_max_repair = 1.0
            qg_min_play = 0.0
            effective_candidates_per_section = max(1, self.candidates_per_section)
            effective_ai_candidate_count = max(1, self.ai_candidate_count)
            effective_regen_attempts = self.ai_max_regeneration_attempts
            section_generation_enabled = False
            weak_section_regeneration = False
            enable_critic = self.enable_critic
        elif low_cost_mode:
            qg_max_repair = min(self.max_repair_loss_ratio, 0.55)
            qg_min_play = 0.4
            effective_candidates_per_section = 1
            effective_ai_candidate_count = 1
            effective_regen_attempts = 0
            section_generation_enabled = False
            weak_section_regeneration = False
            enable_critic = False
        else:
            qg_max_repair = min(self.max_repair_loss_ratio, 0.45)
            qg_min_play = 0.5
            effective_candidates_per_section = max(3, self.candidates_per_section)
            effective_ai_candidate_count = max(3, self.ai_candidate_count)
            effective_regen_attempts = max(2, self.ai_max_regeneration_attempts)
            section_generation_enabled = qm in {"extreme ml", "extreme_ml"}
            weak_section_regeneration = True
            enable_critic = self.enable_critic

        output_path = Path(self.output_path)
        resolved_dataset_dir = resolve_dataset_dir(self.dataset_dir)
        learning_store_dir = self.learning_store_dir or str(dataset_learning_examples_dir(resolved_dataset_dir))
        learned_data_store_dir = self.learned_data_store_dir or str(dataset_learned_data_store_dir(resolved_dataset_dir))
        config = {
            "audio_file": self.audio_file,
            "prompt": self.prompt,
            "output_dir": str(output_path.parent if str(output_path.parent) else Path(".")),
            "output_name": output_path.stem or self.level_name or "generated_level",
            "generated_author": self.creator_name,
            "difficulty": self.difficulty,
            "target_duration": self.target_duration,
            "object_budget": self.object_budget,
            "object_multiplier": self.object_multiplier,
            "target_object_count": self.target_object_count,
            "seed": self.seed,
            "safe_mode": self.safe_mode,
            "highObjectsEnabled": self.high_detail_allowed,
            "twoPlayerMode": self.two_player_mode,
            "song_offset": self.song_offset,
            "sync_strength": self.sync_strength,
            "beat_snap_tolerance": self.beat_snap_tolerance,
            "onset_event_threshold": self.onset_event_threshold,
            "energy_density_scale": self.energy_density_scale,
            "section_change_sensitivity": self.section_change_sensitivity,
            "drop_emphasis": self.drop_emphasis,
            "max_events_per_beat": self.max_events_per_beat,
            "start_speed": self.start_speed,
            "allow_speed_portals": self.allow_speed_portals,
            "speed_portal_policy": self.speed_portal_policy,
            "allow_triggers": self.allow_triggers,
            "trigger_safety_level": self.trigger_safety_level,
            "group_id_policy": self.group_id_policy,
            "style_reference_level": self.reference_level_file or None,
            "reference_levels_dir": self.reference_levels_dir,
            "context_dir": self.context_dir,
            "ai_provider": self.ai_provider,
            "primary_provider": self.ai_provider,
            "use_ai_planner": True,
            "ollama_model": self.ollama_model,
            "ollama_base_url": self.ollama_base_url,
            "ollama_base_url_env": "OLLAMA_HOST",
            "ollama_num_ctx": self.ollama_num_ctx,
            "ollama_timeout_seconds": self.ai_timeout_seconds,
            "ollama_max_retries": self.ai_retry_count,
            "ollama_save_debug_artifacts": True,
            "fallback_providers": [],
            "low_cost_mode": low_cost_mode,
            "max_ai_calls_per_generation": max(1, self.max_ai_calls_per_generation),
            "max_ai_calls_per_section": 1 if low_cost_mode else max(1, self.candidates_per_section),
            "cache_ai_responses": self.cache_ai_responses,
            "compact_context": low_cost_mode,
            "max_prompt_chars": self.max_prompt_chars,
            "ai_max_context_chars": min(self.max_prompt_chars, 4000) if low_cost_mode else self.max_prompt_chars,
            "max_output_tokens": self.max_output_tokens,
            "disable_critic_in_low_cost": True,
            "candidates_per_section_by_mode": {"low_cost": 1, "balanced": 3, "extreme": 5},
            "style_strength": self.style_strength,
            "decoration_intensity": self.decoration_intensity,
            "structure_density": self.structure_density,
            "gameplay_density": self.gameplay_density,
            "motif_reuse_strength": self.motif_reuse_strength,
            "ai_temperature": self.ai_temperature,
            "ai_timeout_seconds": self.ai_timeout_seconds,
            "ollama_max_retries": self.ai_retry_count,
            "ai_candidate_count": effective_ai_candidate_count,
            "ai_max_regeneration_attempts": effective_regen_attempts,
            "ai_quality_retry_enabled": self.ai_quality_retry_enabled,
            "quality_mode": self.quality_mode,
            "section_generation_enabled": section_generation_enabled,
            "candidates_per_section": effective_candidates_per_section,
            "section_retry_count": self.section_retry_count,
            "global_refinement_passes": self.global_refinement_passes,
            "enforce_quality_gate": self.enforce_quality_gate,
            "require_ai_planning": self.require_ai_planning,
            "fail_on_ai_planning_error": self.fail_on_ai_planning_error,
            "max_generation_seconds": self.max_generation_seconds,
            "max_extreme_ml_seconds": self.max_extreme_ml_seconds,
            "fast_materialization": self.fast_materialization,
            "enable_critic": enable_critic,
            "weak_section_regeneration": weak_section_regeneration,
            "max_quality_gate_retries": max(2 if qm == "extreme" else 1, self.ai_max_regeneration_attempts),
            "estimated_ollama_call_count": self.estimated_ollama_call_count(),
            "estimated_ai_call_count": self.estimated_ai_call_count(),
            "min_acceptable_score": self.min_acceptable_score,
            "min_final_object_count": self.min_final_object_count,
            "max_repair_loss_ratio": self.max_repair_loss_ratio,
            "min_drop_impact_score": self.min_drop_impact_score,
            "quality_gate_max_repair_loss": qg_max_repair,
            "quality_gate_min_playability": qg_min_play,
            "allow_low_quality_draft_save": qm != "extreme",
            "real_generation_requires_external_ai": True,
            "cli_generate_disabled": True,
            "output_format": self.output_format,
            "save_validation_report": self.save_validation_report,
            "save_debug_bundle": self.save_debug_bundle,
            "open_output_folder_after_generation": self.open_output_folder_after_generation,
            "use_learning_memory": self.use_learning_memory,
            "save_learning_data": self.save_learning_data,
            "learning_store_dir": learning_store_dir,
            "use_learned_data": self.use_learned_data,
            "learned_data_store_dir": learned_data_store_dir,
            "include_debug_artifacts_in_learning": self.include_debug_artifacts_in_learning,
            "dataset_dir": str(resolved_dataset_dir),
            "use_dataset_context": self.use_dataset_context,
            "recursive_dataset_scan": True,
            "include_all_dataset_files": True,
            "save_generation_to_dataset_learning": self.save_generation_to_dataset_learning,
            "use_dataset_learning_memory": self.use_dataset_learning_memory,
        }
        if self.enable_local_test_provider:
            config["allow_local_test_provider"] = True
        return config

    def estimated_ollama_call_count(self) -> int:
        return self.estimated_ai_call_count()

    def estimated_ai_call_count(self) -> int:
        if self.quality_mode.lower() == "extreme":
            base_sections = 6
            critic_multiplier = 2 if self.enable_critic or self.quality_mode.lower() == "extreme" else 1
            return max(1, base_sections * max(1, self.candidates_per_section) * critic_multiplier + self.global_refinement_passes)
        if (self.low_cost_mode and self.quality_mode.lower() in {"low cost", "low_cost"}) or self.quality_mode.lower() == "fast":
            return min(max(1, self.max_ai_calls_per_generation), 2)
        return max(1, self.ai_candidate_count)


@dataclass(slots=True)
class GuiAppState:
    training_result: AutoTrainingResult | None = None
    audit_result: Any | None = None
    context_ready: bool = False
    logs: list[str] = field(default_factory=list)
    last_learning_example_id: str = ""
    last_learned_data_summary: dict[str, Any] = field(default_factory=dict)


def environment_api_key_available(env_name: str = "OLLAMA_HOST") -> bool:
    return bool(os.environ.get(env_name, "").strip())



def ollama_package_available() -> bool:
    return importlib.util.find_spec("requests") is not None


def mask_api_key(api_key: str) -> str:
    if not api_key:
        return ""
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return api_key[:3] + "*" * max(4, len(api_key) - 7) + api_key[-4:]


def redact_secret(value: str) -> str:
    if not value:
        return value
    if value.startswith("sk-"):
        return "sk-[REDACTED]"
    if value.startswith("AIza"):
        return "AIza[REDACTED]"
    return value


def redact_text(text: str) -> str:
    if not text:
        return text
    result = text
    ollama_env_key = os.environ.get("OLLAMA_HOST", "")
    if ollama_env_key:
        result = result.replace(ollama_env_key, "[REDACTED_OLLAMA_HOST]")
    for token in result.split():
        if token.startswith("sk-"):
            result = result.replace(token, "sk-[REDACTED]")
        if token.startswith("AIza"):
            result = result.replace(token, "AIza[REDACTED]")
    return result


def sanitize_report(report: dict[str, Any]) -> dict[str, Any]:
    sanitized = {}
    for key, value in report.items():
        if "api_key" in str(key).lower() or "base_url" in str(key).lower() or "host" in str(key).lower():
            continue
        if isinstance(value, str):
            sanitized[key] = redact_text(value)
        elif isinstance(value, dict):
            sanitized[key] = sanitize_report(value)  # type: ignore
        elif isinstance(value, list):
            sanitized[key] = [redact_text(item) if isinstance(item, str) else item for item in value]  # type: ignore
        else:
            sanitized[key] = value
    return sanitized


def sanitize_debug_artifact(data: dict[str, Any]) -> dict[str, Any]:
    return sanitize_report(data)


def safe_gui_callback(func: Any) -> Any:
    def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return func(self, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001
            if hasattr(self, "_handle_gui_exception"):
                self._handle_gui_exception(exc, stage=getattr(func, "__name__", "callback"))
                return None
            raise

    return wrapper


def summarize_generation_error(error_text: str) -> str:
    text = redact_text(error_text)
    if "code=" in text and "user_message=" in text:
        for line in text.splitlines():
            if line.startswith("user_message="):
                return line.split("=", 1)[1][:1200]
    missing_match = re.search(
        r"(?P<path>[\w.\[\]]+): required is missing keys: \[(?P<keys>[^\]]+)\]",
        text,
    )
    if missing_match:
        keys = missing_match.group("keys").replace("'", "").replace('"', "")
        return f"Ollama schema error: schema missing required key(s): {keys} at {missing_match.group('path')}"
    api_missing_match = re.search(r"Missing '([^']+)'", text)
    if "invalid_json_schema" in text and api_missing_match:
        return f"Ollama schema error: schema missing required key: {api_missing_match.group(1)}"
    if "Ollama structured output schema is invalid" in text:
        return text.splitlines()[0][:900]
    if "invalid_json_schema" in text:
        return "Ollama schema error: structured output schema is invalid. See Logs / Audit for details."
    return text[:1200]


def validate_gui_generation_config(config: GuiGenerationConfig) -> list[str]:
    errors: list[str] = []
    if not config.audio_file.strip():
        errors.append("Audio file is required.")
    else:
        audio = Path(config.audio_file)
        if not audio.exists():
            errors.append(f"Audio file does not exist: {audio}")
        elif not audio.is_file():
            errors.append(f"Audio file must be a file: {audio}")
        elif audio.suffix.lower() not in {".wav", ".mp3", ".ogg", ".flac"}:
            errors.append(f"Unsupported audio file extension: {audio.suffix}")
    if not config.output_path.strip():
        errors.append("Output path is required.")
    else:
        output_parent = Path(config.output_path).expanduser().parent
        try:
            output_parent.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Output directory cannot be created: {output_parent} ({exc})")
    provider = config.ai_provider
    if provider == "ollama" and not config.ollama_model.strip() and not config.enable_local_test_provider:
        errors.append("ollama_model is required")
    has_ollama_key = bool(config.ollama_base_url.strip()) or (
        config.use_ollama_environment_key and environment_api_key_available("OLLAMA_HOST")
    )
    has_key = has_ollama_key
    if not has_key and not config.enable_local_test_provider:
        errors.append("Ollama base URL or OLLAMA_HOST is required")
    if provider == "ollama" and has_key and not config.enable_local_test_provider and not ollama_package_available():
        errors.append("requests package is not installed. Run: pip install requests")
    if config.object_budget <= 0:
        errors.append("Object budget must be positive.")
    if config.target_duration <= 0:
        errors.append("Target duration must be positive.")
    if not 0.0 <= config.sync_strength <= 1.0:
        errors.append("Sync strength must be between 0.0 and 1.0.")
    if not 0.0 <= config.ai_temperature <= 2.0:
        errors.append("AI temperature must be between 0.0 and 2.0.")
    if config.ai_timeout_seconds <= 0:
        errors.append("AI timeout seconds must be positive.")
    if config.max_ai_calls_per_generation <= 0:
        errors.append("Max AI calls per generation must be positive.")
    if config.candidates_per_section <= 0:
        errors.append("Candidates per section must be positive.")
    if not config.dataset_dir.strip():
        errors.append("Dataset path is required.")
    else:
        try:
            resolve_dataset_dir(config.dataset_dir).mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"Dataset directory cannot be created: {config.dataset_dir} ({exc})")
    return errors


def gui_labels() -> dict[str, str]:
    return {
        "required": AI_PROVIDER_REQUIRED_LABEL,
        "notice": AI_GENERATOR_NOTICE,
        "failure": AI_FAILURE_NOTICE,
        "local_test": LOCAL_TEST_NOTICE,
    }


class GuiGenerationWorker:
    def __init__(self, app_state: GuiAppState, config: GuiGenerationConfig) -> None:
        self.app_state = app_state
        self.config = config
        self.result: dict[str, Any] | None = None
        self.error: str = ""

    def run(self) -> dict[str, Any]:
        config_dict = self.config.to_generation_config()
        prior_ollama_key = os.environ.get("OLLAMA_HOST")
        should_set_ollama_env_key = bool(self.config.ollama_base_url.strip())
        if should_set_ollama_env_key:
            os.environ["OLLAMA_HOST"] = self.config.ollama_base_url.strip()
        try:
            result = generate_from_config(config_dict)
        except QualityGateFailure:
            # Preserve as quality failure - GUI will handle separately
            raise
        except Exception as exc:  # noqa: BLE001
            error_info = sanitize_exception(exc)
            self.error = redact_text(format_error_for_log(error_info))
            raise RuntimeError(format_error_for_gui(error_info)) from exc
        finally:
            if should_set_ollama_env_key:
                if prior_ollama_key is None:
                    os.environ.pop("OLLAMA_HOST", None)
                else:
                    os.environ["OLLAMA_HOST"] = prior_ollama_key
        self.result = sanitize_report(result)
        if self.config.save_learning_data:
            example = build_learning_example_from_result(
                self.result,
                self.config.to_generation_config(),
            )
            explicit_learning_store = bool(str(self.config.learning_store_dir or "").strip())
            default_dataset_store = str(dataset_learning_examples_dir(self.config.dataset_dir))
            if self.config.save_generation_to_dataset_learning and (
                not explicit_learning_store
                or str(self.config.learning_store_dir).strip() == default_dataset_store
            ):
                self.app_state.last_learning_example_id = save_learning_example_to_dataset(
                    example,
                    self.config.dataset_dir,
                )
            else:
                self.app_state.last_learning_example_id = save_learning_example(
                    example,
                    store_dir=self.config.learning_store_dir or None,
                )
            self.result["learning_example_id"] = self.app_state.last_learning_example_id
        return self.result


class DataLearningWorker:
    def __init__(self, app_state: GuiAppState, *, data_path: str, store_dir: str = "") -> None:
        self.app_state = app_state
        self.data_path = data_path
        self.store_dir = store_dir
        self.result: dict[str, Any] | None = None
        self.error: str = ""

    def run(self) -> dict[str, Any]:
        path = Path(self.data_path).expanduser()
        if not str(self.data_path).strip():
            raise ValueError("Learning data path is required.")
        if not path.exists():
            raise FileNotFoundError(f"Learning data path does not exist: {path}")
        previous = load_learned_data_store(store_dir=self.store_dir or None)
        learned = learn_from_directory(path) if path.is_dir() else learn_from_file(path)
        merged = update_learned_data_store(previous, learned)
        output_path = save_learned_data_store(merged, store_dir=self.store_dir or None)
        self.result = {
            "success": bool(learned.learned_levels or learned.motif_bank or merged.learned_levels),
            "learned_level_count": len(merged.learned_levels),
            "extracted_motif_count": len(merged.motif_bank),
            "style_profile_count": len(merged.style_profiles),
            "failure_pattern_count": len(merged.failure_patterns),
            "store_path": str(output_path),
            "updated_at": merged.updated_at,
            "warnings": list(learned.failure_patterns[:8]),
        }
        self.app_state.last_learned_data_summary = sanitize_report(self.result)
        return self.result


class GuiApplication:
    def __init__(self, *, training_config: AutoTrainingConfig | None = None) -> None:
        self.training_config = training_config or AutoTrainingConfig()
        self.state = GuiAppState()

    def startup(self) -> GuiAppState:
        training_result, _index = run_auto_training(self.training_config)
        self.state.training_result = training_result
        self.state.context_ready = bool(training_result.success)
        self.state.logs.append("Context ready" if training_result.success else "Context preparation failed")
        return self.state

    def rebuild_context(self, *, rebuild: bool = True) -> AutoTrainingResult:
        try:
            config = rebuild_auto_training_config(self.training_config, rebuild=rebuild)
            training_result, _index = run_auto_training(config)
        except Exception as exc:  # noqa: BLE001
            training_result = AutoTrainingResult(
                success=False,
                errors=[f"context_rebuild_failed: {exc}"],
                rebuild=rebuild,
            )
        self.state.training_result = training_result
        self.state.context_ready = bool(training_result.success)
        self.state.logs.append("Context ready" if training_result.success else "Context rebuild failed")
        return training_result


    def generate(self, config: GuiGenerationConfig) -> dict[str, Any]:
        errors = validate_gui_generation_config(config)
        if errors:
            raise ValueError("; ".join(errors))
        worker = GuiGenerationWorker(self.state, config)
        return worker.run()

    def learn_data(self, data_path: str, *, store_dir: str = "") -> dict[str, Any]:
        worker = DataLearningWorker(
            self.state,
            data_path=data_path,
            store_dir=store_dir or str(dataset_learned_data_store_dir(self.training_config.dataset_dir)),
        )
        return worker.run()

    def clear_learned_data(self, *, store_dir: str = "") -> bool:
        cleared = clear_learned_data_store(
            store_dir=store_dir or str(dataset_learned_data_store_dir(self.training_config.dataset_dir))
        )
        self.state.last_learned_data_summary = {}
        return cleared

    def run_code_validation(self, *, include_pytest: bool = True) -> dict[str, Any]:
        report = run_code_validation_suite(
            Path(__file__).resolve().parents[3],
            dataset_dir=self.training_config.dataset_dir,
            include_pytest=include_pytest,
        )
        return report.to_dict()


def launch_gui() -> int:
    app = GuiApplication()
    app.startup()
    if _headless_environment():
        print("GUI headless environment detected; context initialized only.")
        return 0
    try:
        import tkinter as tk
    except Exception:
        print("GUI dependencies unavailable; run in a desktop Python environment.")
        return 1
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText

    class GuiMainWindow:
        def __init__(self, root: tk.Tk, application: GuiApplication) -> None:
            self.root = root
            self.app = application
            self.worker_thread: threading.Thread | None = None
            self.last_result: dict[str, Any] | None = None
            self.status_var = tk.StringVar(value="Ready")
            self.training_var = tk.StringVar(value="Context ready: unknown")
            self.audit_var = tk.StringVar(value="Audit: unknown")
            self._init_vars()
            self._build_ui(ttk, filedialog, messagebox, ScrolledText)
            self._refresh_runtime_status()

        def _init_vars(self) -> None:
            self.v_audio_file = tk.StringVar()
            self.v_output_path = tk.StringVar(value=str(Path("outputs") / "generated.gmd"))
            self.v_prompt = tk.StringVar()
            self.v_primary_provider = tk.StringVar(value="ollama")
            self.v_ollama_model = tk.StringVar(value="qwen2.5-coder:7b")
            self.v_ollama_base_url = tk.StringVar()
            self.v_use_ollama_environment_key = tk.BooleanVar(value=True)
            self.v_level_name = tk.StringVar(value="generated_level")
            self.v_creator_name = tk.StringVar(value="gmdgen")
            self.v_difficulty = tk.StringVar(value="normal")
            self.v_target_duration = tk.StringVar(value="120.0")
            self.v_object_budget = tk.StringVar(value="1200")
            self.v_seed = tk.StringVar(value="42")
            self.v_safe_mode = tk.BooleanVar(value=True)
            self.v_high_detail_allowed = tk.BooleanVar(value=False)
            self.v_two_player_mode = tk.BooleanVar(value=False)
            self.v_song_offset = tk.StringVar(value="0.0")
            self.v_sync_strength = tk.StringVar(value="0.75")
            self.v_beat_snap_tolerance = tk.StringVar(value="0.08")
            self.v_onset_event_threshold = tk.StringVar(value="0.35")
            self.v_energy_density_scale = tk.StringVar(value="1.0")
            self.v_section_change_sensitivity = tk.StringVar(value="0.5")
            self.v_drop_emphasis = tk.StringVar(value="1.0")
            self.v_max_events_per_beat = tk.StringVar(value="2")
            self.v_start_speed = tk.StringVar(value="normal")
            self.v_allow_speed_portals = tk.BooleanVar(value=True)
            self.v_speed_portal_policy = tk.StringVar(value="musical")
            self.v_allow_triggers = tk.BooleanVar(value=True)
            self.v_trigger_safety_level = tk.StringVar(value="safe")
            self.v_group_id_policy = tk.StringVar(value="sequential")
            self.v_reference_level_file = tk.StringVar()
            self.v_reference_levels_dir = tk.StringVar(value="tests/fixtures/levels")
            self.v_context_dir = tk.StringVar(value="docs")
            self.v_style_strength = tk.StringVar(value="0.5")
            self.v_decoration_intensity = tk.StringVar(value="0.6")
            self.v_structure_density = tk.StringVar(value="0.6")
            self.v_gameplay_density = tk.StringVar(value="0.6")
            self.v_motif_reuse_strength = tk.StringVar(value="0.5")
            self.v_ai_temperature = tk.StringVar(value="0.2")
            self.v_ai_timeout_seconds = tk.StringVar(value="60")
            self.v_ai_retry_count = tk.StringVar(value="1")
            self.v_ollama_num_ctx = tk.StringVar(value="4096")
            self.v_ai_candidate_count = tk.StringVar(value="3")
            self.v_ai_max_regeneration_attempts = tk.StringVar(value="2")
            self.v_ai_quality_retry_enabled = tk.BooleanVar(value=True)
            self.v_quality_mode = tk.StringVar(value="Low Cost")
            self.v_low_cost_mode = tk.BooleanVar(value=True)
            self.v_max_ai_calls_per_generation = tk.StringVar(value="2")
            self.v_cache_ai_responses = tk.BooleanVar(value=True)
            self.v_max_prompt_chars = tk.StringVar(value="6000")
            self.v_max_output_tokens = tk.StringVar(value="4096")
            self.v_candidates_per_section = tk.StringVar(value="3")
            self.v_section_retry_count = tk.StringVar(value="1")
            self.v_global_refinement_passes = tk.StringVar(value="0")
            self.v_enable_critic = tk.BooleanVar(value=False)
            self.v_min_acceptable_score = tk.StringVar(value="0.35")
            self.v_min_final_object_count = tk.StringVar(value="8")
            self.v_max_repair_loss_ratio = tk.StringVar(value="0.45")
            self.v_min_drop_impact_score = tk.StringVar(value="0.25")
            self.v_require_ai_planning = tk.BooleanVar(value=False)
            self.v_max_generation_seconds = tk.StringVar(value="600")
            self.v_max_extreme_ml_seconds = tk.StringVar(value="300")
            self.v_fast_materialization = tk.BooleanVar(value=True)
            self.v_object_multiplier = tk.StringVar(value="1.0")
            self.v_target_object_count = tk.StringVar(value="")
            self.v_output_format = tk.StringVar(value=".gmd")
            self.v_save_validation_report = tk.BooleanVar(value=True)
            self.v_save_debug_bundle = tk.BooleanVar(value=False)
            self.v_open_output_folder_after_generation = tk.BooleanVar(value=False)
            self.v_enable_local_test_provider = tk.BooleanVar(value=False)
            default_dataset_dir = resolve_dataset_dir()
            self.v_dataset_dir = tk.StringVar(value=str(default_dataset_dir))
            self.v_use_dataset_context = tk.BooleanVar(value=True)
            self.v_save_generation_to_dataset_learning = tk.BooleanVar(value=True)
            self.v_use_dataset_learning_memory = tk.BooleanVar(value=True)
            self.v_use_learning_memory = tk.BooleanVar(value=True)
            self.v_save_learning_data = tk.BooleanVar(value=True)
            self.v_learning_store_dir = tk.StringVar(value=str(dataset_learning_examples_dir(default_dataset_dir)))
            self.v_use_learned_data = tk.BooleanVar(value=True)
            self.v_learned_data_path = tk.StringVar()
            self.v_learned_data_store_dir = tk.StringVar(value=str(dataset_learned_data_store_dir(default_dataset_dir)))
            self.v_include_debug_artifacts_in_learning = tk.BooleanVar(value=False)
            self.v_learning_status = tk.StringVar(value="Learned data: not loaded")
            self.v_feedback_rating = tk.StringVar(value="0")
            self.v_feedback_good_bad = tk.StringVar(value="Good")
            self.v_feedback_tags = tk.StringVar()
            self.v_feedback_notes = tk.StringVar()
            self.v_feedback_include_training = tk.BooleanVar(value=True)

        def _build_ui(self, ttk: Any, filedialog: Any, messagebox: Any, ScrolledText: Any) -> None:
            self.filedialog = filedialog
            self.messagebox = messagebox
            self.root.title("gmdgen AI Provider GUI")
            self.root.geometry("1120x760")
            self.root.minsize(980, 650)

            top = ttk.Frame(self.root, padding=10)
            top.pack(fill="x")
            ttk.Label(top, text=AI_GENERATOR_NOTICE).pack(anchor="w")
            ttk.Label(top, text=AI_PROVIDER_REQUIRED_LABEL).pack(anchor="w")
            ttk.Label(top, text=AI_FAILURE_NOTICE).pack(anchor="w")
            ttk.Label(top, text=LOCAL_TEST_NOTICE).pack(anchor="w")
            ttk.Label(top, text="Training scope: Entire ./dataset/ context index; no automatic fine-tuning job.").pack(anchor="w")
            ttk.Label(top, textvariable=self.training_var).pack(anchor="w", pady=(6, 0))
            ttk.Label(top, textvariable=self.audit_var).pack(anchor="w")
            ttk.Label(top, textvariable=self.status_var).pack(anchor="w", pady=(6, 0))
            top_controls = ttk.Frame(top)
            top_controls.pack(fill="x", pady=(8, 0))
            self.generate_btn_top = ttk.Button(top_controls, text="Generate", command=self._generate)
            self.generate_btn_top.pack(side="left")
            self.learn_data_btn_top = ttk.Button(top_controls, text="Learn Data", command=self._learn_data)
            self.learn_data_btn_top.pack(side="left", padx=6)
            ttk.Button(top_controls, text="Rebuild Context", command=self._rebuild_context).pack(side="left", padx=6)
            ttk.Button(top_controls, text="Ollama-only Audit", command=self._run_audit).pack(side="left")
            ttk.Button(top_controls, text="Run Code Validation", command=self._run_code_validation).pack(side="left", padx=6)
            ttk.Button(top_controls, text="Run Extreme ML Validation", command=self._run_extreme_ml_validation).pack(side="left", padx=6)

            notebook = ttk.Notebook(self.root)
            notebook.pack(fill="both", expand=True, padx=10, pady=10)

            tab_input = ttk.Frame(notebook, padding=10)
            tab_ai = ttk.Frame(notebook, padding=10)
            tab_generation = ttk.Frame(notebook, padding=10)
            tab_music = ttk.Frame(notebook, padding=10)
            tab_style = ttk.Frame(notebook, padding=10)
            tab_safety = ttk.Frame(notebook, padding=10)
            tab_learning = ttk.Frame(notebook, padding=10)
            tab_quality = ttk.Frame(notebook, padding=10)
            tab_output = ttk.Frame(notebook, padding=10)
            tab_logs = ttk.Frame(notebook, padding=10)
            notebook.add(tab_input, text="Input")
            notebook.add(tab_ai, text="Ollama AI")
            notebook.add(tab_generation, text="Generation")
            notebook.add(tab_music, text="Music Sync")
            notebook.add(tab_style, text="Style")
            notebook.add(tab_safety, text="Safety")
            notebook.add(tab_learning, text="Learning")
            notebook.add(tab_quality, text="Eval / Quality")
            notebook.add(tab_output, text="Output")
            notebook.add(tab_logs, text="Logs / Audit")

            self._row_file(tab_input, ttk, "Audio file", self.v_audio_file, self._pick_audio_file)
            self._row_file(tab_input, ttk, "Output path", self.v_output_path, self._pick_output_file)
            self._row_text(tab_input, ttk, "Prompt", self.v_prompt)
            self._row_text(tab_input, ttk, "Level name", self.v_level_name)
            self._row_text(tab_input, ttk, "Creator", self.v_creator_name)
            self._row_combo(tab_input, ttk, "Difficulty", self.v_difficulty, ["easy", "normal", "hard", "harder", "insane", "demon"])
            self._row_check(tab_input, ttk, "Two player mode", self.v_two_player_mode)
            self._row_check(tab_input, ttk, "High detail allowed", self.v_high_detail_allowed)

            self._row_text(tab_ai, ttk, "Ollama model", self.v_ollama_model)
            self._row_text(tab_ai, ttk, "Ollama base URL", self.v_ollama_base_url)
            self._row_check(tab_ai, ttk, "Use OLLAMA_HOST env", self.v_use_ollama_environment_key)
            self._row_check(tab_ai, ttk, "Low Cost mode", self.v_low_cost_mode)
            self._row_text(tab_ai, ttk, "Max local Ollama calls per generation", self.v_max_ai_calls_per_generation)
            self._row_check(tab_ai, ttk, "Cache Ollama responses", self.v_cache_ai_responses)
            self._row_text(tab_ai, ttk, "Max prompt chars", self.v_max_prompt_chars)
            self._row_text(tab_ai, ttk, "Max output tokens", self.v_max_output_tokens)
            self._row_text(tab_ai, ttk, "Temperature", self.v_ai_temperature)
            self._row_text(tab_ai, ttk, "Timeout seconds", self.v_ai_timeout_seconds)
            self._row_text(tab_ai, ttk, "Retry count", self.v_ai_retry_count)
            self._row_text(tab_ai, ttk, "Ollama context size (num_ctx)", self.v_ollama_num_ctx)
            self._row_text(tab_ai, ttk, "Candidate count", self.v_ai_candidate_count)
            self._row_text(tab_ai, ttk, "Max regeneration attempts", self.v_ai_max_regeneration_attempts)
            self._row_check(tab_ai, ttk, "Quality retry enabled", self.v_ai_quality_retry_enabled)
            self._row_combo(tab_ai, ttk, "Quality Mode", self.v_quality_mode, ["Low Cost", "Balanced", "Extreme", "Extreme ML"])
            self._row_check(tab_ai, ttk, "Require AI planning (fail if Ollama fails)", self.v_require_ai_planning)
            self._row_text(tab_ai, ttk, "Max Extreme ML seconds", self.v_max_extreme_ml_seconds)
            self._row_text(tab_ai, ttk, "Candidates per section", self.v_candidates_per_section)
            self._row_text(tab_ai, ttk, "Section retry count", self.v_section_retry_count)
            self._row_text(tab_ai, ttk, "Global refinement passes", self.v_global_refinement_passes)
            self._row_check(tab_ai, ttk, "Enable critic", self.v_enable_critic)
            self._row_text(tab_ai, ttk, "Min acceptable score", self.v_min_acceptable_score)
            self._row_text(tab_ai, ttk, "Min final object count", self.v_min_final_object_count)
            self._row_text(tab_ai, ttk, "Max repair loss ratio", self.v_max_repair_loss_ratio)
            self._row_text(tab_ai, ttk, "Min drop impact score", self.v_min_drop_impact_score)
            self._row_check(tab_ai, ttk, "Enable local test provider (dev only)", self.v_enable_local_test_provider)
            ttk.Button(tab_ai, text="Clear Ollama Cache", command=self._clear_ai_cache).pack(anchor="w", pady=(8, 0))
            ttk.Button(tab_ai, text="Ollama-only Audit", command=self._run_audit).pack(anchor="w", pady=(4, 0))

            self._row_text(tab_generation, ttk, "Target duration", self.v_target_duration)
            self._row_text(tab_generation, ttk, "Object budget", self.v_object_budget)
            self._row_text(tab_generation, ttk, "Object multiplier (10x target)", self.v_object_multiplier)
            self._row_text(tab_generation, ttk, "Target object count", self.v_target_object_count)
            self._row_check(tab_generation, ttk, "Fast materialization", self.v_fast_materialization)
            self._row_text(tab_generation, ttk, "Max generation seconds", self.v_max_generation_seconds)
            self._row_text(tab_generation, ttk, "Seed", self.v_seed)
            self._row_check(tab_generation, ttk, "Safe mode", self.v_safe_mode)
            self._row_combo(tab_generation, ttk, "Start speed", self.v_start_speed, ["slow", "normal", "fast", "faster", "fastest"])
            self._row_check(tab_generation, ttk, "Allow speed portals", self.v_allow_speed_portals)
            self._row_combo(tab_generation, ttk, "Speed portal policy", self.v_speed_portal_policy, ["none", "conservative", "musical", "aggressive"])
            self._row_check(tab_generation, ttk, "Allow triggers", self.v_allow_triggers)
            self._row_combo(tab_generation, ttk, "Trigger safety", self.v_trigger_safety_level, ["safe", "balanced", "advanced"])
            self._row_combo(tab_generation, ttk, "Group ID policy", self.v_group_id_policy, ["sequential", "section_scoped"])

            self._row_text(tab_music, ttk, "Song offset", self.v_song_offset)
            self._row_text(tab_music, ttk, "Sync strength", self.v_sync_strength)
            self._row_text(tab_music, ttk, "Beat snap tolerance", self.v_beat_snap_tolerance)
            self._row_text(tab_music, ttk, "Onset event threshold", self.v_onset_event_threshold)
            self._row_text(tab_music, ttk, "Energy density scale", self.v_energy_density_scale)
            self._row_text(tab_music, ttk, "Section change sensitivity", self.v_section_change_sensitivity)
            self._row_text(tab_music, ttk, "Drop emphasis", self.v_drop_emphasis)
            self._row_text(tab_music, ttk, "Max events per beat", self.v_max_events_per_beat)

            self._row_file(tab_style, ttk, "Reference level file", self.v_reference_level_file, self._pick_reference_level_file)
            self._row_dir(tab_style, ttk, "Reference levels dir", self.v_reference_levels_dir, self._pick_reference_levels_dir)
            self._row_dir(tab_style, ttk, "Context dir", self.v_context_dir, self._pick_context_dir)
            self._row_text(tab_style, ttk, "Style strength", self.v_style_strength)
            self._row_text(tab_style, ttk, "Decoration intensity", self.v_decoration_intensity)
            self._row_text(tab_style, ttk, "Structure density", self.v_structure_density)
            self._row_text(tab_style, ttk, "Gameplay density", self.v_gameplay_density)
            self._row_text(tab_style, ttk, "Motif reuse strength", self.v_motif_reuse_strength)

            self._row_combo(tab_safety, ttk, "Output format", self.v_output_format, [".gmd", ".txt", ".json"])
            self._row_check(tab_safety, ttk, "Save validation report", self.v_save_validation_report)
            self._row_check(tab_safety, ttk, "Save debug bundle", self.v_save_debug_bundle)
            self._row_check(tab_safety, ttk, "Open output folder after generation", self.v_open_output_folder_after_generation)
            self._row_check(tab_safety, ttk, "Use learned examples", self.v_use_learning_memory)
            self._row_check(tab_safety, ttk, "Save this generation as learning data", self.v_save_learning_data)
            self._row_dir(tab_safety, ttk, "Learning store dir", self.v_learning_store_dir, self._pick_learning_store_dir)

            self._row_check(tab_learning, ttk, "Use learned data", self.v_use_learned_data)
            self._row_dir(tab_learning, ttk, "Dataset dir", self.v_dataset_dir, self._pick_dataset_dir)
            self._row_check(tab_learning, ttk, "Use dataset context", self.v_use_dataset_context)
            self._row_check(tab_learning, ttk, "Use dataset learning memory", self.v_use_dataset_learning_memory)
            self._row_check(tab_learning, ttk, "Save generation to dataset learning memory", self.v_save_generation_to_dataset_learning)
            self._row_file(tab_learning, ttk, "Data file/folder", self.v_learned_data_path, self._pick_learned_data_path)
            self._row_dir(tab_learning, ttk, "Learned data store dir", self.v_learned_data_store_dir, self._pick_learned_data_store_dir)
            self._row_check(tab_learning, ttk, "Include debug artifacts in learning data", self.v_include_debug_artifacts_in_learning)
            learning_controls = ttk.Frame(tab_learning)
            learning_controls.pack(fill="x", pady=8)
            self.learn_data_btn = ttk.Button(learning_controls, text="Learn Data", command=self._learn_data)
            self.learn_data_btn.pack(side="left")
            ttk.Button(learning_controls, text="Clear Learned Data", command=self._clear_learned_data).pack(side="left", padx=6)
            ttk.Button(learning_controls, text="Export Training Dataset", command=self._export_training_dataset).pack(side="left")
            ttk.Button(learning_controls, text="Rebuild Dataset Context", command=self._rebuild_context).pack(side="left", padx=6)
            ttk.Button(learning_controls, text="Quarantine Bad Memory", command=self._quarantine_bad_memory).pack(side="left")
            ttk.Label(tab_learning, textvariable=self.v_learning_status).pack(anchor="w", pady=(6, 0))

            # Quality Tab
            q_controls = ttk.Frame(tab_quality)
            q_controls.pack(fill="x", pady=8)
            ttk.Button(q_controls, text="Initialize Dataset Structure", command=self._init_dataset).pack(side="left")
            ttk.Button(q_controls, text="Run Reference Analysis", command=self._run_reference_analysis).pack(side="left", padx=6)
            ttk.Button(q_controls, text="Run Quality Eval", command=self._run_quality_eval).pack(side="left")
            ttk.Button(q_controls, text="Run Live AI Eval", command=self._run_live_eval).pack(side="left", padx=6)
            ttk.Button(q_controls, text="Run Geode Check", command=self._run_geode_check).pack(side="left")

            self.v_tuning_recommendation = tk.StringVar(value="No recommendations yet.")
            ttk.Label(tab_quality, text="Tuning Recommendations:").pack(anchor="w", pady=(10, 2))
            self.tuning_label = ttk.Label(tab_quality, textvariable=self.v_tuning_recommendation, foreground="blue", wraplength=900)
            self.tuning_label.pack(anchor="w", pady=(0, 10))

            controls = ttk.Frame(tab_logs)
            controls.pack(fill="x", pady=(0, 8))
            ttk.Button(controls, text="External AI Audit", command=self._run_audit).pack(side="left")
            ttk.Button(controls, text="Rebuild Context", command=self._rebuild_context).pack(side="left", padx=6)
            ttk.Button(controls, text="Clear Context Cache", command=self._clear_context_cache).pack(side="left")
            ttk.Button(controls, text="Run Code Validation", command=self._run_code_validation).pack(side="left", padx=6)
            self.log_text = ScrolledText(tab_logs, height=24, wrap="word")
            self.log_text.pack(fill="both", expand=True)
            self.log_text.insert("end", "GUI initialized.\n")
            self.log_text.configure(state="disabled")
            feedback = ttk.Frame(tab_logs)
            feedback.pack(fill="x", pady=(8, 0))
            ttk.Label(feedback, text="Feedback rating").pack(side="left")
            ttk.Combobox(feedback, textvariable=self.v_feedback_rating, values=["0", "1", "2", "3", "4", "5"], width=4, state="readonly").pack(side="left", padx=4)
            ttk.Combobox(feedback, textvariable=self.v_feedback_good_bad, values=["Good", "Bad"], width=7, state="readonly").pack(side="left", padx=4)
            ttk.Entry(feedback, textvariable=self.v_feedback_tags, width=42).pack(side="left", padx=4)
            ttk.Entry(feedback, textvariable=self.v_feedback_notes, width=42).pack(side="left", padx=4)
            ttk.Checkbutton(feedback, text="Include in training dataset", variable=self.v_feedback_include_training).pack(side="left", padx=4)
            ttk.Button(feedback, text="Save Feedback", command=self._save_feedback).pack(side="left", padx=4)

            bottom = ttk.Frame(self.root, padding=(10, 0, 10, 10))
            bottom.pack(fill="x")
            self.generate_btn = ttk.Button(bottom, text="Generate", command=self._generate)
            self.generate_btn.pack(side="left")
            self.learn_data_btn_bottom = ttk.Button(bottom, text="Learn Data", command=self._learn_data)
            self.learn_data_btn_bottom.pack(side="left", padx=6)
            self.export_ml_btn = ttk.Button(bottom, text="Export ML Dataset", command=self._export_training_dataset)
            self.export_ml_btn.pack(side="left", padx=6)
            self.save_report_btn = ttk.Button(bottom, text="Save Report", command=self._save_report, state="disabled")
            self.save_report_btn.pack(side="left", padx=6)
            self.open_output_btn = ttk.Button(bottom, text="Open Output Folder", command=self._open_output_folder, state="disabled")
            self.open_output_btn.pack(side="left")
            ttk.Button(bottom, text="Quit", command=self.root.destroy).pack(side="right")

        def _row_text(self, parent: Any, ttk: Any, label: str, var: tk.StringVar) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=28).pack(side="left")
            ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)

        def _row_secret(self, parent: Any, ttk: Any, label: str, var: tk.StringVar) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=28).pack(side="left")
            ttk.Entry(frame, textvariable=var, show="*").pack(side="left", fill="x", expand=True)

        def _row_combo(self, parent: Any, ttk: Any, label: str, var: tk.StringVar, values: list[str]) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=28).pack(side="left")
            ttk.Combobox(frame, textvariable=var, values=values, state="readonly").pack(side="left", fill="x", expand=True)

        def _row_check(self, parent: Any, ttk: Any, label: str, var: tk.BooleanVar) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Checkbutton(frame, text=label, variable=var).pack(side="left")

        def _row_file(self, parent: Any, ttk: Any, label: str, var: tk.StringVar, browse_command: Any) -> None:
            frame = ttk.Frame(parent)
            frame.pack(fill="x", pady=2)
            ttk.Label(frame, text=label, width=28).pack(side="left")
            ttk.Entry(frame, textvariable=var).pack(side="left", fill="x", expand=True)
            ttk.Button(frame, text="Browse", command=browse_command).pack(side="left", padx=(6, 0))

        def _row_dir(self, parent: Any, ttk: Any, label: str, var: tk.StringVar, browse_command: Any) -> None:
            self._row_file(parent, ttk, label, var, browse_command)

        def _pick_audio_file(self) -> None:
            path = self.filedialog.askopenfilename(
                title="Select audio file",
                filetypes=[("Audio", "*.wav *.mp3 *.ogg *.flac"), ("All files", "*.*")],
            )
            if path:
                self.v_audio_file.set(path)

        def _pick_output_file(self) -> None:
            path = self.filedialog.asksaveasfilename(
                title="Select output level file",
                defaultextension=".gmd",
                filetypes=[("GMD level", "*.gmd"), ("All files", "*.*")],
            )
            if path:
                self.v_output_path.set(path)

        def _pick_reference_level_file(self) -> None:
            path = self.filedialog.askopenfilename(
                title="Select reference level file",
                filetypes=[("GMD level", "*.gmd"), ("All files", "*.*")],
            )
            if path:
                self.v_reference_level_file.set(path)

        def _pick_reference_levels_dir(self) -> None:
            path = self.filedialog.askdirectory(title="Select reference levels directory")
            if path:
                self.v_reference_levels_dir.set(path)

        def _pick_context_dir(self) -> None:
            path = self.filedialog.askdirectory(title="Select context directory")
            if path:
                self.v_context_dir.set(path)

        def _pick_dataset_dir(self) -> None:
            path = self.filedialog.askdirectory(title="Select dataset directory")
            if path:
                self.v_dataset_dir.set(path)
                self.v_learning_store_dir.set(str(dataset_learning_examples_dir(path)))
                self.v_learned_data_store_dir.set(str(dataset_learned_data_store_dir(path)))

        def _pick_learning_store_dir(self) -> None:
            path = self.filedialog.askdirectory(title="Select learning store directory")
            if path:
                self.v_learning_store_dir.set(path)

        def _pick_learned_data_path(self) -> None:
            path = self.filedialog.askopenfilename(
                title="Select .gmd, .txt, or .json data file",
                filetypes=[("Learning data", "*.gmd *.txt *.json"), ("All files", "*.*")],
            )
            if not path:
                path = self.filedialog.askdirectory(title="Select data directory")
            if path:
                self.v_learned_data_path.set(path)

        def _pick_learned_data_store_dir(self) -> None:
            path = self.filedialog.askdirectory(title="Select learned data store directory")
            if path:
                self.v_learned_data_store_dir.set(path)

        def _append_log(self, text: str) -> None:
            self.log_text.configure(state="normal")
            self.log_text.insert("end", text.rstrip() + "\n")
            self.log_text.see("end")
            self.log_text.configure(state="disabled")

        def _refresh_runtime_status(self) -> None:
            training = self.app.state.training_result
            if training is not None:
                self.training_var.set(
                    f"Context ready: {training.success} | Dataset path: {Path(training.dataset_dir or self.v_dataset_dir.get()).name or './dataset'} | "
                    f"Training scope: Entire dataset | Files indexed: {training.dataset_document_count} | "
                    f"Chunks: {training.dataset_chunk_count} | Reference levels: {training.dataset_reference_level_count} | "
                    f"Skipped: {training.dataset_skipped_files} | Failed: {training.dataset_failed_files} | Cache used: {training.cache_used}"
                )
            if self.app.state.audit_result is not None:
                self.audit_var.set(self.app.state.audit_result.summary)
            self.status_var.set("Ready")

        def _parse_float(self, label: str, value: str) -> float:
            try:
                return float(value.strip())
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{label} must be a number: {value!r}") from exc

        def _parse_int(self, label: str, value: str) -> int:
            try:
                return int(value.strip())
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"{label} must be an integer: {value!r}") from exc

        def _build_generation_config(self) -> GuiGenerationConfig:
            return GuiGenerationConfig(
                audio_file=self.v_audio_file.get().strip(),
                output_path=self.v_output_path.get().strip(),
                prompt=self.v_prompt.get().strip(),
                primary_provider=self.v_primary_provider.get().strip() or "ollama",
                ollama_model=self.v_ollama_model.get().strip() or "qwen2.5-coder:7b",
                ollama_base_url=self.v_ollama_base_url.get().strip(),
                use_ollama_environment_key=bool(self.v_use_ollama_environment_key.get()),
                level_name=self.v_level_name.get().strip() or "generated_level",
                creator_name=self.v_creator_name.get().strip() or "gmdgen",
                difficulty=self.v_difficulty.get().strip() or "normal",
                target_duration=self._parse_float("target_duration", self.v_target_duration.get()),
                object_budget=self._parse_int("object_budget", self.v_object_budget.get()),
                object_multiplier=self._parse_float("object_multiplier", self.v_object_multiplier.get()),
                target_object_count=self._parse_int("target_object_count", self.v_target_object_count.get()) if self.v_target_object_count.get().strip() else None,
                seed=self._parse_int("seed", self.v_seed.get()),
                safe_mode=bool(self.v_safe_mode.get()),
                high_detail_allowed=bool(self.v_high_detail_allowed.get()),
                two_player_mode=bool(self.v_two_player_mode.get()),
                song_offset=self._parse_float("song_offset", self.v_song_offset.get()),
                sync_strength=self._parse_float("sync_strength", self.v_sync_strength.get()),
                beat_snap_tolerance=self._parse_float("beat_snap_tolerance", self.v_beat_snap_tolerance.get()),
                onset_event_threshold=self._parse_float("onset_event_threshold", self.v_onset_event_threshold.get()),
                energy_density_scale=self._parse_float("energy_density_scale", self.v_energy_density_scale.get()),
                section_change_sensitivity=self._parse_float("section_change_sensitivity", self.v_section_change_sensitivity.get()),
                drop_emphasis=self._parse_float("drop_emphasis", self.v_drop_emphasis.get()),
                max_events_per_beat=self._parse_int("max_events_per_beat", self.v_max_events_per_beat.get()),
                start_speed=self.v_start_speed.get().strip() or "normal",
                allow_speed_portals=bool(self.v_allow_speed_portals.get()),
                speed_portal_policy=self.v_speed_portal_policy.get().strip() or "musical",
                allow_triggers=bool(self.v_allow_triggers.get()),
                trigger_safety_level=self.v_trigger_safety_level.get().strip() or "safe",
                group_id_policy=self.v_group_id_policy.get().strip() or "sequential",
                reference_level_file=self.v_reference_level_file.get().strip(),
                reference_levels_dir=self.v_reference_levels_dir.get().strip() or "tests/fixtures/levels",
                context_dir=self.v_context_dir.get().strip() or "docs",
                style_strength=self._parse_float("style_strength", self.v_style_strength.get()),
                decoration_intensity=self._parse_float("decoration_intensity", self.v_decoration_intensity.get()),
                structure_density=self._parse_float("structure_density", self.v_structure_density.get()),
                gameplay_density=self._parse_float("gameplay_density", self.v_gameplay_density.get()),
                motif_reuse_strength=self._parse_float("motif_reuse_strength", self.v_motif_reuse_strength.get()),
                ai_temperature=self._parse_float("ai_temperature", self.v_ai_temperature.get()),
                ai_timeout_seconds=self._parse_int("ai_timeout_seconds", self.v_ai_timeout_seconds.get()),
                ai_retry_count=self._parse_int("ai_retry_count", self.v_ai_retry_count.get()),
                ollama_num_ctx=self._parse_int("ollama_num_ctx", self.v_ollama_num_ctx.get()),
                ai_candidate_count=self._parse_int("ai_candidate_count", self.v_ai_candidate_count.get()),
                ai_max_regeneration_attempts=self._parse_int("ai_max_regeneration_attempts", self.v_ai_max_regeneration_attempts.get()),
                ai_quality_retry_enabled=bool(self.v_ai_quality_retry_enabled.get()),
                quality_mode=self.v_quality_mode.get().strip() or "Low Cost",
                low_cost_mode=bool(self.v_low_cost_mode.get()),
                max_ai_calls_per_generation=self._parse_int("max_ai_calls_per_generation", self.v_max_ai_calls_per_generation.get()),
                cache_ai_responses=bool(self.v_cache_ai_responses.get()),
                max_prompt_chars=self._parse_int("max_prompt_chars", self.v_max_prompt_chars.get()),
                max_output_tokens=self._parse_int("max_output_tokens", self.v_max_output_tokens.get()),
                candidates_per_section=self._parse_int("candidates_per_section", self.v_candidates_per_section.get()),
                section_retry_count=self._parse_int("section_retry_count", self.v_section_retry_count.get()),
                global_refinement_passes=self._parse_int("global_refinement_passes", self.v_global_refinement_passes.get()),
                enable_critic=bool(self.v_enable_critic.get()),
                min_acceptable_score=self._parse_float("min_acceptable_score", self.v_min_acceptable_score.get()),
                min_final_object_count=self._parse_int("min_final_object_count", self.v_min_final_object_count.get()),
                max_repair_loss_ratio=self._parse_float("max_repair_loss_ratio", self.v_max_repair_loss_ratio.get()),
                min_drop_impact_score=self._parse_float("min_drop_impact_score", self.v_min_drop_impact_score.get()),
                require_ai_planning=bool(self.v_require_ai_planning.get()),
                max_generation_seconds=self._parse_int("max_generation_seconds", self.v_max_generation_seconds.get()),
                max_extreme_ml_seconds=self._parse_int("max_extreme_ml_seconds", self.v_max_extreme_ml_seconds.get()),
                fast_materialization=bool(self.v_fast_materialization.get()),
                output_format=self.v_output_format.get().strip() or ".gmd",
                save_validation_report=bool(self.v_save_validation_report.get()),
                save_debug_bundle=bool(self.v_save_debug_bundle.get()),
                open_output_folder_after_generation=bool(self.v_open_output_folder_after_generation.get()),
                enable_local_test_provider=bool(self.v_enable_local_test_provider.get()),
                use_learning_memory=bool(self.v_use_learning_memory.get()),
                save_learning_data=bool(self.v_save_learning_data.get()),
                learning_store_dir=self.v_learning_store_dir.get().strip(),
                use_learned_data=bool(self.v_use_learned_data.get()),
                learned_data_path=self.v_learned_data_path.get().strip(),
                learned_data_store_dir=self.v_learned_data_store_dir.get().strip(),
                include_debug_artifacts_in_learning=bool(self.v_include_debug_artifacts_in_learning.get()),
                dataset_dir=self.v_dataset_dir.get().strip() or "dataset",
                use_dataset_context=bool(self.v_use_dataset_context.get()),
                save_generation_to_dataset_learning=bool(self.v_save_generation_to_dataset_learning.get()),
                use_dataset_learning_memory=bool(self.v_use_dataset_learning_memory.get()),
            )

        def _handle_gui_exception(self, exc: Exception, *, stage: str = "callback") -> None:
            summary = summarize_generation_error(redact_text(str(exc)))
            self.status_var.set(f"{stage} failed")
            self._append_log(f"[{stage}:error] {summary}")
            self._append_log(redact_text(format_error_for_log(sanitize_exception(exc))))
            self._set_generation_running(False)
            self.messagebox.showerror("gmdgen error", summary)

        @safe_gui_callback
        def _run_audit(self) -> None:
            from gmdgen.audit.ollama_only import run_ollama_only_audit
            config = self._build_generation_config()
            audit = run_ollama_only_audit(config.to_generation_config())
            self.audit_var.set(audit.summary)
            self._append_log(f"[ollama-audit] passed={audit.passed} errors={len(audit.errors)} warnings={len(audit.warnings)}")
            for error in audit.errors:
                self._append_log(f"  - error: {error}")
            for warning in audit.warnings:
                self._append_log(f"  - warning: {warning}")

        @safe_gui_callback
        def _rebuild_context(self) -> None:
            self.status_var.set("Rebuilding context...")
            self.root.update_idletasks()
            self.app.training_config = AutoTrainingConfig(
                dataset_dir=self.v_dataset_dir.get().strip() or "dataset",
                context_dirs=[self.v_context_dir.get().strip() or "docs"],
                reference_level_dirs=[self.v_reference_levels_dir.get().strip() or "tests/fixtures/levels"],
                schema_paths=list(self.app.training_config.schema_paths),
                cache_dir=self.app.training_config.cache_dir,
                rebuild=True,
            )
            result = self.app.rebuild_context(rebuild=True)
            self._append_log(
                "[context] rebuilt: "
                f"success={result.success} docs={result.document_count} refs={result.reference_level_count} chunks={result.chunk_count} "
                f"dataset_files={result.dataset_document_count} dataset_chunks={result.dataset_chunk_count} "
                f"dataset_refs={result.dataset_reference_level_count} failed={result.dataset_failed_files} skipped={result.dataset_skipped_files}"
            )
            self._refresh_runtime_status()
            if not result.success:
                self._append_log("[context:error] " + "; ".join(result.errors[:4]))

        @safe_gui_callback
        def _clear_context_cache(self) -> None:
            cache_dir = Path(self.app.training_config.cache_dir)
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            self._append_log(f"[context] cache cleared: {cache_dir}")
            self.status_var.set("Context cache cleared")

        @safe_gui_callback
        def _clear_ai_cache(self) -> None:
            from gmdgen.ai.cache import AIResponseCache

            cache = AIResponseCache(resolve_dataset_dir(self.v_dataset_dir.get()) / "cache" / "ai_responses")
            count = cache.clear()
            self._append_log(f"[ai-cache] cleared {count} cached response file(s)")
            self.status_var.set("AI response cache cleared")

        def _set_generation_running(self, is_running: bool) -> None:
            state = "disabled" if is_running else "normal"
            for button_name in ("generate_btn", "generate_btn_top"):
                button = getattr(self, button_name, None)
                if button is not None:
                    button.configure(state=state)

        def _set_learning_running(self, is_running: bool) -> None:
            state = "disabled" if is_running else "normal"
            for button_name in ("learn_data_btn", "learn_data_btn_top", "learn_data_btn_bottom", "export_ml_btn"):
                button = getattr(self, button_name, None)
                if button is not None:
                    button.configure(state=state)

        @safe_gui_callback
        def _run_extreme_ml_validation(self) -> None:
            if self.worker_thread and self.worker_thread.is_alive():
                self.messagebox.showinfo("Extreme ML Validation", "Another background task is already running.")
                return
            self.status_var.set("Running Extreme ML Validation...")
            self._append_log("[ml_eval] started")

            def task() -> None:
                try:
                    from gmdgen.eval.extreme_validation import run_extreme_ml_validation
                    from pathlib import Path
                    report = run_extreme_ml_validation(Path("dataset"))
                except Exception as exc:  # noqa: BLE001
                    import traceback
                    err_msg = str(exc)
                    self.root.after(0, lambda err=err_msg: self._on_extreme_ml_validation_failed(err))  # type: ignore
                    return
                self.root.after(0, lambda payload=report: self._on_extreme_ml_validation_success(payload))  # type: ignore

            self.worker_thread = threading.Thread(target=task, daemon=True)
            self.worker_thread.start()

        def _on_extreme_ml_validation_success(self, report: Any) -> None:
            self.status_var.set("Extreme ML Validation completed")
            self._append_log(f"[ml_eval] done: {report.to_dict()}")
            self.messagebox.showinfo(
                "Extreme ML Validation",
                f"Validation completed.\nBaseline Score: {report.baseline_score}\nDataset Health: {report.dataset_health}"
            )

        def _on_extreme_ml_validation_failed(self, error_msg: str) -> None:
            self.status_var.set("Extreme ML Validation failed")
            self._append_log(f"[ml_eval] failed: {error_msg}")
            self.messagebox.showerror("Validation Failed", error_msg)

        @safe_gui_callback
        def _run_code_validation(self) -> None:
            if self.worker_thread and self.worker_thread.is_alive():
                self.messagebox.showinfo("Code Validation", "Another background task is already running.")
                return
            self.status_var.set("Running code validation...")
            self._append_log("[code-validation] started")

            def task() -> None:
                try:
                    report = self.app.run_code_validation(include_pytest=True)
                except Exception as exc:  # noqa: BLE001
                    summary = summarize_generation_error(redact_text(str(exc)))
                    self.root.after(0, lambda err=summary: self._on_code_validation_failed(err))  # type: ignore
                    return
                self.root.after(0, lambda payload=report: self._on_code_validation_success(payload))  # type: ignore

            self.worker_thread = threading.Thread(target=task, daemon=True)
            self.worker_thread.start()

        def _on_code_validation_success(self, report: dict[str, Any]) -> None:
            self.status_var.set("Code validation completed")
            self._append_log(
                "[code-validation] done: "
                f"passed={report.get('overall_passed')} errors={len(report.get('errors', []))} "
                f"warnings={len(report.get('warnings', []))}"
            )
            for result in report.get("results", [])[:12]:
                self._append_log(
                    "[code-validation:result] "
                    f"cmd={' '.join(result.get('command', []))} passed={result.get('passed')} "
                    f"skipped={result.get('skipped', False)} rc={result.get('return_code', 0)}"
                )
                if result.get("stderr_tail"):
                    self._append_log(str(result.get("stderr_tail"))[-1200:])

        def _on_code_validation_failed(self, error_text: str) -> None:
            self.status_var.set("Code validation failed")
            self._append_log(f"[code-validation:error] {error_text}")
            self.messagebox.showerror("Code Validation failed", error_text)

        @safe_gui_callback
        def _learn_data(self) -> None:
            data_path = self.v_learned_data_path.get().strip()
            if not data_path:
                self.messagebox.showerror("Learn Data", "Learning data path is required.")
                return
            if self.worker_thread and self.worker_thread.is_alive():
                self.messagebox.showinfo("Learn Data", "Another background task is already running.")
                return
            self._set_learning_running(True)
            self.status_var.set("Learning data...")
            self._append_log(f"[learn] started: {Path(data_path).name}")

            def task() -> None:
                try:
                    result = self.app.learn_data(
                        data_path,
                        store_dir=self.v_learned_data_store_dir.get().strip(),
                    )
                except Exception as exc:  # noqa: BLE001
                    summary = summarize_generation_error(redact_text(str(exc)))
                    self.root.after(0, lambda err=summary: self._on_learning_failed(err))  # type: ignore
                    return
                self.root.after(0, lambda payload=result: self._on_learning_success(payload))  # type: ignore

            self.worker_thread = threading.Thread(target=task, daemon=True)
            self.worker_thread.start()

        def _on_learning_success(self, result: dict[str, Any]) -> None:
            self._set_learning_running(False)
            self.status_var.set("Learning completed")
            self.v_learning_status.set(
                "Learned data: "
                f"levels={result.get('learned_level_count', 0)} "
                f"motifs={result.get('extracted_motif_count', 0)} "
                f"profiles={result.get('style_profile_count', 0)} "
                f"updated={result.get('updated_at', '')}"
            )
            self._append_log(
                "[learn] done: "
                f"levels={result.get('learned_level_count', 0)} motifs={result.get('extracted_motif_count', 0)} "
                f"profiles={result.get('style_profile_count', 0)} store={result.get('store_path', '')}"
            )
            for warning in result.get("warnings", [])[:5]:
                self._append_log(f"[learn:warning] {warning}")

        def _on_learning_failed(self, error_text: str) -> None:
            self._set_learning_running(False)
            self.status_var.set("Learning failed")
            self._append_log(f"[learn:error] {error_text}")
            self.messagebox.showerror("Learn Data failed", error_text)

        @safe_gui_callback
        def _clear_learned_data(self) -> None:
            cleared = self.app.clear_learned_data(store_dir=self.v_learned_data_store_dir.get().strip())
            self.v_learning_status.set("Learned data: cleared" if cleared else "Learned data: nothing to clear")
            self._append_log(f"[learn] cleared={cleared}")

        @safe_gui_callback
        def _quarantine_bad_memory(self) -> None:
            from gmdgen.learning.store import quarantine_bad_learning_examples
            store_dir = self.v_learned_data_store_dir.get().strip() or None
            count = quarantine_bad_learning_examples(store_dir=store_dir)
            self._append_log(f"[learn] quarantined {count} bad examples.")
            self.messagebox.showinfo("Quarantine Bad Memory", f"Successfully quarantined {count} bad memory examples.")

        @safe_gui_callback
        def _export_training_dataset(self) -> None:
            output = self.filedialog.asksaveasfilename(
                title="Export fine-tuning JSONL",
                defaultextension=".jsonl",
                filetypes=[("JSONL", "*.jsonl"), ("All files", "*.*")],
            )
            if not output:
                return
            path = export_finetune_jsonl_from_learning_store(
                output,
                store_dir=self.v_learned_data_store_dir.get().strip() or None,
            )
            self._append_log(f"[learn] exported training dataset: {path}")

        @safe_gui_callback
        def _generate(self) -> None:
            if self.worker_thread and self.worker_thread.is_alive():
                self.messagebox.showinfo("Generate", "Generation already in progress.")
                return
            try:
                config = self._build_generation_config()
                errors = validate_gui_generation_config(config)
                if errors:
                    raise ValueError("\n".join(errors))
            except Exception as exc:  # noqa: BLE001
                self.messagebox.showerror("Invalid input", str(exc))
                return

            self._set_generation_running(True)
            self.status_var.set("Generating...")
            self._append_log("[generate] started")

            def task() -> None:
                try:
                    result = self.app.generate(config)
                except QualityGateFailure as qgf:
                    self.root.after(0, lambda e=qgf: self._on_quality_gate_failed(e))  # type: ignore
                    return
                except Exception as exc:  # noqa: BLE001
                    error_info = sanitize_exception(exc)
                    gui_msg = format_error_for_gui(error_info)
                    log_msg = format_error_for_log(error_info)
                    self.root.after(0, lambda err=gui_msg, detail=log_msg: self._on_generate_failed(err, detail))  # type: ignore
                    return
                self.root.after(0, lambda payload=result, cfg=config: self._on_generate_success(payload, cfg))  # type: ignore

            self.worker_thread = threading.Thread(target=task, daemon=True)
            self.worker_thread.start()

        def _on_generate_failed(self, error_text: str, detail_text: str | None = None) -> None:
            self._set_generation_running(False)
            self.status_var.set("Generation failed")
            self._append_log(f"[generate:error] {error_text}")
            if detail_text and detail_text != error_text:
                self._append_log("[generate:details]")
                self._append_log(detail_text)
            self.messagebox.showerror("Generation failed", error_text)

        def _on_quality_gate_failed(self, exc: QualityGateFailure) -> None:
            self._set_generation_running(False)
            details = getattr(exc, "details", {}) or {}
            qg_report = details.get("quality_gate_report", {}) or {}
            # failures can be in quality_gate_report or top-level details
            failures = qg_report.get("failures", details.get("failures", []))
            causes = qg_report.get("primary_causes", details.get("primary_causes", []))
            actions = qg_report.get("recommended_actions", details.get("recommended_actions", []))
            draft_result = details.get("save_result")

            # Build human-readable message
            fail_lines = "\n".join(f"  • {f}" for f in failures[:6]) if failures else str(exc)
            cause_lines = ("\nCauses:\n" + "\n".join(f"  - {c}" for c in causes[:3])) if causes else ""
            action_lines = ("\nSuggested actions:\n" + "\n".join(f"  → {a}" for a in actions[:3])) if actions else ""
            msg = f"Quality gate failed. The generated level did not meet quality standards.\n\nFailed checks:\n{fail_lines}{cause_lines}{action_lines}"

            self._append_log(f"[generate:quality_failure] quality_gate_failed")
            for f in failures:
                self._append_log(f"  [check] {f}")

            # Only show "draft saved" if a draft was actually written to disk
            if draft_result and draft_result.get("success") and draft_result.get("file_exists"):
                draft_path = draft_result.get("resolved_output_path", "")
                draft_size = draft_result.get("file_size_bytes", 0)
                self._append_log(f"[generate] draft_saved: path={draft_path} size={draft_size}")
                self.status_var.set("Quality gate failed (draft saved)")
                self.messagebox.showwarning(
                    "Quality Gate Failed – Draft Saved",
                    msg + f"\n\nA low-quality draft was saved to:\n{draft_path}\n(size: {draft_size} bytes)"
                )
            else:
                self.status_var.set("Quality gate failed – nothing saved")
                self._append_log("[generate] no_file_saved (quality gate failure, no draft)")
                self.messagebox.showwarning("Quality Gate Failed", msg)


        def _on_generate_success(self, result: dict[str, Any], config: GuiGenerationConfig) -> None:
            self.last_result = result
            self._set_generation_running(False)
            
            save_res = result.get("save_result")
            if save_res and not save_res.get("success", False):
                self.status_var.set("Generation failed to save")
                self._append_log("[generate:error] Level saving failed.")
                for err in save_res.get("errors", []):
                    self._append_log(f"  - {err}")
                self.messagebox.showerror("Save Failed", "Generated level could not be saved to disk.")
                return

            if getattr(self, "save_report_btn", None) is not None:
                self.save_report_btn.configure(state="normal")
            if getattr(self, "open_output_btn", None) is not None:
                self.open_output_btn.configure(state="normal")
                
            status_summary = summarize_generation_status(result)
            if status_summary["state"] == "fallback_draft":
                self.status_var.set(status_summary["status"])
                fallback_reason = result.get("planner_fallback_reason") or result.get("ai_fallback_reason") or "planner_fallback"
                validation = result.get("syntax_validation", result.get("validation_report", {}).get("syntax_validation", {}))
                report_path = result.get("report_path", "")
                self._append_log(f"[generate:fallback] {fallback_reason}")
                
                passed_str = str(validation.get('passed', 'unknown')).lower() if isinstance(validation, dict) else 'unknown'
                self.messagebox.showwarning(
                    "Fallback Draft Saved — Planner Failed",
                    (
                        "- Ollama planner output did not match the required schema.\n"
                        "- A deterministic fallback draft was saved only for inspection.\n"
                        "- This is not an AI-planned final success.\n\n"
                        f"Reason: {fallback_reason}\n"
                        f"Serialized draft validation: {passed_str}\n"
                        f"Final success: false\n"
                        f"Report:\n{report_path}"
                    ),
                )
            elif status_summary["state"] == "low_quality_draft":
                self.status_var.set(status_summary["status"])
                fail_msg = result.get("quality_gate_failure", "Quality gate failed.")
                self._append_log(f"[generate:warning] {fail_msg}")
                already_extreme = config.quality_mode.lower() in {"extreme ml", "extreme_ml", "extreme"}
                if already_extreme:
                    suggestion = "Try adding more reference levels or increasing object budget."
                else:
                    suggestion = "Try enabling Extreme ML mode or adding more reference levels."
                playability = result.get("playability_score", result.get("score_breakdown", {}).get("playability_safety", "N/A"))
                repair_loss = result.get("removed_object_ratio", "N/A")
                obj_count = result.get("final_object_count", result.get("num_objects", "N/A"))
                stopped = result.get("stopped_reason", "")
                diag_lines = [f"  Playability: {playability}", f"  Repair loss: {repair_loss}", f"  Final objects: {obj_count}"]
                if stopped:
                    diag_lines.append(f"  Stopped reason: {stopped}")
                diag_text = "\n".join(diag_lines)
                self.messagebox.showwarning(
                    status_summary["title"],
                    f"The generated level did not pass the Quality Gate.\n\n{fail_msg}\n\nDiagnostics:\n{diag_text}\n\nIt has been saved as a draft. {suggestion}"
                )
            else:
                self.status_var.set(status_summary["status"])
                
            output_path = save_res.get("resolved_output_path", "") if save_res else result.get("output_path", "")
            is_fallback = result.get('planner_fallback_used', False)
            score = result.get("final_score", result.get("score", {}).get("total", 0.0))
            score_log = "fallback_draft" if is_fallback else score
            report_for_log = result.get("validation_report", {})
            if not isinstance(report_for_log, dict):
                report_for_log = {}
            self._append_log(f"[generate] done: output={output_path}")
            self._append_log(f"[generate] saved: {output_path}")
            self._append_log(f"[generate] score={score_log}")
            self._append_log(f"[generate] ai_provider={result.get('ai_provider')} valid={result.get('valid')}")
            self._append_log(
                "[generate:status] "
                f"planner={result.get('planner_status', report_for_log.get('planner_status'))} "
                f"fallback={result.get('planner_fallback_used', report_for_log.get('planner_fallback_used'))} "
                f"quality_gate={result.get('quality_gate_passed')} "
                f"final_success={result.get('final_success')}"
            )
            removed_params = result.get("ollama_removed_unsupported_params", [])
            if removed_params:
                self._append_log(
                    "[ollama] removed unsupported params: "
                    + ", ".join(str(item) for item in removed_params)
                )
            candidates_count = len(result.get('candidate_reports', []))
            selected_cand = result.get('selected_candidate_id', 0) if candidates_count > 0 else 'null'
            is_fallback = result.get('planner_fallback_used', False)
            raw_ai = result.get('validation_report', {}).get('raw_ai_object_count', 0)
            raw_log = f"fallback_generated_objects={result.get('num_objects', 0)}" if is_fallback else f"raw_objects={raw_ai} candidate_ir_objects={raw_ai}"
            
            self._append_log(
                "[quality] "
                f"selected_candidate={selected_cand} "
                f"candidates={candidates_count} "
                f"{raw_log} "
                f"final_objects={result.get('num_objects', 0)} "
                f"repair_loss={result.get('removed_object_ratio', 0.0)} "
                f"drop_impact={result.get('drop_impact_score', 0.0)}"
            )
            for reason in result.get("quality_loss_reason_summary", [])[:5]:
                self._append_log(f"[quality:loss] {reason}")
            for candidate in result.get("candidate_reports", [])[:5]:
                self._append_log(
                    "[quality:candidate] "
                    f"id={candidate.get('candidate_id')} selected={candidate.get('selected')} "
                    f"score={candidate.get('score')} objects={candidate.get('object_count')} "
                    f"triggers={candidate.get('trigger_count')} reason={candidate.get('reject_reason', '')}"
                )
            if result.get("section_candidate_reports"):
                self._append_log(
                    "[quality:sections] "
                    f"selected_sections={result.get('selected_section_count', 0)} "
                    f"section_candidates={len(result.get('section_candidate_reports', []))} "
                    f"gate={result.get('quality_gate_report', {}).get('passed')}"
                )
            if config.open_output_folder_after_generation and output_path:
                try:
                    output_dir = str(Path(output_path).resolve().parent)
                    if os.name == "nt":
                        os.startfile(output_dir)  # type: ignore[attr-defined]
                except Exception as exc:  # noqa: BLE001
                    self._append_log(f"[generate:warning] failed to open output folder: {exc}")
            if status_summary["state"] == "final_success":
                self.messagebox.showinfo(status_summary["title"], f"Output:\n{output_path}")

        @safe_gui_callback
        def _save_report(self) -> None:
            result = getattr(self, "last_result", None)
            if not result:
                self.messagebox.showinfo("Save Report", "No generation report is available yet.")
                return
            output_path = Path(str(result.get("output_path", "outputs/generated.gmd")))
            report_path = output_path.with_suffix(".report.json")
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(json.dumps(sanitize_report(result.get("validation_report", result)), ensure_ascii=False, indent=2), encoding="utf-8")
            self._append_log(f"[report] saved: {report_path}")

        @safe_gui_callback
        def _open_output_folder(self) -> None:
            result = getattr(self, "last_result", None)
            if not result:
                self.messagebox.showinfo("Open Output Folder", "No output is available yet.")
                return
            output_dir = Path(str(result.get("output_path", "outputs/generated.gmd"))).resolve().parent
            if os.name == "nt":
                os.startfile(str(output_dir))  # type: ignore[attr-defined]
            else:
                self._append_log(f"[output] folder: {output_dir}")

        @safe_gui_callback
        def _save_feedback(self) -> None:
            example_id = self.app.state.last_learning_example_id
            if not example_id:
                self.messagebox.showinfo("Feedback", "Generate a level before saving feedback.")
                return
            tags = [
                tag.strip()
                for tag in self.v_feedback_tags.get().replace(";", ",").split(",")
                if tag.strip()
            ]
            good_bad = self.v_feedback_good_bad.get().strip().lower()
            if good_bad == "bad" and "bad" not in tags:
                tags.append("bad")
            if good_bad == "good" and "good" not in tags:
                tags.append("good")
            feedback_payload = {
                "user_rating": int(self.v_feedback_rating.get() or 0),
                "user_tags": tags,
                "user_notes": self.v_feedback_notes.get().strip(),
                "accepted_for_training": bool(self.v_feedback_include_training.get()),
            }
            if bool(self.v_save_generation_to_dataset_learning.get()):
                updated = update_dataset_feedback(example_id, feedback_payload, self.v_dataset_dir.get().strip() or "dataset")
            else:
                updated = update_learning_example_feedback(
                    example_id,
                    feedback_payload,
                    store_dir=self.v_learning_store_dir.get().strip() or None,
                )
            self._append_log(f"[feedback] saved={updated} example_id={example_id}")
            if not updated:
                self.messagebox.showwarning("Feedback", "Could not find the learning example to update.")

        @safe_gui_callback
        def _init_dataset(self) -> None:
            from gmdgen.dataset.bootstrap import initialize_dataset_structure
            res = initialize_dataset_structure(self.v_dataset_dir.get().strip() or "dataset")
            self._append_log(f"[dataset] Initialized. Created: {len(res['created'])}, Existed: {len(res['existed'])}")
            self.messagebox.showinfo("Dataset Bootstrap", f"Created {len(res['created'])} directories.")

        @safe_gui_callback
        def _run_reference_analysis(self) -> None:
            from gmdgen.dataset.reference_quality import evaluate_reference_levels
            self.status_var.set("Analyzing reference levels...")
            res = evaluate_reference_levels(self.v_dataset_dir.get().strip() or "dataset")
            self.status_var.set("Ready")
            if "error" in res:
                self.messagebox.showerror("Reference Analysis", res["error"])
                return
            self._append_log(f"[reference-analysis] valid: {res['valid_count']}, invalid: {res['invalid_count']}, motifs: {res['motifs_estimated']}")
            self.messagebox.showinfo("Reference Analysis", f"Found {res['valid_count']} valid reference levels.")

        @safe_gui_callback
        def _run_quality_eval(self) -> None:
            self._append_log("[eval] Running offline quality suite...")
            self.status_var.set("Running Quality Eval...")
            
            def task():
                from gmdgen.eval.suite import QualityEvalSuite
                from gmdgen.eval.cases import EvalCase
                import traceback
                try:
                    out_dir = Path("outputs") / "eval"
                    suite = QualityEvalSuite(out_dir)
                    case = EvalCase("offline_check", "tests/fixtures/audio/clicks.wav")
                    res = suite.run_case(case, is_live_ollama=False)
                    msg = f"Passed: {res.passed}, Report: {res.report_path}"
                    self.root.after(0, lambda m=msg: self._append_log(f"[eval] done: {m}"))
                except Exception as e:
                    self.root.after(0, lambda e=e: self._append_log(f"[eval] Error: {e} {traceback.format_exc()}"))
                self.root.after(0, lambda: self.status_var.set("Ready"))

            if not getattr(self, "worker_thread", None) or not self.worker_thread.is_alive():  # type: ignore
                self.worker_thread = threading.Thread(target=task, daemon=True)
                self.worker_thread.start()

        @safe_gui_callback
        def _run_live_eval(self) -> None:
            self._append_log("[eval] Running live Ollama eval...")
            self.status_var.set("Running Live Eval...")
            
            def task():
                from gmdgen.eval.live_ollama_eval import run_live_eval
                try:
                    out_dir = Path("outputs") / "eval"
                    res = run_live_eval(out_dir)
                    msg = f"Skipped: {res.get('skipped')}, Passed: {res.get('passed')}"
                    self.root.after(0, lambda m=msg: self._append_log(f"[eval:live] done: {m}"))
                except Exception as e:
                    self.root.after(0, lambda e=e: self._append_log(f"[eval:live] Error: {e}"))
                self.root.after(0, lambda: self.status_var.set("Ready"))

            if not getattr(self, "worker_thread", None) or not self.worker_thread.is_alive():  # type: ignore
                self.worker_thread = threading.Thread(target=task, daemon=True)
                self.worker_thread.start()

        @safe_gui_callback
        def _run_geode_check(self) -> None:
            from gmdgen.gd.geode_protocol import GeodeBridgeConfig
            from gmdgen.gd.geode_external_bridge import ExternalGeodeBridge
            cfg = GeodeBridgeConfig(enabled=True, helper_executable_path="geode_helper.exe")
            bridge = ExternalGeodeBridge(cfg)
            ver = bridge.get_version()
            self._append_log(f"[geode] Version check: {ver}")
            self.messagebox.showinfo("Geode Check", f"Geode Helper version: {ver}")

    root = tk.Tk()
    GuiMainWindow(root, app)
    root.mainloop()
    return 0


def main() -> None:
    raise_code = launch_gui()
    if raise_code != 0:
        raise SystemExit(raise_code)


def _headless_environment() -> bool:
    if os.environ.get("GMDGEN_HEADLESS", "").strip() == "1":
        return True
    if os.name == "nt":
        return False
    return not bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
