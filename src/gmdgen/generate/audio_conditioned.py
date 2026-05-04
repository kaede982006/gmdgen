# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from gmdgen.ai.context import (
    LocalKeywordRetriever,
    load_context_documents,
    load_reference_levels,
    summarize_context_documents,
)
from gmdgen.ai.dataset_index import (
    build_dataset_index,
    dataset_cache_path,
    dataset_context_chunk,
    load_dataset_index,
    resolve_dataset_dir,
)
from gmdgen.ai.factory import create_ai_provider_from_config
def effective_ai_provider(config: dict) -> str:
    return str(config.get("ai_provider", "ollama")).strip().lower()

def local_test_provider_allowed(config: dict) -> bool:
    return bool(config.get("allow_local_test_provider", False))

def sanitize_ai_error(text: str) -> str:
    import re
    val = str(text)
    val = re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "sk-[REDACTED]", val)
    return val

AI_GENERATION_FAILED_MESSAGE = "AI generation failed."
from gmdgen.errors import QualityGateFailure
from gmdgen.ai.prompts import (
    summarize_audio_for_model,
    summarize_reference_style_for_model,
    summarize_sections_for_model,
    summarize_trigger_schema_for_model,
    truncate_context_to_budget,
)
from gmdgen.ai.context import summarize_reference_level
from gmdgen.ai.schemas import (
    AILevelPlanRequest,
    AILevelPlanResponse,
    AIPlanConversionResult,
    convert_ai_response_to_plans,
)
from gmdgen.audio.analysis import AudioAnalysisResult, AudioFeatures, BeatFeature, SectionFeature, analyze_audio
from gmdgen.audio.paths import resolve_audio_file_from_config
from gmdgen.data.preprocess import extract_level_header, split_level_objects
from gmdgen.features.tokenizer import extract_object_id, extract_object_number, rewrite_object_xy
from gmdgen.gd.guidelines import build_guideline_string
from gmdgen.gd.plans import (
    AudioEvent,
    GameplayEvent,
    ObjectPlan,
    SectionPlan,
    TriggerMode,
    TriggerPlan,
    ValidationReport,
    apply_trigger_defaults,
    build_level_settings,
    plans_to_level_objects,
    validate_trigger_plan_schema,
)
from gmdgen.gd.time_mapping import (
    SpeedObject,
    SpeedState,
    build_beat_x_map,
    build_and_sort_speed_objects,
    normalize_speed_state,
    pos_for_time_like_gd,
    round_trip_error_report,
    speed_state_at_time,
    sync_error_for_x,
    time_for_pos_like_gd,
)
from gmdgen.gd.geode_bridge import NullGeodeBridge, compare_time_mapping_with_geode
from gmdgen.generate.repairer import repair_level_objects, repair_report_to_dict
from gmdgen.generate.editor_safety import validate_save_string_safety
from gmdgen.generate.group_allocation import allocate_trigger_target_groups
from gmdgen.generate.playability import repair_playability_plans, validate_trajectory_playability
from gmdgen.generate.scoring import compute_audio_conditioned_score
from gmdgen.generate.quality import (
    PlanSnapshot,
    build_candidate_report,
    build_quality_feedback_prompt,
    build_repair_quality_report,
    compute_actual_density_by_section_from_objects,
    compute_buildup_progression_score,
    compute_density_target_by_section,
    compute_density_target_error,
    compute_drop_impact_score,
    diff_snapshots,
    snapshot_from_level_objects,
    snapshot_from_plans,
    summarize_quality_loss,
)
from gmdgen.generate.role_mapping import diversify_object_ids
from gmdgen.generate.quality_gate import QualityGateThresholds, evaluate_quality_gate
from gmdgen.generate.section_pipeline import run_section_generation_pipeline
from gmdgen.generate.style_bank import MotifBank, build_motif_bank_from_files
from gmdgen.generate.validator import round_trip_validate, validate_gmd_file, validate_playability_v1
from gmdgen.learning.feature_extractor import (
    load_learned_data_store,
    summarize_learned_data_for_prompt,
)
from gmdgen.learning.store import summarize_learning_examples_for_context
from gmdgen.io.gmd_decoder import (
    decode_level_data,
    encode_level_data,
    encode_level_description,
)
from gmdgen.io.gmd_parser import parse_gmd_file
from gmdgen.io.gmd_writer import write_gmd_file
from gmdgen.representation.object_classifier import ObjectClass, classify


import time
from gmdgen.generate.materializer import MaterializationConfig, materialize_level_plans

_GAMEPLAY_MODES = ["cube", "ship", "ball", "ufo", "wave", "robot", "spider"]
_ORB_IDS = ["36", "84", "141", "1022"]
_PAD_IDS = ["35", "140"]


class GroupAllocator:
    def __init__(self, *, max_group_id: int, policy: str = "sequential") -> None:
        self.max_group_id = max(1, int(max_group_id))
        self.policy = policy
        self._next_id = 1
        self._section_cache: dict[int, int] = {}

    def allocate(self, *, section_id: int | None = None) -> int:
        if self.policy == "section_scoped" and section_id is not None:
            if section_id not in self._section_cache:
                self._section_cache[section_id] = self._allocate_next()
            return self._section_cache[section_id]
        return self._allocate_next()

    def _allocate_next(self) -> int:
        if self._next_id > self.max_group_id:
            raise ValueError(f"group id budget exhausted at max_group_id={self.max_group_id}")
        group_id = self._next_id
        self._next_id += 1
        return group_id


def generate_audio_synced_level_from_config(config: dict[str, Any]) -> dict[str, Any]:
    start_total_time = time.time()
    audio_path = resolve_audio_file_from_config(config)
    if audio_path is None:
        raise ValueError("audio_file is required for audio-conditioned generation")
    audio_file_name = audio_path.name

    output_dir = Path(config.get("output_dir", "outputs"))
    output_name = str(config.get("output_name", "audio_synced_level"))
    output_dir.mkdir(parents=True, exist_ok=True)
    save_draft = bool(config.get("save_report", True)) or bool(config.get("save_draft", True))
    report_path = output_dir / f"{output_name}_report.json"
    target_duration = _optional_float(config, "target_duration", None)
    start_speed = normalize_speed_state(config.get("start_speed", "normal"))
    song_offset = float(config.get("song_offset", 0.0))
    allow_speed_portals = bool(config.get("allow_speed_portals", False))
    allow_triggers = bool(config.get("allow_triggers", True))
    speed_portal_policy = str(config.get("speed_portal_policy", "conservative"))
    difficulty = _difficulty_value(config.get("difficulty", 0.5))
    sync_strength = float(config.get("audio_sync_strength", config.get("sync_strength", 0.75)))
    object_budget = int(config.get("object_budget", config.get("num_objects", 1200) or 1200))
    safe_mode = bool(config.get("safe_mode", True))
    editor_safe_mode = bool(config.get("editor_safe_mode", True))
    max_group_id = int(config.get("max_group_id", 9999))
    max_events_per_beat = int(config.get("max_events_per_beat", 2))
    beat_snap_tolerance = float(config.get("beat_snap_tolerance", 0.08))
    onset_event_threshold = float(config.get("onset_event_threshold", 0.35))
    group_id_policy = str(config.get("group_id_policy", "sequential"))
    write_decoded_preview = bool(config.get("write_decoded_preview", True))
    generated_author = str(config.get("generated_author", "gmdgen"))
    style_reference_level = config.get("style_reference_level") or config.get("template_level")
    ai_provider_name = effective_ai_provider(config)
    
    # Materialization Config
    mat_config = MaterializationConfig(
        object_multiplier=float(config.get("object_multiplier", 1.0)),
        target_object_count=config.get("target_object_count"),
        min_object_count=int(config.get("min_object_count", 100)),
        max_object_count=int(config.get("max_object_count", 40000)),
        detail_density=float(config.get("detail_density", 0.5)),
        decoration_density=float(config.get("decoration_density", 0.5)),
        gameplay_density=float(config.get("gameplay_density", 0.5)),
        sync_accent_density=float(config.get("sync_accent_density", 0.5)),
        fast_materialization=config.get("fast_materialization", True),
        seed=int(config.get("seed", 42)),
    )
    
    max_gen_seconds = config.get("max_generation_seconds", 600)

    start_audio_time = time.time()
    features: AudioAnalysisResult = analyze_audio(
        str(audio_path),
        song_offset=song_offset,
        target_duration=target_duration,
        backend=str(config.get("audio_backend", "auto")),
    )
    audio_analysis_seconds = time.time() - start_audio_time
    
    confidence_report = features.confidence_report
    confidence_value = float(confidence_report.overall if confidence_report else features.confidence)
    if confidence_value < 0.35 and speed_portal_policy == "aggressive":
        speed_portal_policy = "musical"
    guideline_string = build_guideline_string(features.beat_times)
    level_settings = build_level_settings(
        start_speed=start_speed,
        song_offset=song_offset,
        custom_song=True,
        guideline_string=guideline_string,
    )

    style_profile = _load_style_profile(style_reference_level)
    learned_prompt_context = _load_learned_prompt_context(config)
    if learned_prompt_context:
        style_profile = _merge_learned_style_profile(style_profile, learned_prompt_context)
    
    start_plan_time = time.time()
    speed_candidates = _plan_speed_portals(
        features.sections,
        allow_speed_portals=allow_speed_portals,
        policy=speed_portal_policy,
        start_speed=start_speed,
        difficulty=difficulty,
    )
    speed_objects = build_and_sort_speed_objects(
        speed_candidates,
        start_speed=start_speed,
        song_offset=song_offset,
    )
    beat_x_map = build_beat_x_map(
        features.beats or features.beat_features,
        speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
    )
    time_x_report = round_trip_error_report(
        features.beats or features.beat_features,
        beat_x_map,
        speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
    )
    geode_bridge = config.get("geode_bridge") or NullGeodeBridge()
    geode_parity_report = compare_time_mapping_with_geode(
        geode_bridge,
        (features.beats or features.beat_features)[:16],
        speed_objects,
        start_speed,
        song_offset,
    )
    section_plans = _plan_sections(
        features.sections,
        features=features,
        speed_objects=speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        difficulty=difficulty,
        sync_strength=sync_strength,
    )
    audio_events = _build_audio_events(features)
    motif_library = _load_motif_library(config)
    motif_library.extend(_learned_motif_library(learned_prompt_context))

    allocator = GroupAllocator(max_group_id=max_group_id, policy=group_id_policy)
    
    # Deterministic Baseline Materialization
    start_mat_time = time.time()
    object_plans = materialize_level_plans(
        section_plans,
        config=mat_config,
        audio_features=features,
        motif_library=motif_library,
        style_profile=style_profile,
        total_object_budget=object_budget,
        speed_objects=speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
    )
    
    # Add speed portals manually as they are critical
    object_plans.extend(
        _speed_portal_object_plans(speed_objects, editor_safe_mode=editor_safe_mode)
    )
    materialization_seconds = time.time() - start_mat_time

    trigger_objects, trigger_plans = _plan_trigger_events(
        features,
        speed_objects=speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        allow_triggers=allow_triggers,
        threshold=onset_event_threshold,
        sync_strength=sync_strength,
        object_budget=object_budget,
        allocator=allocator,
        safe_mode=safe_mode,
    )
    object_plans.extend(trigger_objects)
    deterministic_planning_seconds = time.time() - start_plan_time

    start_ai_time = time.time()
    ai_conversion, ai_metadata = _maybe_apply_ai_provider(
        config=config,
        features=features,
        section_plans=section_plans,
        time_x_report=time_x_report,
        style_profile=style_profile,
        object_budget=object_budget,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        start_speed=start_speed.value,
        song_offset=song_offset,
        # Pass materialization components to AI loop if needed
        motif_library=motif_library,
        mat_config=mat_config,
        speed_objects=speed_objects,
        song_offset_val=song_offset,
        start_speed_val=start_speed,
    )
    ai_planning_seconds = time.time() - start_ai_time

    if ai_conversion.valid:
        if ai_conversion.object_plans:
            # AI suggested objects are prioritized if valid
            # In new mode, AI only suggests a few, materializer expands
            # For now, we keep the simple merge
            ai_budget = max(0, object_budget - len(object_plans) - len(trigger_plans))
            object_plans.extend(ai_conversion.object_plans[:ai_budget])
        if ai_conversion.trigger_plans:
            trigger_budget = max(0, object_budget - len(object_plans) - len(trigger_plans))
            trigger_plans.extend(ai_conversion.trigger_plans[:trigger_budget])
    elif ai_provider_name != "local_test_only":
        detail = "; ".join(ai_conversion.errors) or "AI output validation failed"
        raise ValueError(f"{AI_GENERATION_FAILED_MESSAGE} {detail}")

    start_repair_time = time.time()
    pre_repair_snapshot = snapshot_from_plans(
        "validated_plan",
        object_plans,
        trigger_plans,
        section_plans,
        warnings=list(ai_conversion.warnings),
        fatal_errors=list(ai_conversion.errors),
    )
    playability_repair_report = repair_playability_plans(
        object_plans,
        section_plans,
        difficulty=difficulty,
    )
    validation_report = _repair_and_validate_plans(
        object_plans=object_plans,
        trigger_plans=trigger_plans,
        section_plans=section_plans,
        speed_objects=speed_objects,
        features=features,
        start_speed=start_speed,
        song_offset=song_offset,
        object_budget=object_budget,
        beat_snap_tolerance=beat_snap_tolerance,
        max_events_per_beat=max_events_per_beat,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        editor_safe_mode=editor_safe_mode,
    )
    repair_seconds = time.time() - start_repair_time
    
    start_val_time = time.time()
    validation_report.playability_breakdown.update(playability_repair_report.to_dict())
    if playability_repair_report.converted_gameplay_to_decoration or playability_repair_report.simplified_dense_orb_chain:
        validation_report.add_warning(
            "playability_repair_applied: "
            f"converted={playability_repair_report.converted_gameplay_to_decoration} "
            f"simplified={playability_repair_report.simplified_dense_orb_chain}"
        )
    validation_report.time_x_avg_error = float(time_x_report["average_error"])
    validation_report.time_x_max_error = float(time_x_report["max_error"])
    validation_report.audio_backend = features.backend
    validation_report.audio_file = audio_file_name
    validation_report.audio_file_name = audio_file_name
    validation_report.detected_bpm = float(features.bpm)
    validation_report.beat_count = len(features.beat_times)
    validation_report.onset_count = len(features.onset_times)
    validation_report.section_count = len(section_plans)
    validation_report.speed_object_count = len(speed_objects)
    validation_report.generated_trigger_count = len(trigger_plans)
    validation_report.metrics["time_x_invalid_count"] = int(time_x_report["invalid_count"])
    validation_report.metrics["time_x_checked_count"] = int(time_x_report["checked_count"])
    validation_report.geode_available = bool(geode_parity_report.available)
    validation_report.geode_version = str(geode_bridge.get_version() or "")
    validation_report.geode_time_x_checked = bool(geode_parity_report.checked)
    validation_report.geode_time_x_avg_error = float(geode_parity_report.average_abs_time_error)
    validation_report.geode_time_x_max_error = float(geode_parity_report.max_abs_time_error)
    validation_report.geode_parity_passed = bool(geode_parity_report.passed)
    validation_report.geode_warnings.extend(geode_parity_report.warnings)
    _apply_ai_metadata(validation_report, ai_metadata, ai_conversion)
    repaired_plan_snapshot = snapshot_from_plans(
        "repaired_plan",
        object_plans,
        trigger_plans,
        section_plans,
        warnings=list(validation_report.warnings),
        fatal_errors=list(validation_report.issues),
    )
    validation_report.playability_warnings.extend(
        _playability_warnings(
            object_plans,
            section_plans=section_plans,
            difficulty=difficulty,
        )
    )
    validation_report.playability_warnings.extend(
        warning.message
        for warning in validate_playability_v1(
            object_plans=object_plans,
            section_plans=section_plans,
            speed_objects=speed_objects,
            difficulty=difficulty,
        )
    )
    # trajectory_warnings = validate_trajectory_playability(
    #     section_plans,
    #     gameplay_events, # Need to pass something else if gameplay_events not used
    #     object_plans,
    #     difficulty=difficulty,
    # )
    # validation_report.playability_warnings.extend(warning.message for warning in trajectory_warnings)
    # validation_report.metrics["trajectory_warning_count"] = len(trajectory_warnings)

    trigger_mode = TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED
    level_objects = plans_to_level_objects(
        object_plans,
        trigger_plans,
        trigger_mode=trigger_mode,
    )
    validation_seconds = time.time() - start_val_time
    
    if safe_mode or editor_safe_mode:
        level_objects, repair_report = repair_level_objects(
            level_objects,
            fix_x_monotone=True,
            fix_group_ids=True,
            fix_orphan_triggers=True,
            fix_density=True,
            fix_grid_snap=False,
            fix_duplicates=True,
            max_density_per_grid=max(4, max_events_per_beat * 4),
            object_budget=object_budget,
            max_group_id=max_group_id,
            safe_mode=safe_mode,
        )
        validation_report.metrics["string_repair_total_fixed"] = repair_report.total_fixed
    else:
        repair_report = None
    final_plan_snapshot = snapshot_from_level_objects(
        "final_encoded_plan",
        level_objects,
        section_plans,
        warnings=validation_report.editor_validity_warnings,
        fatal_errors=validation_report.issues,
    )
    pre_score_density_target = compute_density_target_by_section(section_plans)
    pre_score_actual_density = compute_actual_density_by_section_from_objects(level_objects, section_plans)
    validation_report.metrics["density_alignment_score"] = max(
        0.0,
        1.0 - compute_density_target_error(pre_score_density_target, pre_score_actual_density),
    )
    validation_report.metrics["drop_impact_score"] = compute_drop_impact_score(section_plans, pre_score_actual_density)
    validation_report.metrics.update(_learned_quality_metrics(level_objects, style_profile, pre_score_actual_density))
    validation_report.metrics["repair_loss_ratio"] = max(
        0.0,
        (pre_repair_snapshot.object_count + pre_repair_snapshot.trigger_count - final_plan_snapshot.object_count - final_plan_snapshot.trigger_count)
        / max(1, pre_repair_snapshot.object_count + pre_repair_snapshot.trigger_count),
    )

    level_header = style_profile.get("level_header") or "kA11,0"
    decoded_level_data = _compose_level_data(level_header, level_objects)
    round_trip = round_trip_validate(
        decoded_level_data,
        safe_mode=safe_mode,
        max_group_id=max_group_id,
    )
    editor_safety_report = validate_save_string_safety(
        decoded_level_data,
        trigger_mode,
        max_group_id=max_group_id,
    )
    validation_report.editor_safety_report = editor_safety_report.to_dict()
    validation_report.round_trip_valid = bool(round_trip["valid"])
    validation_report.metrics["editor_safety_score"] = editor_safety_report.to_dict()["score"]
    if not round_trip["valid"]:
        validation_report.editor_validity_warnings.extend(
            str(issue) for issue in round_trip["issues"]
        )
    if editor_safety_report.fatal_errors:
        for issue in editor_safety_report.fatal_errors:
            validation_report.add_issue(issue)
            validation_report.editor_validity_warnings.append(issue)
    else:
        validation_report.editor_validity_warnings.extend(editor_safety_report.warnings[:8])
    encoded_level_data = encode_level_data(decoded_level_data)

    tags = {
        "k1": ("i", "900000001"),
        "k2": ("s", output_name),
        "k3": (
            "s",
            encode_level_description(
                "Audio-conditioned GD level generated from structured beat/onset plan."
            ),
        ),
        "k4": ("s", encoded_level_data),
        "k5": ("s", generated_author),
        "k95": ("i", str(len(level_objects))),
    }

    from gmdgen.output.save import save_level_output
    save_res = save_level_output(
        encoded_level_data=encoded_level_data,
        output_path=output_dir / f"{output_name}.gmd",
        tags=tags,
        default_name=output_name
    )
    output_path = Path(save_res.resolved_output_path) if save_res.success else (output_dir / f"{output_name}.gmd")

    decoded_preview_path = output_dir / f"{output_name}.decoded.txt"
    if write_decoded_preview and save_res.success:
        decoded_preview_path.write_text(decoded_level_data.replace(";", ";\n"), encoding="utf-8")

    is_valid, issues = validate_gmd_file(
        output_path,
        object_budget=object_budget,
        max_group_id=max_group_id,
        safe_mode=safe_mode,
        check_round_trip=True,
    )
    if not is_valid:
        for issue in issues:
            validation_report.add_issue(issue)
            validation_report.editor_validity_warnings.append(issue)

    start_score_time = time.time()
    score = compute_audio_conditioned_score(
        level_objects,
        audio_features=features,
        speed_objects=speed_objects,
        start_speed=start_speed,
        song_offset=song_offset,
        beat_snap_tolerance=beat_snap_tolerance,
        object_budget=object_budget,
        editor_issues=validation_report.editor_validity_warnings + validation_report.issues,
        playability_warning_count=len(validation_report.playability_warnings),
        section_plans=section_plans,
        quality_metrics=validation_report.metrics,
    )
    scoring_seconds = time.time() - start_score_time
    
    score_dict = score.to_dict()
    validation_report.final_object_count = len(level_objects)
    validation_report.score = float(score_dict["total"])
    validation_report.score_breakdown = score_dict
    validation_report.beat_sync_avg_error = _average_alignment_error(
        [plan.beat_aligned_time for plan in object_plans if plan.beat_aligned_time is not None],
        features.beat_times,
    )
    validation_report.onset_sync_avg_error = _average_alignment_error(
        [trigger.beat_aligned_time for trigger in trigger_plans if trigger.beat_aligned_time is not None],
        features.onset_times or features.beat_times,
    )
    validation_report.generation_mode = "audio_conditioned"
    _finalize_quality_report(
        validation_report=validation_report,
        ai_metadata=ai_metadata,
        pre_repair_snapshot=pre_repair_snapshot,
        repaired_plan_snapshot=repaired_plan_snapshot,
        final_plan_snapshot=final_plan_snapshot,
        repair_report=repair_report,
        section_plans=section_plans,
        level_objects=level_objects,
    )
    validation_report.quality_mode = str(config.get("quality_mode", "Balanced"))
    validation_report.quality_gate_report = evaluate_quality_gate(
        validation_report.to_dict(),
        QualityGateThresholds(
            min_score=float(config.get("quality_gate_min_score", config.get("min_acceptable_score", 0.35))),
            min_object_count=int(config.get("quality_gate_min_object_count", config.get("min_final_object_count", 8))),
            max_repair_loss=float(config.get("quality_gate_max_repair_loss", config.get("max_repair_loss_ratio", 0.55))),
            min_drop_impact=float(config.get("quality_gate_min_drop_impact", config.get("min_drop_impact_score", 0.2))),
            min_density_alignment=float(config.get("quality_gate_min_density_alignment", 0.2)),
            min_editor_safety=float(config.get("quality_gate_min_editor_safety", 0.5)),
            min_playability=float(config.get("quality_gate_min_playability", 0.5)),
        ),
    ).to_dict()
    
    total_generation_seconds = time.time() - start_total_time
    validation_report.metrics.update({
        "audio_analysis_seconds": audio_analysis_seconds,
        "deterministic_planning_seconds": deterministic_planning_seconds,
        "materialization_seconds": materialization_seconds,
        "ai_planning_seconds": ai_planning_seconds,
        "repair_seconds": repair_seconds,
        "validation_seconds": validation_seconds,
        "scoring_seconds": scoring_seconds,
        "total_generation_seconds": total_generation_seconds,
    })

    if bool(config.get("enforce_quality_gate", True)) and not validation_report.quality_gate_report.get("passed", False):
        failure_msg = "quality_gate_failed: " + "; ".join(validation_report.quality_gate_report.get("failures", [])[:4])
        validation_report.add_issue(failure_msg)
        if save_draft and report_path:
            _save_final_report(validation_report, report_path)
        raise QualityGateFailure(failure_msg, details=validation_report.to_dict())
    
    if bool(config.get("ollama_save_debug_artifacts", False) or config.get("save_debug_bundle", False)):
        validation_report.ai_debug_artifact_path = _save_quality_debug_artifacts(
            config=config,
            output_dir=output_dir,
            output_name=output_name,
            validation_report=validation_report,
        )

    result = {
        "output_path": str(output_path),
        "save_result": save_res.to_dict() if save_res else None,
        "decoded_preview_path": str(decoded_preview_path) if write_decoded_preview else None,
        "generation_mode": "audio_conditioned",
        "audio_file": audio_file_name,
        "audio_file_name": audio_file_name,
        "audio_backend": features.backend,
        "duration": round(features.duration, 4),
        "bpm": round(features.bpm, 4),
        "detected_bpm": round(features.bpm, 4),
        "analysis_confidence": round(features.confidence, 4),
        "audio_confidence_report": asdict(confidence_report) if confidence_report else {},
        "num_onsets": len(features.onset_times),
        "onset_count": len(features.onset_times),
        "num_beats": len(features.beat_times),
        "beat_count": len(features.beat_times),
        "num_sections": len(section_plans),
        "section_count": len(section_plans),
        "num_speed_portals": len(speed_objects),
        "num_objects": len(level_objects),
        "num_triggers": len(trigger_plans),
        "score": score_dict,
        "score_breakdown": score_dict,
        "plan_snapshots": validation_report.plan_snapshots,
        "plan_diffs": validation_report.plan_diffs,
        "candidate_reports": validation_report.candidate_reports,
        "selected_candidate_id": validation_report.selected_candidate_id,
        "section_candidate_reports": validation_report.section_candidate_reports,
        "selected_section_count": validation_report.selected_section_count,
        "global_consistency_report": validation_report.global_consistency_report,
        "quality_gate_report": validation_report.quality_gate_report,
        "quality_mode": validation_report.quality_mode,
        "quality_loss_reason_summary": validation_report.quality_loss_reason_summary,
        "repair_quality_report": validation_report.repair_quality_report,
        "removed_object_ratio": validation_report.removed_object_ratio,
        "removed_trigger_ratio": validation_report.removed_trigger_ratio,
        "drop_impact_score": validation_report.metrics.get("drop_impact_score", 0.0),
        "density_alignment_score": validation_report.metrics.get("density_alignment_score", 0.0),
        "object_diversity_score": validation_report.metrics.get("object_diversity_score", 0.0),
        "learned_style_match_score": validation_report.metrics.get("learned_style_match_score", 0.0),
        "learned_motif_match_score": validation_report.metrics.get("learned_motif_match_score", 0.0),
        "learned_density_match_score": validation_report.metrics.get("learned_density_match_score", 0.0),
        "learned_trigger_usage_score": validation_report.metrics.get("learned_trigger_usage_score", 0.0),
        "BeatSyncScore": score_dict["beat_sync"],
        "OnsetSyncScore": score_dict["onset_sync"],
        "EnergyDensityScore": score_dict["energy_density"],
        "TimeXConsistencyScore": score_dict["time_to_x_consistency"],
        "TriggerValidityScore": score_dict["trigger_validity"],
        "EditorValidityScore": score_dict["editor_validity"],
        "PlayabilitySafetyScore": score_dict["playability"],
        "final_score": score_dict["total"],
        "time_x_average_error": validation_report.time_x_avg_error,
        "time_x_max_error": validation_report.time_x_max_error,
        "round_trip_valid": validation_report.round_trip_valid,
        "editor_safety": validation_report.editor_safety_report,
        "editor_safety_status": validation_report.editor_safety_report.get("valid", False),
        "playability_warning_count": len(validation_report.playability_warnings),
        "trigger_schema_warning_count": sum(1 for warning in validation_report.warnings if "trigger_" in warning),
        "ai_provider": validation_report.ai_provider,
        "ai_model": validation_report.ai_model,
        "ai_used": validation_report.ai_used,
        "ai_fallback_used": validation_report.ai_fallback_used,
        "ai_fallback_reason": validation_report.ai_fallback_reason,
        "ai_context_chunks_used": validation_report.ai_context_chunks_used,
        "ai_response_valid": validation_report.ai_response_valid,
        "ai_output_object_count": validation_report.ai_output_object_count,
        "ai_output_trigger_count": validation_report.ai_output_trigger_count,
        "ai_debug_artifact_path": validation_report.ai_debug_artifact_path,
        "ai_required": validation_report.ai_required,
        "ai_error_type": validation_report.ai_error_type,
        "ai_error_message_sanitized": validation_report.ai_error_message_sanitized,
        "ai_retry_count": validation_report.ai_retry_count,
        "ollama_removed_unsupported_params": validation_report.ollama_removed_unsupported_params,
        "ollama_param_retry_count": validation_report.ollama_param_retry_count,
        "ollama_model_capabilities": validation_report.ollama_model_capabilities,
        "provider_chain": validation_report.provider_chain,
        "ai_calls_used": validation_report.ai_calls_used,
        "ai_cache_hits": validation_report.ai_cache_hits,
        "ai_request_hashes": validation_report.ai_request_hashes,
        "local_fallback_used": validation_report.local_fallback_used,
        "ollama_only_audit_passed": validation_report.ollama_only_audit_passed,
        "ai_normalization_warnings": validation_report.ai_normalization_warnings,
        "pruned_trigger_property_count": validation_report.pruned_trigger_property_count,
        "normalized_object_role_count": validation_report.normalized_object_role_count,
        "normalized_easing_count": validation_report.normalized_easing_count,
        "fatal_validation_issue_count": validation_report.fatal_validation_issue_count,
        "nonfatal_validation_warning_count": validation_report.nonfatal_validation_warning_count,
        "ai_output_preview": ai_metadata.get("ai_output_preview", {}),
        "valid": validation_report.valid,
        "issues": validation_report.issues,
        "validation_warnings": validation_report.warnings + validation_report.playability_warnings + validation_report.editor_validity_warnings,
        "validation_report": validation_report.to_dict(),
        "time_x_report": time_x_report,
        "geode_parity_report": geode_parity_report.to_dict(),
        "geode_available": validation_report.geode_available,
        "geode_time_x_checked": validation_report.geode_time_x_checked,
        "geode_parity_passed": validation_report.geode_parity_passed,
        "repair": repair_report_to_dict(repair_report) if repair_report else {},
        "level_settings": {
            "speed": level_settings.speed.value,
            "song_offset": level_settings.song_offset,
            "custom_song": level_settings.custom_song,
            "guideline_entries": len(features.beat_times),
        },
        "section_plan": [_section_plan_to_dict(plan) for plan in section_plans[:32]],
        "speed_plan": [asdict(speed_object) | {"speed_state": speed_object.speed_state.value} for speed_object in speed_objects],
        "gameplay_event_preview": [asdict(event) for event in []], # gameplay_events not available anymore
        "audio_event_preview": [asdict(event) for event in audio_events[:32]],
        "report_path": str(report_path),
        "timing": validation_report.metrics,
    }
    if bool(config.get("debug_paths", False)):
        result["audio_file_full_path"] = str(audio_path)
    
    if save_draft and report_path:
        _save_final_report(validation_report, report_path)
        
    return _ensure_num_sections_for_report(result)


def _save_final_report(report: ValidationReport, path: Path) -> None:
    path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def _maybe_apply_ai_provider(
    *,
    config: dict[str, Any],
    features: AudioAnalysisResult,
    section_plans: list[SectionPlan],
    time_x_report: dict[str, Any],
    style_profile: dict[str, Any],
    object_budget: int,
    max_group_id: int,
    safe_mode: bool,
    start_speed: str,
    song_offset: float,
    motif_library: list[dict[str, Any]] = None,
    mat_config: MaterializationConfig = None,
    speed_objects: list[Any] = None,
    song_offset_val: float = 0.0,
    start_speed_val: SpeedState = SpeedState.NORMAL,
) -> tuple[AIPlanConversionResult, dict[str, Any]]:
    start_ai_loop_time = time.time()
    provider_name = effective_ai_provider(config)
    
    # Consistently read flags
    use_ai_planner = bool(config.get("use_ai_planner", config.get("enable_ai_planning", False)))
    require_ai_planning = bool(config.get("require_ai_planning", config.get("fail_on_ai_planning_error", False)))
    has_explicit_client = config.get("ollama_client") is not None

    metadata: dict[str, Any] = {
        "ai_provider": provider_name,
        "ai_model": _provider_model_name(config, provider_name),
        "ai_used": False,
        "ai_planning_used": False,
        "ai_planning_attempted": False,
        "deterministic_fallback_used": False,
        "ai_planning_error": "",
        "ai_fallback_used": False,
        "ai_fallback_reason": "",
        "ai_context_chunks_used": 0,
        "ai_response_valid": False,
        "ai_output_object_count": 0,
        "ai_output_trigger_count": 0,
        "ai_debug_artifact_path": "",
        "ai_required": bool(config.get("real_generation_requires_ollama", config.get("real_generation_requires_external_ai", True))) or require_ai_planning,
        "ai_error_type": "",
        "ai_error_message_sanitized": "",
        "ai_retry_count": 0,
        "ollama_sanitized_params": [],
        "ollama_removed_unsupported_params": [],
        "ollama_param_retry_count": 0,
        "ollama_model_capabilities": {},
        "provider_chain": [],
        "ai_calls_used": 0,
        "ai_cache_hits": 0,
        "ai_request_hashes": [],
        "local_fallback_used": False,
        "ai_normalization_warnings": [],
        "pruned_trigger_property_count": 0,
        "normalized_object_role_count": 0,
        "normalized_easing_count": 0,
        "fatal_validation_issue_count": 0,
        "nonfatal_validation_warning_count": 0,
        "candidate_reports": [],
        "selected_candidate_id": 0,
        "plan_snapshots": [],
        "plan_diffs": [],
        "raw_ai_object_count": 0,
        "raw_ai_trigger_count": 0,
        "ai_output_preview": {"object_plans": [], "trigger_plans": []},
    }
    
    if provider_name == "local_test_only":
        metadata["ai_model"] = "local-heuristic"
        metadata["ai_response_valid"] = True
        metadata["ai_fallback_reason"] = "This output was generated without Ollama and is for testing only."
        return AIPlanConversionResult(
            response=AILevelPlanResponse(provider="local_test_only", model="local-heuristic")
        ), metadata

    # If AI planning is not explicitly enabled, return local deterministic plan
    if not use_ai_planner and not has_explicit_client:
        metadata["ai_provider"] = "local"
        metadata["ai_model"] = "local-deterministic"
        metadata["ai_response_valid"] = True
        metadata["ai_fallback_reason"] = "AI planning disabled by config; using deterministic local path."
        return AIPlanConversionResult(
            response=AILevelPlanResponse(provider="local", model="deterministic")
        ), metadata

    metadata["ai_planning_attempted"] = True
    context_chunks = _load_ai_context_chunks(config)
    metadata["ai_context_chunks_used"] = len(context_chunks)
    
    from gmdgen.errors import ProviderError, GmdgenError
    try:
        provider = create_ai_provider_from_config(config)
    except Exception as exc:
        if require_ai_planning:
            raise
        metadata["ai_planning_error"] = getattr(exc, "code", "provider_setup_failed")
        metadata["deterministic_fallback_used"] = True
        return AIPlanConversionResult(
            response=AILevelPlanResponse(provider="local", model="deterministic", fallback_used=True)
        ), metadata

    quality_mode = str(config.get("quality_mode", "")).strip().lower()
    candidate_count = max(1, int(config.get("ai_candidate_count", 3)))
    if quality_mode in {"extreme", "extreme ml", "extreme_ml"}:
        candidate_count = max(candidate_count, 5)
    
    best_conversion: AIPlanConversionResult | None = None
    best_report_score = -1.0
    candidate_reports: list[dict[str, Any]] = []
    selected_candidate_id = 0
    rounds_without_improvement = 0
    last_error_code = ""

    for candidate_id in range(1, candidate_count + 1):
        # Time budget check
        elapsed = time.time() - start_ai_loop_time
        if elapsed > config.get("max_extreme_ml_seconds", 300):
            metadata["stopped_reason"] = "time_budget_exceeded"
            break

        request = _build_ai_level_plan_request(
            config=config,
            features=features,
            section_plans=section_plans,
            time_x_report=time_x_report,
            style_profile=style_profile,
            context_chunks=context_chunks,
            object_budget=object_budget,
            safe_mode=safe_mode,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        
        try:
            response = provider.generate_level_plan(request)
            metadata["ai_calls_used"] += 1
        except Exception as exc:
            message = sanitize_ai_error(str(exc))
            last_error_code = getattr(exc, "code", "ollama_unknown_error")
            # If it's a known Ollama error class, use its code
            from gmdgen.ai.ollama_provider import OllamaProviderError
            if isinstance(exc, OllamaProviderError):
                last_error_code = exc.code
            
            candidate_reports.append({"candidate_id": candidate_id, "reject_reason": f"provider_error: {message}"})
            
            if require_ai_planning:
                raise ProviderError(message, code=last_error_code) from exc
            break

        # Convert AI response
        conversion = convert_ai_response_to_plans(
            response,
            object_budget=object_budget,
            max_group_id=max_group_id,
            safe_mode=safe_mode,
            section_plans=section_plans,
        )
        
        if conversion.valid:
            # Deterministic Expansion of AI-suggested plan.
            # Vary seed per candidate so deterministic materialization produces
            # distinct candidates even when the AI returns identical responses.
            if mat_config:
                from dataclasses import replace as _dc_replace
                candidate_mat_config = _dc_replace(mat_config, seed=mat_config.seed + candidate_id * 7919)
                expanded_objects = materialize_level_plans(
                    section_plans, # We use original sections for now
                    config=candidate_mat_config,
                    audio_features=features,
                    motif_library=motif_library,
                    style_profile=style_profile,
                    total_object_budget=object_budget,
                    speed_objects=speed_objects,
                    start_speed=start_speed_val,
                    song_offset=song_offset_val,
                )
                conversion.object_plans.extend(expanded_objects)

            score = _candidate_score_from_conversion(conversion, section_plans=section_plans, min_final_object_count=8)
            report = build_candidate_report(
                candidate_id=candidate_id,
                conversion_valid=conversion.valid,
                object_count=len(conversion.object_plans),
                trigger_count=len(conversion.trigger_plans),
                errors=conversion.errors,
                warnings=conversion.warnings,
                section_plans=section_plans,
                score=score,
            )
            candidate_reports.append(report.to_dict())

            if score > best_report_score + config.get("min_quality_improvement_delta", 0.03):
                best_conversion = conversion
                best_report_score = score
                selected_candidate_id = candidate_id
                rounds_without_improvement = 0
            else:
                rounds_without_improvement += 1
        else:
            candidate_reports.append({
                "candidate_id": candidate_id,
                "conversion_valid": False,
                "errors": conversion.errors,
                "score": 0.0
            })
            rounds_without_improvement += 1
            
        if rounds_without_improvement >= config.get("stop_if_no_improvement_rounds", 2):
            metadata["stopped_reason"] = "no_improvement"
            break

    if best_conversion is None:
        metadata["ai_planning_used"] = False
        metadata["deterministic_fallback_used"] = True
        metadata["ai_planning_error"] = last_error_code or "ai_output_invalid"
        
        fallback_response = AILevelPlanResponse(provider="local", model="deterministic", fallback_used=True)
        fallback_res = AIPlanConversionResult(response=fallback_response)
        fallback_res.warnings.append(f"Ollama AI plan was invalid JSON or failed: {last_error_code}")
        
        return fallback_res, metadata

    metadata.update({
        "ai_used": True,
        "ai_planning_used": True,
        "ai_response_valid": True,
        "selected_candidate_id": selected_candidate_id,
        "candidate_reports": candidate_reports,
        "ai_output_object_count": len(best_conversion.object_plans),
        "ai_output_trigger_count": len(best_conversion.trigger_plans),
    })
    
    return best_conversion, metadata


    metadata.update({
        "ai_used": True,
        "selected_candidate_id": selected_candidate_id,
        "candidate_reports": candidate_reports,
        "ai_output_object_count": len(best_conversion.object_plans),
        "ai_output_trigger_count": len(best_conversion.trigger_plans),
    })
    
    return best_conversion, metadata


def _provider_model_name(config: dict[str, Any], provider_name: str) -> str:
    if provider_name == "ollama":
        return str(config.get("ollama_model", "ollama-2.5-flash"))
    if provider_name == "ollama":
        return str(config.get("ollama_model", ""))
    return "local-heuristic" if provider_name == "local_test_only" else ""


def _build_ai_level_plan_request(
    *,
    config: dict[str, Any],
    features: AudioAnalysisResult,
    section_plans: list[SectionPlan],
    time_x_report: dict[str, Any],
    style_profile: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    object_budget: int,
    safe_mode: bool,
    start_speed: str,
    song_offset: float,
) -> AILevelPlanRequest:
    audio_summary, beat_summary, onset_summary = summarize_audio_for_model(features)
    reference_summaries = load_reference_levels(config.get("ollama_reference_levels_dir"))
    style_reference_summary = summarize_reference_style_for_model(style_profile)
    learned_style_summary = dict(style_profile.get("learned_style_summary", {})) if isinstance(style_profile.get("learned_style_summary", {}), dict) else {}
    user_prompt = str(config.get("prompt", "") or "").strip()
    if user_prompt:
        project_goal = (
            "Generate an editor-safe audio-conditioned Geometry Dash structured level plan "
            f"that follows this user prompt: {user_prompt}"
        )
    else:
        project_goal = "Generate an editor-safe audio-conditioned Geometry Dash structured level plan."
    if reference_summaries:
        style_reference_summary["reference_levels"] = reference_summaries[:8]
    return AILevelPlanRequest(
        project_goal=project_goal,
        generation_mode="audio_conditioned",
        difficulty=config.get("difficulty", "normal"),
        safe_mode=safe_mode,
        object_budget=object_budget,
        song_offset=song_offset,
        start_speed=start_speed,
        audio_summary=audio_summary,
        beat_summary=beat_summary,
        onset_summary=onset_summary,
        section_plans=summarize_sections_for_model(section_plans),
        time_x_summary={
            "average_error": time_x_report.get("average_error", 0.0),
            "max_error": time_x_report.get("max_error", 0.0),
            "invalid_count": time_x_report.get("invalid_count", 0),
            "checked_count": time_x_report.get("checked_count", 0),
        },
        trigger_schema_summary=summarize_trigger_schema_for_model(safe_mode=safe_mode),
        playability_rules_summary={
            "validator": "conservative rule-based spacing and trajectory envelope",
            "portal_recovery_required": True,
            "raw_save_string_allowed": False,
        },
        user_prompt=user_prompt,
        style_reference_summary=style_reference_summary,
        learned_style_summary=learned_style_summary,
        retrieved_motifs=list(_load_learned_prompt_context(config).get("retrieved_motifs", []))[:8],
        learned_object_distribution=dict(style_profile.get("learned_object_distribution", {})),
        learned_trigger_distribution=dict(style_profile.get("learned_trigger_distribution", {})),
        learned_density_profile=dict(style_profile.get("learned_density_profile", {})),
        learned_failure_patterns=list(_load_learned_prompt_context(config).get("learned_failure_patterns", []))[:8],
        learned_success_patterns=list(_load_learned_prompt_context(config).get("learned_success_patterns", []))[:8],
        retrieved_context=context_chunks,
        output_requirements={
            "format": "JSON only",
            "raw_save_string_allowed": False,
            "object_budget": object_budget,
            "safe_mode": safe_mode,
            "quality_feedback_from_previous_candidates": str(config.get("quality_feedback_prompt", "") or ""),
            "minimum_expectations": {
                "do_not_leave_drop_sections_empty": True,
                "use_valid_trigger_properties_only": True,
                "prefer_music_sync_and_section_contrast": True,
            },
        },
    )


def _load_ai_context_chunks(config: dict[str, Any]) -> list[dict[str, Any]]:
    if not bool(config.get("ollama_enable_retrieval", False)):
        documents = load_context_documents(config.get("ollama_context_dir"))
        chunks = summarize_context_documents(
            documents,
            int(config.get("ollama_max_context_chars", 6000)),
        )
        return _append_learning_context(chunks, config)
    documents = load_context_documents(config.get("ollama_context_dir"))
    retriever = LocalKeywordRetriever(documents)
    chunks = [chunk.to_dict() for chunk in retriever.retrieve("Geometry Dash trigger schema time-X playability", top_k=6)]
    return _append_learning_context(
        truncate_context_to_budget(chunks, int(config.get("ollama_max_context_chars", 6000))),
        config,
    )


def _append_learning_context(chunks: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    if (
        not bool(config.get("use_learning_memory", True))
        and not bool(config.get("use_learned_data", True))
        and not bool(config.get("use_dataset_context", True))
    ):
        return chunks
    enriched = list(chunks)
    if bool(config.get("use_learning_memory", True)):
        context = summarize_learning_examples_for_context(
            store_dir=config.get("learning_store_dir") or None,
            max_chars=int(config.get("ollama_max_context_chars", 6000)) // 3,
        )
        payload = context.to_dict()
        if payload.get("good_examples_summary") or payload.get("failure_patterns_summary"):
            enriched.append(
                {
                    "path": "learning_memory",
                    "title": "Learning memory summary",
                    "text": json.dumps(payload, ensure_ascii=False, sort_keys=True),
                }
            )
    learned_payload = _load_learned_prompt_context(config)
    if learned_payload:
        enriched.append(
            {
                "path": "learned_data",
                "title": "Learned style and motif memory",
                "text": json.dumps(learned_payload, ensure_ascii=False, sort_keys=True),
            }
        )
    if bool(config.get("use_dataset_context", True)):
        try:
            dataset_dir = resolve_dataset_dir(config.get("dataset_dir"))
            dataset_index = load_dataset_index(dataset_cache_path(dataset_dir))
            if dataset_index is None:
                dataset_index = build_dataset_index(
                    dataset_dir,
                    recursive=bool(config.get("recursive_dataset_scan", True)),
                    max_file_size_mb=float(config.get("max_file_size_mb", 8.0)),
                    max_total_context_chars=int(config.get("max_total_context_chars", config.get("ollama_max_context_chars", 6000))),
                )
            dataset_chunk = dataset_context_chunk(
                dataset_index,
                config,
                max_chars=max(600, int(config.get("ollama_max_context_chars", 6000)) // 4),
            )
            if dataset_chunk:
                enriched.append(dataset_chunk)
        except Exception as exc:  # noqa: BLE001
            enriched.append(
                {
                    "path": "dataset_context_warning",
                    "title": "Dataset context warning",
                    "text": f"Dataset context unavailable: {exc}",
                }
            )
    return truncate_context_to_budget(enriched, int(config.get("ollama_max_context_chars", 6000)))


def _load_learned_prompt_context(config: dict[str, Any]) -> dict[str, Any]:
    if not bool(config.get("use_learned_data", config.get("use_learning_memory", True))):
        return {}
    payload = summarize_learned_data_for_prompt(
        store_dir=config.get("learned_data_store_dir") or None,
        max_chars=int(config.get("ollama_max_context_chars", 6000)) // 3,
    )
    if not payload.get("learned_level_count") and not payload.get("motif_count"):
        return {}
    return payload


def _merge_learned_style_profile(style_profile: dict[str, Any], learned_payload: dict[str, Any]) -> dict[str, Any]:
    merged = dict(style_profile or {})
    learned_style = learned_payload.get("learned_style_summary", {})
    learned_objects = learned_payload.get("learned_object_distribution", {})
    learned_triggers = learned_payload.get("learned_trigger_distribution", {})
    merged["learned_style_summary"] = learned_style if isinstance(learned_style, dict) else {}
    merged["learned_object_distribution"] = learned_objects if isinstance(learned_objects, dict) else {}
    merged["learned_trigger_distribution"] = learned_triggers if isinstance(learned_triggers, dict) else {}
    merged["learned_density_profile"] = dict(learned_payload.get("learned_density_profile", {}))
    object_dist = dict(merged.get("object_id_distribution", {}))
    for object_id, count in merged["learned_object_distribution"].items():
        object_dist[str(object_id)] = int(object_dist.get(str(object_id), 0)) + int(count)
    merged["object_id_distribution"] = object_dist
    return merged


def _learned_motif_library(learned_payload: dict[str, Any]) -> list[dict[str, Any]]:
    motifs = learned_payload.get("retrieved_motifs", [])
    if not isinstance(motifs, list):
        return []
    library: list[dict[str, Any]] = []
    for motif in motifs[:16]:
        if not isinstance(motif, dict):
            continue
        object_ids = motif.get("object_ids", [])
        if not isinstance(object_ids, list):
            object_ids = []
        library.append(
            {
                "source": "learned_data",
                "section_type_hint": motif.get("section_type_hint", "normal"),
                "object_ids": [str(item) for item in object_ids[:24]],
                "density": float(motif.get("density", 0.0) or 0.0),
                "style_tags": list(motif.get("style_tags", []))[:8] if isinstance(motif.get("style_tags", []), list) else [],
            }
        )
    return library


def _learned_quality_metrics(
    level_objects: list[str],
    style_profile: dict[str, Any],
    actual_density: dict[str, float],
) -> dict[str, float]:
    learned_distribution = style_profile.get("learned_object_distribution", {})
    learned_triggers = style_profile.get("learned_trigger_distribution", {})
    learned_density = style_profile.get("learned_density_profile", {})
    if not learned_distribution and not learned_triggers and not learned_density:
        return {
            "learned_style_match_score": 0.0,
            "learned_motif_match_score": 0.0,
            "learned_density_match_score": 0.0,
            "learned_trigger_usage_score": 0.0,
        }
    object_ids: dict[str, int] = {}
    trigger_ids: dict[str, int] = {}
    for raw in level_objects:
        object_id = extract_object_id(raw)
        if not object_id:
            continue
        object_ids[str(object_id)] = object_ids.get(str(object_id), 0) + 1
        try:
            if classify(str(object_id)) == ObjectClass.TRIGGER:
                trigger_ids[str(object_id)] = trigger_ids.get(str(object_id), 0) + 1
        except Exception:
            pass
    style_score = _distribution_overlap(object_ids, learned_distribution if isinstance(learned_distribution, dict) else {})
    trigger_score = _distribution_overlap(trigger_ids, learned_triggers if isinstance(learned_triggers, dict) else {})
    learned_avg_density = float((learned_density if isinstance(learned_density, dict) else {}).get("average", 0.0) or 0.0)
    actual_avg_density = sum(actual_density.values()) / max(1, len(actual_density)) if actual_density else 0.0
    if learned_avg_density <= 0:
        density_score = 0.0
    else:
        density_score = max(0.0, 1.0 - min(1.0, abs(actual_avg_density - learned_avg_density) / max(learned_avg_density, 1e-6)))
    motif_score = max(style_score, min(1.0, (style_score + trigger_score + density_score) / 3.0))
    return {
        "learned_style_match_score": round(style_score, 4),
        "learned_motif_match_score": round(motif_score, 4),
        "learned_density_match_score": round(density_score, 4),
        "learned_trigger_usage_score": round(trigger_score, 4),
    }


def _distribution_overlap(actual: dict[str, int], learned: dict[str, Any]) -> float:
    if not actual or not learned:
        return 0.0
    actual_total = sum(max(0, int(value)) for value in actual.values()) or 1
    learned_int = {str(key): max(0, int(value)) for key, value in learned.items() if str(value).lstrip("-").isdigit()}
    learned_total = sum(learned_int.values()) or 1
    overlap = 0.0
    for key in set(actual) | set(learned_int):
        overlap += min(actual.get(key, 0) / actual_total, learned_int.get(key, 0) / learned_total)
    return round(max(0.0, min(1.0, overlap)), 4)


def _apply_ai_metadata(
    validation_report: ValidationReport,
    ai_metadata: dict[str, Any],
    ai_conversion: AIPlanConversionResult,
) -> None:
    validation_report.ai_provider = str(ai_metadata.get("ai_provider", "local"))
    validation_report.ai_model = str(ai_metadata.get("ai_model", ""))
    validation_report.ai_used = bool(ai_metadata.get("ai_used", False))
    validation_report.ai_fallback_used = bool(ai_metadata.get("ai_fallback_used", False))
    validation_report.ai_fallback_reason = str(ai_metadata.get("ai_fallback_reason", ""))
    validation_report.ai_context_chunks_used = int(ai_metadata.get("ai_context_chunks_used", 0))
    validation_report.ai_response_valid = bool(ai_metadata.get("ai_response_valid", False))
    validation_report.ai_output_object_count = int(ai_metadata.get("ai_output_object_count", 0))
    validation_report.ai_output_trigger_count = int(ai_metadata.get("ai_output_trigger_count", 0))
    validation_report.ai_debug_artifact_path = str(ai_metadata.get("ai_debug_artifact_path", ""))
    validation_report.ai_required = bool(ai_metadata.get("ai_required", True))
    validation_report.ai_error_type = str(ai_metadata.get("ai_error_type", ""))
    validation_report.ai_error_message_sanitized = str(ai_metadata.get("ai_error_message_sanitized", ""))
    validation_report.ai_retry_count = int(ai_metadata.get("ai_retry_count", 0))
    validation_report.ollama_sanitized_params = list(ai_metadata.get("ollama_sanitized_params", []))
    validation_report.ollama_removed_unsupported_params = list(ai_metadata.get("ollama_removed_unsupported_params", []))
    validation_report.ollama_param_retry_count = int(ai_metadata.get("ollama_param_retry_count", 0))
    validation_report.ollama_model_capabilities = dict(ai_metadata.get("ollama_model_capabilities", {}))
    validation_report.provider_chain = ai_metadata.get("provider_chain", {})
    validation_report.ai_calls_used = int(ai_metadata.get("ai_calls_used", 0))
    validation_report.ai_cache_hits = int(ai_metadata.get("ai_cache_hits", 0))
    validation_report.ai_request_hashes = [str(item) for item in ai_metadata.get("ai_request_hashes", [])]
    validation_report.local_fallback_used = bool(ai_metadata.get("local_fallback_used", False))
    validation_report.ai_normalization_warnings = list(ai_metadata.get("ai_normalization_warnings", []))
    validation_report.pruned_trigger_property_count = int(ai_metadata.get("pruned_trigger_property_count", 0))
    validation_report.ignored_irrelevant_trigger_property_count = int(ai_metadata.get("ignored_irrelevant_trigger_property_count", 0))
    validation_report.materialized_trigger_intent_count = int(ai_metadata.get("materialized_trigger_intent_count", 0))
    validation_report.auto_assigned_target_group_count = max(
        validation_report.auto_assigned_target_group_count,
        int(ai_metadata.get("auto_assigned_target_group_count", 0)),
    )
    validation_report.unresolved_missing_target_group_count = max(
        validation_report.unresolved_missing_target_group_count,
        int(ai_metadata.get("unresolved_missing_target_group_count", 0)),
    )
    validation_report.normalized_object_role_count = int(ai_metadata.get("normalized_object_role_count", 0))
    validation_report.normalized_easing_count = int(ai_metadata.get("normalized_easing_count", 0))
    validation_report.fatal_validation_issue_count = int(ai_metadata.get("fatal_validation_issue_count", 0))
    validation_report.nonfatal_validation_warning_count = int(ai_metadata.get("nonfatal_validation_warning_count", 0))
    validation_report.candidate_reports = list(ai_metadata.get("candidate_reports", []))
    validation_report.selected_candidate_id = int(ai_metadata.get("selected_candidate_id", 0))
    validation_report.section_candidate_reports = list(ai_metadata.get("section_candidate_reports", []))
    validation_report.selected_section_count = int(ai_metadata.get("selected_section_count", 0))
    validation_report.global_consistency_report = dict(ai_metadata.get("global_consistency_report", {}))
    validation_report.plan_snapshots.extend(list(ai_metadata.get("plan_snapshots", [])))
    validation_report.plan_diffs.extend(list(ai_metadata.get("plan_diffs", [])))
    validation_report.raw_ai_object_count = int(ai_metadata.get("raw_ai_object_count", 0))
    validation_report.raw_ai_trigger_count = int(ai_metadata.get("raw_ai_trigger_count", 0))
    validation_report.ollama_only_audit_passed = (
        validation_report.ai_required
        and validation_report.ai_provider in {"ollama", "ollama", "local_test_only"}
        and not (validation_report.ai_provider in {"ollama", "ollama"} and validation_report.local_fallback_used)
    )
    if validation_report.ai_provider == "local_test_only":
        validation_report.add_warning(
            "This output was generated without an external AI provider and is for testing only."
        )
    for error in ai_conversion.errors:
        validation_report.add_warning(f"ai_output_rejected: {error}")
    for warning in ai_conversion.warnings:
        validation_report.add_warning(warning)


def _candidate_score_from_conversion(
    conversion: AIPlanConversionResult,
    *,
    section_plans: list[SectionPlan],
    min_final_object_count: int,
) -> float:
    if not conversion.valid:
        return 0.0
    snapshot = snapshot_from_plans(
        "candidate_score",
        conversion.object_plans,
        conversion.trigger_plans,
        section_plans,
        warnings=conversion.warnings,
        fatal_errors=conversion.errors,
    )
    diversity = len(snapshot.object_id_distribution) / max(1, snapshot.object_count)
    drop_impact = compute_drop_impact_score(section_plans, snapshot.density_by_section)
    count_score = min(1.0, snapshot.object_count / max(1, min_final_object_count))
    trigger_score = min(1.0, snapshot.trigger_count / 6.0)
    warning_penalty = min(0.35, len(conversion.warnings) * 0.01)
    return max(0.0, 0.35 * count_score + 0.2 * trigger_score + 0.25 * drop_impact + 0.2 * diversity - warning_penalty)


def _finalize_quality_report(
    *,
    validation_report: ValidationReport,
    ai_metadata: dict[str, Any],
    pre_repair_snapshot: PlanSnapshot,
    repaired_plan_snapshot: PlanSnapshot,
    final_plan_snapshot: PlanSnapshot,
    repair_report: Any,
    section_plans: list[SectionPlan],
    level_objects: list[str],
) -> None:
    validation_report.plan_snapshots.extend(
        [
            pre_repair_snapshot.to_dict(),
            repaired_plan_snapshot.to_dict(),
            final_plan_snapshot.to_dict(),
        ]
    )
    all_snapshot_objects = [
        _snapshot_from_dict(snapshot)
        for snapshot in validation_report.plan_snapshots
        if isinstance(snapshot, dict)
    ]
    for previous, current in zip(all_snapshot_objects, all_snapshot_objects[1:]):
        validation_report.plan_diffs.append(diff_snapshots(previous, current).to_dict())
    validation_report.raw_ai_object_count = int(ai_metadata.get("raw_ai_object_count", ai_metadata.get("ai_output_object_count", 0)))
    validation_report.raw_ai_trigger_count = int(ai_metadata.get("raw_ai_trigger_count", ai_metadata.get("ai_output_trigger_count", 0)))
    validation_report.final_trigger_count = final_plan_snapshot.trigger_count
    original_objects = max(1, pre_repair_snapshot.object_count)
    original_triggers = max(1, pre_repair_snapshot.trigger_count)
    validation_report.removed_object_ratio = round(max(0, pre_repair_snapshot.object_count - final_plan_snapshot.object_count) / original_objects, 4)
    validation_report.removed_trigger_ratio = round(max(0, pre_repair_snapshot.trigger_count - final_plan_snapshot.trigger_count) / original_triggers, 4)
    density_target = compute_density_target_by_section(section_plans)
    actual_density = compute_actual_density_by_section_from_objects(level_objects, section_plans)
    density_error = compute_density_target_error(density_target, actual_density)
    drop_impact = compute_drop_impact_score(section_plans, actual_density)
    buildup_score = compute_buildup_progression_score(section_plans, actual_density)
    repair_quality = build_repair_quality_report(
        ai_normalization=getattr(ai_metadata, "normalization_report", None),
        plan_validation_warnings=validation_report.warnings,
        string_repair_report=repair_report,
        editor_safety_fatal_count=len(validation_report.issues),
    )
    repair_quality.pruned_irrelevant_trigger_properties = validation_report.pruned_trigger_property_count
    repair_quality.normalized_easing_count = validation_report.normalized_easing_count
    repair_quality.normalized_object_role_count = validation_report.normalized_object_role_count
    validation_report.repair_quality_report = repair_quality.to_dict()
    validation_report.quality_loss_reason_summary = summarize_quality_loss(
        repair_quality,
        raw_object_count=pre_repair_snapshot.object_count,
        final_object_count=final_plan_snapshot.object_count,
        raw_trigger_count=pre_repair_snapshot.trigger_count,
        final_trigger_count=final_plan_snapshot.trigger_count,
        drop_impact_score=drop_impact,
    )
    validation_report.density_target_by_section = density_target
    validation_report.actual_density_by_section = actual_density
    validation_report.density_target_error = density_error
    validation_report.drop_impact_score = drop_impact
    validation_report.buildup_progression_score = buildup_score
    validation_report.metrics["density_target_error"] = density_error
    validation_report.metrics["density_alignment_score"] = max(0.0, 1.0 - density_error)
    validation_report.metrics["drop_impact_score"] = drop_impact
    validation_report.metrics["buildup_progression_score"] = buildup_score
    validation_report.metrics["object_diversity_score"] = len(final_plan_snapshot.object_id_distribution) / max(1, final_plan_snapshot.object_count + final_plan_snapshot.trigger_count)

    validation_report.repair_loss_breakdown = {
        "x_monotone_fixed": getattr(repair_report, "x_monotone_fixed", 0),
        "group_id_remapped": getattr(repair_report, "group_id_remapped", 0),
        "orphan_trigger_removed": getattr(repair_report, "orphan_trigger_removed", 0),
        "unsafe_trigger_removed": getattr(repair_report, "unsafe_trigger_removed", 0),
        "trigger_schema_removed": getattr(repair_report, "trigger_schema_removed", 0),
        "playability_pruned": getattr(repair_report, "playability_pruned", 0),
        "density_spread": getattr(repair_report, "density_spread", 0),
        "budget_pruned": getattr(repair_report, "budget_pruned", 0),
    }
    
    playability_breakdown = dict(validation_report.playability_breakdown)
    playability_breakdown.update({
        "trajectory_warning_count": validation_report.metrics.get("trajectory_warning_count", 0),
        "playability_warning_count": len(validation_report.playability_warnings),
        "overcrowded_sections": len(validation_report.overcrowded_sections),
    })
    validation_report.playability_breakdown = playability_breakdown

    validation_report.plan_count_report.raw_ai_objects = validation_report.raw_ai_object_count
    validation_report.plan_count_report.raw_ai_triggers = validation_report.raw_ai_trigger_count
    validation_report.plan_count_report.repaired_objects = pre_repair_snapshot.object_count
    validation_report.plan_count_report.repaired_triggers = pre_repair_snapshot.trigger_count
    validation_report.plan_count_report.final_encoded_objects = final_plan_snapshot.object_count
    validation_report.plan_count_report.final_encoded_triggers = final_plan_snapshot.trigger_count
    if validation_report.selected_candidate_id:
        validation_report.plan_count_report.selected_candidate_objects = final_plan_snapshot.object_count
        validation_report.plan_count_report.selected_candidate_triggers = final_plan_snapshot.trigger_count


def _snapshot_from_dict(payload: dict[str, Any]) -> PlanSnapshot:
    return PlanSnapshot(
        stage=str(payload.get("stage", "")),
        section_count=int(payload.get("section_count", 0)),
        object_count=int(payload.get("object_count", 0)),
        trigger_count=int(payload.get("trigger_count", 0)),
        role_distribution=dict(payload.get("role_distribution", {})),
        object_id_distribution=dict(payload.get("object_id_distribution", {})),
        trigger_type_distribution=dict(payload.get("trigger_type_distribution", {})),
        density_by_section=dict(payload.get("density_by_section", {})),
        average_objects_per_second=float(payload.get("average_objects_per_second", 0.0)),
        average_triggers_per_section=float(payload.get("average_triggers_per_section", 0.0)),
        beat_aligned_event_count=int(payload.get("beat_aligned_event_count", 0)),
        onset_aligned_trigger_count=int(payload.get("onset_aligned_trigger_count", 0)),
        removed_object_count=int(payload.get("removed_object_count", 0)),
        removed_trigger_count=int(payload.get("removed_trigger_count", 0)),
        warnings=list(payload.get("warnings", [])),
        fatal_errors=list(payload.get("fatal_errors", [])),
    )


def _save_quality_debug_artifacts(
    *,
    config: dict[str, Any],
    output_dir: Path,
    output_name: str,
    validation_report: ValidationReport,
) -> str:
    debug_dir = Path(str(config.get("ollama_debug_dir", output_dir / "debug")))
    debug_dir.mkdir(parents=True, exist_ok=True)
    path = debug_dir / f"{output_name}.quality_report.json"
    payload = {
        "plan_snapshots": validation_report.plan_snapshots,
        "plan_diffs": validation_report.plan_diffs,
        "candidate_reports": validation_report.candidate_reports,
        "quality_loss_reason_summary": validation_report.quality_loss_reason_summary,
        "repair_quality_report": validation_report.repair_quality_report,
        "validation_report": validation_report.to_dict(),
    }
    path.write_text(json.dumps(_sanitize_debug_payload(payload), ensure_ascii=False, indent=2), encoding="utf-8")
    return str(path)


def _sanitize_debug_payload(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            str(key): _sanitize_debug_payload(item)
            for key, item in value.items()
            if "api_key" not in str(key).lower()
            and "base_url" not in str(key).lower()
            and "host" not in str(key).lower()
        }
    if isinstance(value, list):
        return [_sanitize_debug_payload(item) for item in value]
    if isinstance(value, str):
        if value.startswith("sk-"):
            return "sk-[REDACTED]"
        return value
    return value


def _optional_float(config: dict[str, Any], key: str, default: float | None) -> float | None:
    value = config.get(key, default)
    if value is None:
        return None
    return float(value)


def _difficulty_value(raw: Any) -> float:
    if isinstance(raw, (int, float)):
        return max(0.0, min(1.0, float(raw)))
    mapping = {
        "auto": 0.1,
        "easy": 0.2,
        "normal": 0.35,
        "hard": 0.55,
        "harder": 0.7,
        "insane": 0.82,
        "demon": 0.95,
    }
    return mapping.get(str(raw).strip().lower(), 0.5)


def _load_style_profile(path_value: Any) -> dict[str, Any]:
    if not path_value:
        return {}
    path = Path(str(path_value))
    if not path.exists():
        return {}

    document = parse_gmd_file(path)
    k4_entry = document.tags.get("k4")
    if not k4_entry:
        return {}
    decoded = decode_level_data(k4_entry[1])
    objects = split_level_objects(decoded)
    y_by_class: dict[str, list[float]] = {}
    ids_by_class: dict[str, list[str]] = {}
    object_id_distribution: dict[str, int] = {}
    for obj in objects:
        object_id = extract_object_id(obj)
        y = extract_object_number(obj, "3")
        if not object_id:
            continue
        object_id_distribution[object_id] = object_id_distribution.get(object_id, 0) + 1
        cls = classify(object_id).value
        ids_by_class.setdefault(cls, []).append(object_id)
        if y is not None:
            y_by_class.setdefault(cls, []).append(float(y))
    summary = summarize_reference_level(path)
    return {
        "level_header": extract_level_header(decoded),
        "ids_by_class": ids_by_class,
        "y_by_class": y_by_class,
        "object_id_distribution": object_id_distribution,
        **summary,
    }


def _load_motif_library(config: dict[str, Any]) -> list[dict[str, Any]]:
    artifact_path = Path(str(config.get("artifact_path", "artifacts/model.json")))
    if not artifact_path.exists():
        return []
    try:
        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return []
    assets = payload.get("generation_assets", {})
    if not isinstance(assets, dict):
        return []
    library = assets.get("chunk_library", [])
    if not isinstance(library, list):
        return []
    return [chunk for chunk in library if isinstance(chunk, dict)]


def _build_motif_bank_from_config(config: dict[str, Any]) -> MotifBank:
    paths: list[Path] = []
    reference_file = config.get("style_reference_level") or config.get("template_level")
    if reference_file:
        paths.append(Path(str(reference_file)))
    reference_dir = config.get("ollama_reference_levels_dir")
    if reference_dir:
        directory = Path(str(reference_dir))
        if directory.exists() and directory.is_dir():
            paths.extend(sorted(directory.glob("*.gmd")))
            paths.extend(sorted(directory.glob("*.txt")))
    return build_motif_bank_from_files(paths[:24])


def _plan_speed_portals(
    sections: list[SectionFeature],
    *,
    allow_speed_portals: bool,
    policy: str,
    start_speed: SpeedState,
    difficulty: float,
) -> list[tuple[float, SpeedState]]:
    if not allow_speed_portals or policy == "none":
        return []

    min_gap = {"conservative": 12.0, "musical": 8.0, "aggressive": 4.0}.get(policy, 12.0)
    max_count = {"conservative": 4, "musical": 8, "aggressive": 16}.get(policy, 4)
    candidates: list[tuple[float, SpeedState]] = []
    last_time = 0.0
    current_speed = start_speed

    for section in sections:
        if section.start_time <= 0.25 or section.start_time - last_time < min_gap:
            continue
        next_speed = current_speed
        if section.section_type == "drop":
            next_speed = SpeedState.FAST if difficulty < 0.75 else SpeedState.FASTER
        elif section.section_type == "buildup" and policy != "conservative":
            next_speed = SpeedState.FAST
        elif section.section_type in {"break", "outro"}:
            next_speed = SpeedState.NORMAL if difficulty >= 0.5 else SpeedState.SLOW
        elif policy == "aggressive" and section.section_type == "normal":
            next_speed = SpeedState.FAST

        if next_speed != current_speed:
            candidates.append((section.start_time, next_speed))
            current_speed = next_speed
            last_time = section.start_time
        if len(candidates) >= max_count:
            break
    return candidates


def _plan_sections(
    sections: list[SectionFeature],
    *,
    features: AudioAnalysisResult,
    speed_objects: list[SpeedObject],
    start_speed: SpeedState,
    song_offset: float,
    difficulty: float,
    sync_strength: float,
) -> list[SectionPlan]:
    plans: list[SectionPlan] = []
    confidence = float(
        features.confidence_report.overall if features.confidence_report else features.confidence
    )
    low_confidence_scale = 0.55 + 0.45 * max(0.0, min(1.0, confidence))
    max_energy = max((section.mean_energy for section in sections), default=1.0)
    max_density = max((section.rhythmic_density for section in sections), default=1.0)
    for idx, section in enumerate(sections):
        start_time = _snap_to_nearby_music_boundary(
            section.start_time,
            features=features,
            tolerance=0.12 + sync_strength * 0.1,
        )
        end_time = _snap_to_nearby_music_boundary(
            section.end_time,
            features=features,
            tolerance=0.12 + sync_strength * 0.1,
        )
        if idx == 0:
            start_time = section.start_time
        if end_time <= start_time:
            end_time = section.end_time
        base_density = _clamp((section.mean_energy / max(max_energy, 1e-9)) * (0.55 + difficulty))
        if section.section_type in {"intro", "break", "outro"}:
            density = base_density * 0.55
        elif section.section_type == "buildup":
            density = _clamp(base_density * (0.75 + idx / max(1, len(sections))))
        elif section.section_type == "drop":
            density = _clamp(base_density * 1.25)
        else:
            density = base_density
        trigger_intensity = _clamp(section.rhythmic_density / max(max_density, 1e-9))
        if section.section_type == "drop":
            trigger_intensity = _clamp(trigger_intensity * 1.35)
        if confidence < 0.45:
            density = _clamp(min(density, 0.65) * low_confidence_scale)
            trigger_intensity = _clamp(trigger_intensity * low_confidence_scale * 0.8)
        mode = _GAMEPLAY_MODES[min(len(_GAMEPLAY_MODES) - 1, int((idx + difficulty * 2) % len(_GAMEPLAY_MODES)))]
        plans.append(
            SectionPlan(
                start_time=start_time,
                end_time=end_time,
                start_x=pos_for_time_like_gd(start_time, speed_objects, start_speed, song_offset),
                end_x=pos_for_time_like_gd(end_time, speed_objects, start_speed, song_offset),
                section_type=section.section_type,
                gameplay_mode=mode,
                speed_state=speed_state_at_time(start_time, speed_objects, start_speed, song_offset),
                density_target=density,
                decoration_intensity=_clamp(
                    density
                    * (1.1 if section.section_type == "drop" else 0.85)
                    * (low_confidence_scale if confidence < 0.45 else 1.0)
                ),
                trigger_intensity=trigger_intensity,
                difficulty_target=_clamp(difficulty + density * 0.2),
            )
        )
    return plans


def _snap_to_nearby_music_boundary(
    time_value: float,
    *,
    features: AudioAnalysisResult,
    tolerance: float,
) -> float:
    candidates = list(features.downbeat_times or features.downbeat_candidates)
    candidates.extend(features.onset_times)
    if not candidates:
        return time_value
    nearest = min(candidates, key=lambda candidate: abs(candidate - time_value))
    return nearest if abs(nearest - time_value) <= tolerance else time_value


def _build_audio_events(features: AudioFeatures) -> list[AudioEvent]:
    section_lookup = features.sections or [
        SectionFeature(0, 0.0, features.duration, "high_energy", 0.0, 0.0, 0.5)
    ]
    events: list[AudioEvent] = []
    for beat in features.beat_features:
        section = _section_at_time(section_lookup, beat.time)
        events.append(
            AudioEvent(
                time=beat.time,
                beat_index=beat.index,
                is_downbeat=beat.is_downbeat,
                onset_strength=beat.strength,
                energy=beat.local_energy,
                section_id=section.section_id,
                section_type=section.section_type,
                confidence=max(0.4, min(1.0, beat.strength + beat.local_energy)),
            )
        )
    return events


def _plan_gameplay_events(
    beats: list[BeatFeature],
    *,
    section_plans: list[SectionPlan],
    speed_objects: list[SpeedObject],
    start_speed: SpeedState,
    song_offset: float,
    difficulty: float,
    sync_strength: float,
    max_events_per_beat: int,
) -> list[GameplayEvent]:
    events: list[GameplayEvent] = []
    max_strength = max((beat.strength for beat in beats), default=1.0)
    base_stride = max(1, int(round(3 - min(2, difficulty * 2))))

    for beat in beats:
        section = _section_plan_at_time(section_plans, beat.time)
        if section.section_type in {"intro", "break", "outro"}:
            stride = base_stride + 1
        elif section.section_type == "drop":
            stride = max(1, base_stride - 1)
        else:
            stride = base_stride
        beat_importance = max(
            beat.strength / max(max_strength, 1e-9),
            0.35 if beat.is_downbeat else 0.0,
            section.density_target,
        )
        if beat.index % stride != 0 and beat_importance < sync_strength:
            continue

        x_pos = pos_for_time_like_gd(beat.time, speed_objects, start_speed, song_offset)
        sync_error = sync_error_for_x(
            x=x_pos,
            expected_audio_time=beat.time,
            speed_objects=speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        section_id = section_plans.index(section) if section in section_plans else 0
        event_type = "structure"
        if beat.is_downbeat and section.section_type in {"drop", "buildup"}:
            event_type = "transition"
        elif beat.strength >= max_strength * (0.45 + (1.0 - sync_strength) * 0.2) and difficulty >= 0.25:
            event_type = "orb" if difficulty >= 0.45 else "pad"
        events.append(
            GameplayEvent(
                time=beat.time,
                x=x_pos,
                event_type=event_type,
                importance=beat_importance,
                beat_index=beat.index,
                sync_error=sync_error,
                section_id=section_id,
            )
        )
        if max_events_per_beat <= 1 or event_type in {"orb", "pad"}:
            continue
        if section.section_type == "drop" and beat_importance >= 0.65:
            events.append(
                GameplayEvent(
                    time=beat.time,
                    x=x_pos + 36,
                    event_type="structure_accent",
                    importance=beat_importance * 0.8,
                    beat_index=beat.index,
                    sync_error=sync_error,
                    section_id=section_id,
                )
            )
    return events


def _objects_from_gameplay_events(
    gameplay_events: list[GameplayEvent],
    *,
    style_profile: dict[str, Any],
    difficulty: float,
    object_budget: int,
) -> list[ObjectPlan]:
    object_plans: list[ObjectPlan] = []
    y_structure = _style_y(style_profile, ObjectClass.STRUCTURE.value, default=90.0)
    y_orb = _style_y(style_profile, ObjectClass.SPECIAL.value, default=180.0)
    for event in sorted(gameplay_events, key=lambda item: (item.x, -item.importance)):
        if len(object_plans) >= object_budget:
            break
        if event.event_type == "orb":
            object_id = _ORB_IDS[event.beat_index % len(_ORB_IDS)]
            y = y_orb + (event.beat_index % 3) * 18
            role = "beat_orb"
            x = event.x + 24
        elif event.event_type == "pad":
            object_id = _PAD_IDS[event.beat_index % len(_PAD_IDS)]
            y = y_structure + 42
            role = "beat_pad"
            x = event.x + 18
        elif event.event_type == "transition":
            object_id = "1"
            y = y_structure + (18 if difficulty >= 0.5 else 0)
            role = "section_transition_structure"
            x = event.x
        else:
            object_id = "1"
            y = y_structure
            role = event.event_type
            x = event.x
        object_plans.append(
            ObjectPlan(
                object_id=object_id,
                x=x,
                y=y,
                role=role,
                editor_layer=1,
                beat_aligned_time=event.time,
                sync_error=event.sync_error,
                safety_flags={"importance": round(event.importance, 4), "section_id": event.section_id},
            )
        )
    return object_plans


def _plan_style_motif_objects(
    *,
    section_plans: list[SectionPlan],
    motif_library: list[dict[str, Any]],
    style_profile: dict[str, Any],
    object_budget: int,
    seed: int,
    safe_mode: bool,
) -> list[ObjectPlan]:
    if object_budget <= 0:
        return []
    import random

    rng = random.Random(seed + 17041)
    result: list[ObjectPlan] = []
    visible_chunks = [
        chunk for chunk in motif_library
        if isinstance(chunk.get("objects"), list) and chunk.get("objects")
    ]
    if not visible_chunks:
        return _fallback_style_motif_objects(
            section_plans=section_plans,
            style_profile=style_profile,
            object_budget=object_budget,
        )

    for section_index, section in enumerate(section_plans):
        if len(result) >= object_budget:
            break
        section_width = max(1.0, section.end_x - section.start_x)
        if section.section_type in {"intro", "break", "outro"}:
            target_count = max(0, int(section_width / 900 * section.decoration_intensity))
        else:
            target_count = max(1, int(section_width / 650 * section.decoration_intensity))
        target_count = min(target_count, object_budget - len(result), 24)
        if target_count <= 0:
            continue
        chunk = rng.choice(visible_chunks)
        candidates = []
        for raw_obj in chunk.get("objects", []):
            object_id = extract_object_id(str(raw_obj))
            if not object_id:
                continue
            obj_class = classify(object_id)
            if obj_class not in {ObjectClass.STRUCTURE, ObjectClass.DECORATION}:
                continue
            if safe_mode and obj_class == ObjectClass.TRIGGER:
                continue
            x_val = extract_object_number(str(raw_obj), "2")
            y_val = extract_object_number(str(raw_obj), "3")
            if x_val is None or y_val is None:
                continue
            candidates.append((object_id, float(x_val), float(y_val), obj_class.value))
        if not candidates:
            continue
        xs = [item[1] for item in candidates]
        x_min = min(xs)
        x_span = max(1.0, max(xs) - x_min)
        for object_id, x_val, y_val, cls_name in candidates[:target_count]:
            warped_x = section.start_x + ((x_val - x_min) / x_span) * section_width
            result.append(
                ObjectPlan(
                    object_id=object_id,
                    x=warped_x,
                    y=y_val,
                    role=f"style_motif_{cls_name}",
                    editor_layer=2,
                    safety_flags={"section_id": section_index, "source": "markov_chunk_motif"},
                )
            )
            if len(result) >= object_budget:
                break
    return _ensure_num_sections_for_report(result)
def _fallback_style_motif_objects(
    *,
    section_plans: list[SectionPlan],
    style_profile: dict[str, Any],
    object_budget: int,
) -> list[ObjectPlan]:
    result: list[ObjectPlan] = []
    decoration_ids = style_profile.get("ids_by_class", {}).get(ObjectClass.DECORATION.value, ["500"])
    for section_index, section in enumerate(section_plans):
        if len(result) >= object_budget:
            break
        if section.decoration_intensity < 0.35:
            continue
        object_id = str(decoration_ids[section_index % len(decoration_ids)])
        result.append(
            ObjectPlan(
                object_id=object_id,
                x=section.start_x + max(60.0, (section.end_x - section.start_x) * 0.5),
                y=240,
                role="fallback_style_motif_decoration",
                editor_layer=2,
                safety_flags={"section_id": section_index, "source": "style_profile"},
            )
        )
    return _ensure_num_sections_for_report(result)
def _speed_portal_object_plans(
    speed_objects: list[SpeedObject],
    *,
    editor_safe_mode: bool,
) -> list[ObjectPlan]:
    if editor_safe_mode and len(speed_objects) > 12:
        speed_objects = speed_objects[:12]
    return [
        ObjectPlan(
            object_id=speed_object.object_id or "201",
            x=speed_object.x,
            y=150,
            role="speed_portal",
            editor_layer=1,
            beat_aligned_time=speed_object.time,
        )
        for speed_object in speed_objects
    ]


def _plan_trigger_events(
    features: AudioFeatures,
    *,
    speed_objects: list[SpeedObject],
    start_speed: SpeedState,
    song_offset: float,
    allow_triggers: bool,
    threshold: float,
    sync_strength: float,
    object_budget: int,
    allocator: GroupAllocator,
    safe_mode: bool,
) -> tuple[list[ObjectPlan], list[TriggerPlan]]:
    if not allow_triggers:
        return [], []

    onset_values = [value for _, value in features.onset_envelope]
    max_onset = max(onset_values, default=1.0)
    cutoff = max_onset * max(0.05, min(1.0, threshold))
    max_triggers = max(0, object_budget // (10 if safe_mode else 6))
    trigger_objects: list[ObjectPlan] = []
    trigger_plans: list[TriggerPlan] = []

    for time_value, onset_value in features.onset_envelope:
        if len(trigger_plans) >= max_triggers:
            break
        if onset_value < cutoff:
            continue
        nearest_beat = _nearest(features.beat_times, time_value)
        if abs(time_value - nearest_beat) > max(0.25, 0.1 + (1.0 - sync_strength) * 0.3):
            continue
        x_pos = pos_for_time_like_gd(time_value, speed_objects, start_speed, song_offset)
        sync_error = sync_error_for_x(
            x=x_pos,
            expected_audio_time=time_value,
            speed_objects=speed_objects,
            start_speed=start_speed,
            song_offset=song_offset,
        )
        target_group = allocator.allocate()
        trigger_objects.append(
            ObjectPlan(
                object_id="500",
                x=x_pos,
                y=240,
                role="visual_accent_target",
                group_ids=[target_group],
                editor_layer=2,
                beat_aligned_time=time_value,
                sync_error=sync_error,
            )
        )
        trigger_type = "pulse" if onset_value >= max_onset * 0.75 else "alpha"
        trigger_id = "1006" if trigger_type == "pulse" else "1007"
        trigger_plans.append(
            TriggerPlan(
                trigger_type=trigger_type,
                object_id=trigger_id,
                x=x_pos,
                y=300,
                target_group=target_group,
                duration=min(0.35, max(0.05, 0.16 + onset_value)),
                multi_trigger=False,
                editor_disable=False,
                beat_aligned_time=time_value,
                sync_error=sync_error,
            )
        )
    return trigger_objects, trigger_plans


def _repair_and_validate_plans(
    *,
    object_plans: list[ObjectPlan],
    trigger_plans: list[TriggerPlan],
    section_plans: list[SectionPlan] | None = None,
    speed_objects: list[SpeedObject],
    features: AudioFeatures,
    start_speed: SpeedState,
    song_offset: float,
    object_budget: int,
    beat_snap_tolerance: float,
    max_events_per_beat: int,
    max_group_id: int,
    safe_mode: bool,
    editor_safe_mode: bool,
) -> ValidationReport:
    report = ValidationReport()
    total_plans = len(object_plans) + len(trigger_plans)
    if total_plans > object_budget:
        overflow = total_plans - object_budget
        del object_plans[max(0, len(object_plans) - overflow) :]
        report.add_warning(f"object_budget_pruned: removed {overflow} low priority plans")

    group_report = allocate_trigger_target_groups(
        object_plans,
        trigger_plans,
        section_plans=section_plans or [],
        max_group_id=max_group_id,
    )
    report.metrics["auto_assigned_target_group_count"] = group_report.auto_assigned_target_group_count
    report.metrics["repaired_orphan_trigger_count"] = group_report.repaired_orphan_trigger_count
    report.metrics["unresolved_missing_target_group_count"] = group_report.unresolved_missing_target_group_count
    report.auto_assigned_target_group_count = group_report.auto_assigned_target_group_count
    report.unresolved_missing_target_group_count = group_report.unresolved_missing_target_group_count
    for warning in group_report.warnings:
        report.add_warning(warning)

    defined_groups = {
        group_id for plan in object_plans for group_id in plan.group_ids if group_id > 0
    }
    if any(group_id > max_group_id for group_id in defined_groups):
        report.add_issue("group_id_out_of_range")

    valid_triggers: list[TriggerPlan] = []
    trigger_mode = TriggerMode.SAFE if safe_mode else TriggerMode.ADVANCED
    for trigger in trigger_plans:
        defaulted = apply_trigger_defaults(trigger, trigger_mode)
        if defaulted is None:
            report.add_warning(f"removed_unsupported_trigger: {trigger.trigger_type}")
            continue
        trigger = defaulted
        schema_issues = validate_trigger_plan_schema(
            trigger,
            trigger_mode,
            max_group_id=max_group_id,
        )
        if schema_issues:
            for issue in schema_issues:
                report.add_warning(issue)
            continue
        if trigger.target_group is not None and trigger.target_group not in defined_groups:
            report.add_warning(
                f"removed_orphan_trigger: {trigger.trigger_type} target={trigger.target_group}"
            )
            continue
        if safe_mode and trigger.duration > 1.0:
            trigger.duration = 1.0
        if trigger.spawn_delay < 0:
            report.add_warning("negative_spawn_delay_clamped")
            trigger.spawn_delay = 0.0
        valid_triggers.append(trigger)
    trigger_plans[:] = valid_triggers

    beat_buckets: dict[int, int] = {}
    for plan in object_plans:
        if plan.beat_aligned_time is None:
            continue
        nearest_idx = _nearest_index(features.beat_times, plan.beat_aligned_time)
        beat_buckets[nearest_idx] = beat_buckets.get(nearest_idx, 0) + 1
    crowded = [idx for idx, count in beat_buckets.items() if count > max_events_per_beat + 2]
    if crowded:
        report.add_warning(f"crowded_beats: {len(crowded)} beats exceed event target")

    sync_errors = []
    for plan in object_plans:
        if plan.beat_aligned_time is None:
            continue
        actual_time = time_for_pos_like_gd(plan.x, speed_objects, start_speed, song_offset)
        plan.sync_error = actual_time - plan.beat_aligned_time
        sync_errors.append(abs(plan.sync_error))

    max_sync_error = max(sync_errors, default=0.0)
    report.metrics["max_sync_error"] = round(max_sync_error, 6)
    report.metrics["mean_sync_error"] = round(sum(sync_errors) / len(sync_errors), 6) if sync_errors else 0.0
    report.metrics["speed_object_count"] = len(speed_objects)
    report.metrics["object_plan_count"] = len(object_plans)
    report.metrics["trigger_plan_count"] = len(trigger_plans)
    if max_sync_error > beat_snap_tolerance * 2:
        report.add_warning(
            f"sync_error_high: max={round(max_sync_error, 4)} tolerance={beat_snap_tolerance}"
        )

    if editor_safe_mode and len(speed_objects) != len(sorted(speed_objects, key=lambda obj: obj.x)):
        report.add_issue("speed_objects_not_sorted")
    return report


def _compose_level_data(header: str, objects: list[str]) -> str:
    chunks = [header] if header else []
    chunks.extend(objects)
    return ";".join(chunks) + ";"


def _style_y(style_profile: dict[str, Any], class_name: str, *, default: float) -> float:
    values = style_profile.get("y_by_class", {}).get(class_name, [])
    if not values:
        return default
    return float(sum(values[:128]) / len(values[:128]))


def _section_plan_at_time(section_plans: list[SectionPlan], time_value: float) -> SectionPlan:
    for section in section_plans:
        if section.start_time <= time_value < section.end_time:
            return section
    if section_plans:
        return section_plans[-1]
    return SectionPlan(
        start_time=0.0,
        end_time=max(0.0, time_value),
        start_x=0.0,
        end_x=0.0,
        section_type="normal",
        gameplay_mode="cube",
        speed_state=SpeedState.NORMAL,
        density_target=0.5,
        decoration_intensity=0.5,
        trigger_intensity=0.5,
        difficulty_target=0.5,
    )


def _playability_warnings(
    object_plans: list[ObjectPlan],
    *,
    section_plans: list[SectionPlan],
    difficulty: float,
) -> list[str]:
    import bisect
    gameplay = [
        plan for plan in object_plans
        if plan.role in {
            "beat_orb",
            "beat_pad",
            "structure",
            "structure_accent",
            "section_transition_structure",
            "ground_or_structure",
        }
        or "structure" in plan.role
    ]
    gameplay.sort(key=lambda plan: plan.x)
    warnings: list[str] = []
    
    # Sort section plans by x for efficient lookup
    sorted_sections = sorted(section_plans, key=lambda s: s.start_x)
    section_starts = [s.start_x for s in sorted_sections]
    
    for prev, current in zip(gameplay, gameplay[1:]):
        section_idx = bisect.bisect_right(section_starts, current.x) - 1
        section = sorted_sections[max(0, section_idx)]
        
        min_gap = _min_spacing_for_mode(section.gameplay_mode, section.speed_state, difficulty)
        if current.x - prev.x < min_gap:
            warnings.append(
                f"tight_spacing: {round(current.x - prev.x, 2)}px < {round(min_gap, 2)}px near x={round(current.x, 2)}"
            )
            if len(warnings) >= 32:
                break

    portal_xs = sorted([
        plan.x for plan in object_plans
        if plan.role == "speed_portal" or plan.object_id in {"12", "13", "47", "111", "660", "745", "1331", "200", "201", "202", "203", "1334"}
    ])
    
    gameplay_xs = [p.x for p in gameplay]
    
    for portal_x in portal_xs:
        idx = bisect.bisect_right(gameplay_xs, portal_x)
        if idx < len(gameplay_xs):
            next_gameplay_x = gameplay_xs[idx]
            if next_gameplay_x < portal_x + 90:
                warnings.append(f"portal_safety_margin: gameplay object too close after portal x={round(portal_x, 2)}")
                if len(warnings) >= 32:
                    break
    return warnings


def _section_plan_for_x(section_plans: list[SectionPlan], x_value: float) -> SectionPlan:
    for section in section_plans:
        if section.start_x <= x_value < section.end_x:
            return section
    return section_plans[-1] if section_plans else _section_plan_at_time([], 0.0)


def _min_spacing_for_mode(mode: str, speed_state: SpeedState, difficulty: float) -> float:
    mode_base = {
        "cube": 42.0,
        "ship": 64.0,
        "ball": 54.0,
        "ufo": 58.0,
        "wave": 72.0,
        "robot": 58.0,
        "spider": 52.0,
    }.get(mode, 52.0)
    speed_mult = {
        SpeedState.SLOW: 0.8,
        SpeedState.NORMAL: 1.0,
        SpeedState.FAST: 1.18,
        SpeedState.FASTER: 1.35,
        SpeedState.FASTEST: 1.55,
    }[speed_state]
    difficulty_relief = 1.0 - min(0.45, difficulty * 0.35)
    return mode_base * speed_mult * difficulty_relief


def _average_alignment_error(
    event_times: list[float | None],
    target_times: list[float],
) -> float:
    import bisect
    clean_events = [float(time) for time in event_times if time is not None]
    if not clean_events or not target_times:
        return 0.0
        
    sorted_targets = sorted(target_times)
    errors = []
    
    for event_time in clean_events:
        idx = bisect.bisect_left(sorted_targets, event_time)
        if idx == 0:
            nearest = sorted_targets[0]
        elif idx == len(sorted_targets):
            nearest = sorted_targets[-1]
        else:
            prev = sorted_targets[idx - 1]
            curr = sorted_targets[idx]
            nearest = prev if event_time - prev < curr - event_time else curr
            
        errors.append(abs(event_time - nearest))
        
    return sum(errors) / len(errors)


def _section_at_time(sections: list[SectionFeature], time_value: float) -> SectionFeature:
    for section in sections:
        if section.start_time <= time_value < section.end_time:
            return section
    return sections[-1]


def _nearest(values: list[float], target: float) -> float:
    if not values:
        return target
    import bisect
    idx = bisect.bisect_left(values, target)
    if idx == 0:
        return values[0]
    if idx == len(values):
        return values[-1]
    prev = values[idx - 1]
    curr = values[idx]
    return prev if target - prev < curr - target else curr


def _nearest_index(values: list[float], target: float) -> int:
    if not values:
        return 0
    import bisect
    idx = bisect.bisect_left(values, target)
    if idx == 0:
        return 0
    if idx == len(values):
        return len(values) - 1
    prev = values[idx - 1]
    curr = values[idx]
    return idx - 1 if target - prev < curr - target else idx


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _section_plan_to_dict(plan: SectionPlan) -> dict[str, Any]:
    payload = asdict(plan)
    payload["speed_state"] = plan.speed_state.value
    return payload


def _ensure_num_sections_for_report(result_obj):
    if isinstance(result_obj, dict) and "num_sections" not in result_obj:
        candidates = (
            result_obj.get("sections")
            or result_obj.get("section_reports")
            or result_obj.get("audio_sections")
            or []
        )
        try:
            count = len(candidates)
        except TypeError:
            count = 0

        if result_obj.get("generation_mode") == "audio_conditioned" or result_obj.get("audio_backend"):
            count = max(1, count)

        result_obj["num_sections"] = count

    return result_obj

