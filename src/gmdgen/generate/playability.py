# SPDX-License-Identifier: GPL-3.0-or-later
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from gmdgen.gd.plans import GameplayEvent, ObjectPlan, PlayabilityWarning, SectionPlan
from gmdgen.gd.time_mapping import SpeedState, normalize_speed_state


@dataclass(slots=True, frozen=True)
class ModePhysicsProfile:
    mode: str
    speed_state: SpeedState
    min_input_interval: float
    min_obstacle_spacing: float
    portal_recovery_time: float
    safe_y_margin: float
    max_reasonable_density: float
    tight_timing_allowed_by_difficulty: float
    notes: str = "Conservative envelope profile, not an exact GD physics engine."


@dataclass(slots=True)
class TrajectoryEnvelope:
    mode: str
    speed_state: SpeedState
    start_time: float
    end_time: float
    start_x: float
    end_x: float
    min_y: float
    max_y: float
    safe_margin: float
    confidence: float
    warnings: list[PlayabilityWarning] = field(default_factory=list)


@dataclass(slots=True)
class PlayabilityRepairReport:
    spacing_score_before: float = 1.0
    spacing_score_after: float = 1.0
    portal_safety_score: float = 1.0
    mode_transition_score: float = 1.0
    orb_pad_timing_score: float = 1.0
    hazard_distance_score: float = 1.0
    input_density_score_before: float = 1.0
    input_density_score_after: float = 1.0
    speed_safety_score: float = 1.0
    section_difficulty_curve_score: float = 1.0
    converted_gameplay_to_decoration: int = 0
    removed_hazards_after_portal: int = 0
    simplified_dense_orb_chain: int = 0
    recovery_margin_adjusted: int = 0
    warnings: list[str] = field(default_factory=list)

    @property
    def score_before(self) -> float:
        return min(self.spacing_score_before, self.input_density_score_before)

    @property
    def score_after(self) -> float:
        return min(
            self.spacing_score_after,
            self.portal_safety_score,
            self.mode_transition_score,
            self.orb_pad_timing_score,
            self.hazard_distance_score,
            self.input_density_score_after,
            self.speed_safety_score,
            self.section_difficulty_curve_score,
        )

    def to_dict(self) -> dict[str, float | int | list[str]]:
        return {
            "spacing_score": round(self.spacing_score_after, 4),
            "spacing_score_before": round(self.spacing_score_before, 4),
            "portal_safety_score": round(self.portal_safety_score, 4),
            "mode_transition_score": round(self.mode_transition_score, 4),
            "orb_pad_timing_score": round(self.orb_pad_timing_score, 4),
            "hazard_distance_score": round(self.hazard_distance_score, 4),
            "input_density_score": round(self.input_density_score_after, 4),
            "input_density_score_before": round(self.input_density_score_before, 4),
            "speed_safety_score": round(self.speed_safety_score, 4),
            "section_difficulty_curve_score": round(self.section_difficulty_curve_score, 4),
            "converted_gameplay_to_decoration": self.converted_gameplay_to_decoration,
            "removed_hazards_after_portal": self.removed_hazards_after_portal,
            "simplified_dense_orb_chain": self.simplified_dense_orb_chain,
            "recovery_margin_adjusted": self.recovery_margin_adjusted,
            "score_before": round(self.score_before, 4),
            "score_after": round(self.score_after, 4),
            "warnings": list(self.warnings),
        }


_BASE_PROFILES: dict[str, tuple[float, float, float, float, float]] = {
    # min_input_interval, min_obstacle_spacing, recovery_time, safe_y_margin, density
    "cube": (0.18, 54.0, 0.24, 24.0, 0.025),
    "ship": (0.12, 72.0, 0.30, 54.0, 0.018),
    "ball": (0.20, 62.0, 0.28, 32.0, 0.022),
    "ufo": (0.22, 68.0, 0.30, 36.0, 0.020),
    "wave": (0.10, 46.0, 0.22, 42.0, 0.030),
    "robot": (0.22, 68.0, 0.28, 30.0, 0.020),
    "spider": (0.18, 58.0, 0.24, 28.0, 0.024),
}

_SPEED_MULT = {
    SpeedState.SLOW: 0.82,
    SpeedState.NORMAL: 1.0,
    SpeedState.FAST: 1.18,
    SpeedState.FASTER: 1.35,
    SpeedState.FASTEST: 1.55,
}


def build_mode_physics_profile(
    mode: str,
    speed_state: SpeedState | str,
    difficulty: str | float | int,
) -> ModePhysicsProfile:
    normalized_mode = str(mode).lower()
    speed = normalize_speed_state(speed_state)
    base = _BASE_PROFILES.get(normalized_mode, _BASE_PROFILES["cube"])
    relief = _difficulty_relief(difficulty)
    speed_mult = _SPEED_MULT[speed]
    return ModePhysicsProfile(
        mode=normalized_mode,
        speed_state=speed,
        min_input_interval=base[0] * relief,
        min_obstacle_spacing=base[1] * speed_mult * relief,
        portal_recovery_time=base[2] * speed_mult * relief,
        safe_y_margin=base[3],
        max_reasonable_density=base[4] / max(0.5, relief),
        tight_timing_allowed_by_difficulty=1.0 - relief,
    )


def estimate_trajectory_envelope(
    section_plan: SectionPlan,
    gameplay_events: Iterable[GameplayEvent] | None = None,
    object_plans: Iterable[ObjectPlan] | None = None,
    *,
    difficulty: str | float | int = "normal",
) -> TrajectoryEnvelope:
    profile = build_mode_physics_profile(
        section_plan.gameplay_mode,
        section_plan.speed_state,
        difficulty,
    )
    ys = [plan.y for plan in object_plans or [] if section_plan.start_x <= plan.x <= section_plan.end_x]
    if not ys:
        ys = [90.0, 180.0]
    envelope = TrajectoryEnvelope(
        mode=profile.mode,
        speed_state=profile.speed_state,
        start_time=section_plan.start_time,
        end_time=section_plan.end_time,
        start_x=section_plan.start_x,
        end_x=section_plan.end_x,
        min_y=max(0.0, min(ys)),
        max_y=max(ys),
        safe_margin=profile.safe_y_margin,
        confidence=0.55,
    )
    envelope.warnings.extend(
        validate_trajectory_envelope(
            envelope,
            hazards=list(_role_filter(object_plans or [], {"hazard"})),
            portals=list(_portal_filter(object_plans or [])),
            orbs=list(_role_filter(object_plans or [], {"beat_orb"})),
            pads=list(_role_filter(object_plans or [], {"beat_pad"})),
            difficulty=difficulty,
        )
    )
    return envelope


def validate_trajectory_envelope(
    envelope: TrajectoryEnvelope,
    *,
    hazards: list[ObjectPlan],
    portals: list[ObjectPlan],
    orbs: list[ObjectPlan],
    pads: list[ObjectPlan],
    difficulty: str | float | int = "normal",
) -> list[PlayabilityWarning]:
    warnings: list[PlayabilityWarning] = []
    profile = build_mode_physics_profile(envelope.mode, envelope.speed_state, difficulty)
    corridor_height = envelope.max_y - envelope.min_y
    if envelope.mode == "ship" and corridor_height < profile.safe_y_margin * 2.0:
        warnings.append(_warning("ship_corridor_too_narrow", envelope, f"ship corridor height {corridor_height:.2f}px is narrow"))
    if envelope.mode == "wave" and corridor_height < profile.safe_y_margin * 1.7:
        warnings.append(_warning("wave_corridor_too_narrow", envelope, f"wave corridor height {corridor_height:.2f}px is narrow"))
    for hazard in hazards:
        if envelope.start_x <= hazard.x <= envelope.end_x:
            if min(abs(hazard.y - envelope.min_y), abs(hazard.y - envelope.max_y)) < envelope.safe_margin:
                warnings.append(_warning("hazard_margin", envelope, f"hazard near trajectory envelope at x={hazard.x:.2f}", x=hazard.x))
    for portal in portals:
        recovery_x = portal.x + profile.portal_recovery_time * 311.58
        if any(portal.x < hazard.x < recovery_x for hazard in hazards):
            warnings.append(_warning("portal_transition_recovery", envelope, f"hazard before portal recovery margin x={portal.x:.2f}", x=portal.x))
        if any(portal.x < event.x < recovery_x for event in [*orbs, *pads]):
            warnings.append(_warning("portal_immediate_input", envelope, f"input event before portal recovery margin x={portal.x:.2f}", x=portal.x))
    inputs = sorted([*orbs, *pads], key=lambda plan: plan.x)
    for prev, current in zip(inputs, inputs[1:]):
        if current.x - prev.x < profile.min_obstacle_spacing * 0.65:
            warnings.append(_warning("trajectory_input_density", envelope, "input events are too dense", x=current.x))
            break
    density = len(inputs) / max(1.0, envelope.end_x - envelope.start_x)
    if density > profile.max_reasonable_density:
        warnings.append(_warning("trajectory_density", envelope, f"input density {density:.4f} exceeds profile {profile.max_reasonable_density:.4f}"))
    return warnings


def validate_mode_transition_trajectory(
    section_plans: list[SectionPlan],
    gameplay_events: list[GameplayEvent] | None = None,
) -> list[PlayabilityWarning]:
    warnings: list[PlayabilityWarning] = []
    for prev, current in zip(section_plans, section_plans[1:]):
        if prev.gameplay_mode != current.gameplay_mode:
            gap = current.start_x - prev.end_x
            profile = build_mode_physics_profile(current.gameplay_mode, current.speed_state, current.difficulty_target)
            if gap < profile.min_obstacle_spacing * 0.5:
                warnings.append(
                    PlayabilityWarning(
                        warning_type="mode_transition_safety",
                        severity="warning",
                        time=current.start_time,
                        x=current.start_x,
                        mode=current.gameplay_mode,
                        speed_state=current.speed_state.value,
                        message="mode transition has little recovery space",
                    )
                )
    return warnings


def validate_cube_jump_arc_approx(envelope: TrajectoryEnvelope, hazards: list[ObjectPlan]) -> list[PlayabilityWarning]:
    if envelope.mode != "cube":
        return []
    return [
        _warning("cube_jump_arc_hazard", envelope, f"hazard inside cube jump envelope x={hazard.x:.2f}", x=hazard.x)
        for hazard in hazards
        if envelope.start_x <= hazard.x <= envelope.end_x and envelope.min_y <= hazard.y <= envelope.max_y
    ]


def validate_robot_jump_window_approx(envelope: TrajectoryEnvelope, hazards: list[ObjectPlan]) -> list[PlayabilityWarning]:
    if envelope.mode != "robot":
        return []
    return validate_cube_jump_arc_approx(envelope, hazards)


def validate_wave_corridor_safety(envelope: TrajectoryEnvelope) -> list[PlayabilityWarning]:
    if envelope.mode == "wave" and (envelope.max_y - envelope.min_y) < envelope.safe_margin * 1.7:
        return [_warning("wave_corridor_too_narrow", envelope, "wave corridor is too narrow")]
    return []


def validate_ship_corridor_safety(envelope: TrajectoryEnvelope) -> list[PlayabilityWarning]:
    if envelope.mode == "ship" and (envelope.max_y - envelope.min_y) < envelope.safe_margin * 2.0:
        return [_warning("ship_corridor_too_narrow", envelope, "ship corridor is too narrow")]
    return []


def validate_trajectory_playability(
    section_plans: list[SectionPlan],
    gameplay_events: list[GameplayEvent] | None,
    object_plans: list[ObjectPlan],
    *,
    difficulty: str | float | int = "normal",
) -> list[PlayabilityWarning]:
    warnings: list[PlayabilityWarning] = []
    for section in section_plans:
        envelope = estimate_trajectory_envelope(
            section,
            gameplay_events=gameplay_events,
            object_plans=object_plans,
            difficulty=difficulty,
        )
        warnings.extend(envelope.warnings)
        warnings.extend(validate_cube_jump_arc_approx(envelope, list(_role_filter(object_plans, {"hazard"}))))
    warnings.extend(validate_mode_transition_trajectory(section_plans, gameplay_events))
    return warnings[:64]


def repair_playability_plans(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    *,
    difficulty: str | float | int = "normal",
) -> PlayabilityRepairReport:
    report = PlayabilityRepairReport()
    before_warnings = validate_trajectory_playability(section_plans, None, object_plans, difficulty=difficulty)
    report.spacing_score_before = _spacing_score(object_plans, section_plans, difficulty=difficulty)
    report.input_density_score_before = _input_density_score(object_plans, section_plans)

    _simplify_dense_orb_chains(object_plans, section_plans, report, difficulty=difficulty)
    _convert_dense_gameplay_to_decoration(object_plans, section_plans, report, difficulty=difficulty)
    _repair_portal_recovery(object_plans, section_plans, report, difficulty=difficulty)
    _preserve_drop_impact(object_plans, section_plans, report)

    after_warnings = validate_trajectory_playability(section_plans, None, object_plans, difficulty=difficulty)
    report.spacing_score_after = _spacing_score(object_plans, section_plans, difficulty=difficulty)
    report.input_density_score_after = _input_density_score(object_plans, section_plans)
    report.portal_safety_score = max(0.0, 1.0 - _warning_ratio(after_warnings, "portal"))
    report.mode_transition_score = max(0.0, 1.0 - _warning_ratio(after_warnings, "mode_transition"))
    report.orb_pad_timing_score = max(0.0, 1.0 - _warning_ratio(after_warnings, "input"))
    report.hazard_distance_score = max(0.0, 1.0 - _warning_ratio(after_warnings, "hazard"))
    report.speed_safety_score = 0.85 if any(obj.role == "speed_portal" for obj in object_plans) else 1.0
    report.section_difficulty_curve_score = _difficulty_curve_score(section_plans)
    if len(after_warnings) < len(before_warnings):
        report.warnings.append(f"playability_repair_reduced_warnings: {len(before_warnings)}->{len(after_warnings)}")
    return report


def _role_filter(objects: Iterable[ObjectPlan], roles: set[str]) -> Iterable[ObjectPlan]:
    for plan in objects:
        if plan.role in roles or (plan.object_id in {"8", "39"} and "hazard" in roles):
            yield plan


def _portal_filter(objects: Iterable[ObjectPlan]) -> Iterable[ObjectPlan]:
    portal_ids = {"12", "13", "47", "111", "660", "745", "1331", "200", "201", "202", "203", "1334"}
    for plan in objects:
        if plan.role == "speed_portal" or plan.object_id in portal_ids:
            yield plan


def _warning(
    warning_type: str,
    envelope: TrajectoryEnvelope,
    message: str,
    *,
    x: float | None = None,
) -> PlayabilityWarning:
    return PlayabilityWarning(
        warning_type=warning_type,
        severity="warning",
        time=envelope.start_time,
        x=envelope.start_x if x is None else x,
        mode=envelope.mode,
        speed_state=envelope.speed_state.value,
        message=message,
    )


def _difficulty_relief(difficulty: str | float | int) -> float:
    if isinstance(difficulty, (int, float)):
        value = max(0.0, min(1.0, float(difficulty)))
    else:
        value = {
            "easy": 0.1,
            "normal": 0.25,
            "hard": 0.45,
            "harder": 0.6,
            "insane": 0.75,
            "demon": 0.95,
        }.get(str(difficulty).lower(), 0.35)
    return 1.0 - min(0.45, value * 0.4)


def _simplify_dense_orb_chains(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    report: PlayabilityRepairReport,
    *,
    difficulty: str | float | int,
) -> None:
    for section in section_plans:
        profile = build_mode_physics_profile(section.gameplay_mode, section.speed_state, difficulty)
        inputs = [
            obj
            for obj in object_plans
            if section.start_x <= obj.x <= section.end_x and obj.role in {"beat_orb", "ai_orb", "orb", "beat_pad", "ai_pad", "jump_pad"}
        ]
        inputs.sort(key=lambda obj: obj.x)
        previous: ObjectPlan | None = None
        for obj in inputs:
            if previous is not None and obj.x - previous.x < profile.min_obstacle_spacing * 0.55:
                _convert_to_visual_accent(obj, section)
                report.simplified_dense_orb_chain += 1
            else:
                previous = obj


def _convert_dense_gameplay_to_decoration(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    report: PlayabilityRepairReport,
    *,
    difficulty: str | float | int,
) -> None:
    for section in section_plans:
        profile = build_mode_physics_profile(section.gameplay_mode, section.speed_state, difficulty)
        gameplay = [
            obj
            for obj in object_plans
            if section.start_x <= obj.x <= section.end_x
            and obj.role in {"beat_orb", "ai_orb", "orb", "beat_pad", "ai_pad", "jump_pad", "obstacle", "hazard", "structure_accent"}
        ]
        gameplay.sort(key=lambda obj: obj.x)
        last_x: float | None = None
        for obj in gameplay:
            if last_x is not None and obj.x - last_x < profile.min_obstacle_spacing * 0.42:
                _convert_to_visual_accent(obj, section)
                report.converted_gameplay_to_decoration += 1
                continue
            last_x = obj.x


def _repair_portal_recovery(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    report: PlayabilityRepairReport,
    *,
    difficulty: str | float | int,
) -> None:
    portal_ids = {"12", "13", "47", "111", "660", "745", "1331", "200", "201", "202", "203", "1334"}
    for section in section_plans:
        profile = build_mode_physics_profile(section.gameplay_mode, section.speed_state, difficulty)
        recovery_x = max(54.0, profile.portal_recovery_time * 311.58)
        portals = [
            obj for obj in object_plans
            if section.start_x <= obj.x <= section.end_x and (obj.role == "speed_portal" or obj.object_id in portal_ids)
        ]
        for portal in portals:
            for obj in object_plans:
                if not (portal.x < obj.x < portal.x + recovery_x):
                    continue
                if obj.role in {"hazard", "obstacle", "structure_accent"} or obj.object_id in {"8", "39"}:
                    _convert_to_visual_accent(obj, section)
                    report.removed_hazards_after_portal += 1
                elif obj.role in {"beat_orb", "beat_pad", "ai_orb", "ai_pad"}:
                    obj.x = portal.x + recovery_x
                    obj.safety_flags["playability_repair"] = "portal_recovery_margin"
                    report.recovery_margin_adjusted += 1


def _preserve_drop_impact(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    report: PlayabilityRepairReport,
) -> None:
    for section in section_plans:
        if section.section_type != "drop":
            continue
        in_section = [obj for obj in object_plans if section.start_x <= obj.x <= section.end_x]
        if any(obj.role in {"beat_orb", "beat_pad", "ai_structure", "visual_accent_target", "safe_decoration"} for obj in in_section):
            continue
        object_plans.append(
            ObjectPlan(
                object_id="500",
                x=section.start_x + max(30.0, (section.end_x - section.start_x) * 0.25),
                y=240.0,
                role="visual_accent_target",
                safety_flags={"source": "playability_repair", "section_id": section_plans.index(section)},
            )
        )
        report.warnings.append("playability_repair_preserved_empty_drop_with_visual_accent")


def _convert_to_visual_accent(plan: ObjectPlan, section: SectionPlan) -> None:
    plan.role = "visual_accent_target" if section.section_type == "drop" else "safe_decoration"
    plan.object_id = "500"
    plan.y = max(180.0, plan.y)
    plan.safety_flags["playability_repair"] = "converted_to_decoration"


def _spacing_score(
    object_plans: list[ObjectPlan],
    section_plans: list[SectionPlan],
    *,
    difficulty: str | float | int,
) -> float:
    violations = 0
    checked = 0
    for section in section_plans:
        profile = build_mode_physics_profile(section.gameplay_mode, section.speed_state, difficulty)
        gameplay = sorted(
            [
                obj for obj in object_plans
                if section.start_x <= obj.x <= section.end_x and obj.role in {"beat_orb", "beat_pad", "obstacle", "hazard", "structure_accent"}
            ],
            key=lambda obj: obj.x,
        )
        for prev, current in zip(gameplay, gameplay[1:]):
            checked += 1
            if current.x - prev.x < profile.min_obstacle_spacing * 0.55:
                violations += 1
    return max(0.0, 1.0 - violations / max(1, checked))


def _input_density_score(object_plans: list[ObjectPlan], section_plans: list[SectionPlan]) -> float:
    worst = 1.0
    for section in section_plans:
        width = max(1.0, section.end_x - section.start_x)
        inputs = [
            obj for obj in object_plans
            if section.start_x <= obj.x <= section.end_x and obj.role in {"beat_orb", "beat_pad", "ai_orb", "ai_pad"}
        ]
        density = len(inputs) / width
        target = max(0.004, section.density_target * 0.018)
        worst = min(worst, max(0.0, 1.0 - max(0.0, density - target) / target))
    return worst


def _warning_ratio(warnings: list[PlayabilityWarning], needle: str) -> float:
    count = sum(1 for warning in warnings if needle in warning.warning_type)
    return min(1.0, count / 4.0)


def _difficulty_curve_score(section_plans: list[SectionPlan]) -> float:
    if len(section_plans) < 2:
        return 1.0
    jumps = [
        abs(section_plans[idx + 1].difficulty_target - section_plans[idx].difficulty_target)
        for idx in range(len(section_plans) - 1)
    ]
    return max(0.0, 1.0 - max(jumps, default=0.0))
