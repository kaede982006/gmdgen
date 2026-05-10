# Optimization Strategy (ML View)

## 1. Goal: Error Reduction via Candidate Selection
Since the current system uses a symbolic planner (Gemini) and a deterministic materializer, we do not perform traditional backpropagation. Instead, we use **Random Restart Search** (via `candidate_count`) to find the candidate that minimizes our multi-term "Loss Function" (represented as 1.0 - Score).

## 2. Defined Losses (to be minimized)

### L_schema (Planner Schema Loss)
- **Error**: Malformed JSON, missing fields (`level_plan`, `sections`), wrong field locations.
- **Metric**: `planner_status` (fallback = max loss), `missing_required_fields` count.
- **Reduction Strategy**: Prompt hardening, one-shot repair pass, field normalization.

### L_sync (Music Sync Loss)
- **Error**: Objects not aligned with beats or onsets.
- **Metric**: `max_sync_error`, `mean_sync_error`.
- **Reduction Strategy**: Improved `pos_for_time` mapping, beat-snapping during materialization.

### L_structure (Structural Stability Loss)
- **Error**: Reverse X-monotone objects, overcrowded sections.
- **Metric**: `x_monotone_fixed` count (repairer output), `overcrowding_penalty`.
- **Reduction Strategy**: Penalty for high `x_monotone_fixed` even if `repair_loss` is low.

### L_playability (Playability Loss)
- **Error**: Impossible jumps, orb chains too dense.
- **Metric**: `playability_score`, `playability_warning_count`.
- **Reduction Strategy**: Trajectory validation, conservative spacing rules.

### L_density (Density Alignment Loss)
- **Error**: Generated density differs from requested section density.
- **Metric**: `density_target_error`.
- **Reduction Strategy**: Adjusting object probability in materializer based on section density.

## 3. Implementation Directions

- **Incorporate Structural Fixes into Scoring**: `x_monotone_fixed` should be a negative term in `AudioConditionedScore`.
- **Differentiate AI Success from Fallback**: Candidates produced via `planner_status=fallback` must receive a massive score penalty to ensure AI success is always preferred if valid.
- **Loss-based Feedback for Repair**: When Gemini fails, provide the specific "Loss" (error message) to the repair prompt (One-shot repair already does this).

## 4. Future Gradient Descent Paths
If we transition to a truly neural materializer:
- **Target**: Predicted Object ID and relative X/Y.
- **Loss**: Cross-entropy for IDs, MSE for relative positions.
- **Conditioning**: Beat/Onset features as context.
