# Audio-Conditioned Generation Hardening

The audio-conditioned generator now uses:

- audio analysis with confidence reporting
- approximate GD time-X mapping with fixture comparison hooks
- structured `SectionPlan`, `ObjectPlan`, and `TriggerPlan`
- trigger schema validation before encoding
- conservative playability and trajectory envelope validators
- editor-safety round-trip checks before reporting final output

The playability and trajectory checks are not a full Geometry Dash physics
engine. They are conservative validators designed to catch unsafe density,
portal recovery, corridor, and hazard-margin problems before editor import.

Run the main validation suite:

```powershell
python -m compileall .\src\gmdgen .\tests
python -m pytest -q
```
