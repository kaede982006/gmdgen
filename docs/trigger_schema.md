# Trigger Schema Notes

Trigger support is split into three practical classes:

- `safe`: conservative triggers with known minimal keys that can be encoded in
  safe mode.
- `advanced`: trigger types with partial schema support that need stronger
  GD 2.2 key-map confirmation before broad generation.
- `unsupported`: trigger/property combinations that must not be emitted.

The registry is in `src/gmdgen/gd/triggers.py`.

Safe mode only emits conservative keys such as object id, x/y, duration,
target group, spawn delay, multi-trigger, and editor-disable. Properties with
unknown or unverified GD 2.2 save keys are kept as schema metadata and are not
encoded by `encode_trigger_properties_safe`.

Run:

```powershell
python -m pytest -q tests\test_trigger_schema.py tests\test_editor_safety.py
```
