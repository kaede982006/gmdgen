# Geode Integration

gmdgen still treats Python time-X mapping as an approximate mapper. The Geode
bridge is an optional validation layer for comparing against recorded or future
runtime Geode/GD results.

Current bridge implementations:

- `NullGeodeBridge`: default. Reports Geode as unavailable and never crashes
  generation.
- `OptionalGeodeFixtureBridge`: loads recorded fixture values for parity tests.
  Synthetic fixtures do not prove real GD parity.
- `ExternalProcessGeodeBridge`: JSON subprocess skeleton for a future Geode
  helper/mod. It is intentionally not wired to unknown Geode APIs.

Generation behavior:

- If Geode is unavailable, the generator uses the Python approximate mapper and
  records `geode_available=false`.
- If fixture/runtime data is available, the generator can compare `pos_for_time`
  and `time_for_pos` results and report average/max errors.
- Geode unavailable is a warning, not a crash.
- A future runtime bridge must validate protocol version, timeouts, stdout/stderr
  JSON, and temporary file cleanup.

Do not claim actual GD parity until outputs from the real GD/Geode
`LevelTools::posForTime` / `timeForPos` path are captured and passing.
