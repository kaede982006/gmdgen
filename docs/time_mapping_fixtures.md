# Time-X Fixture Format

This project still uses an approximate Geometry Dash time-to-X mapper. Real
parity requires fixture data exported from GD/Geode `LevelTools::posForTime`
and `LevelTools::timeForPos`.

JSON fixtures live under `tests/fixtures/time_mapping/`.

Required top-level fields:

- `name`
- `start_speed`: `slow`, `normal`, `fast`, `faster`, or `fastest`
- `song_offset`
- `speed_objects`: optional list of `{time, x, speed_state, object_id}`
- `samples`: list of time-to-X or X-to-time checks

Sample fields:

- `time` + `expected_x` for `posForTime`
- `x` + `expected_time` for `timeForPos`
- `tolerance`
- `source`: use `synthetic_approximate` until real Geode exports are available

Run:

```powershell
python -m pytest -q tests\test_time_fixtures.py tests\test_time_mapping.py
```
