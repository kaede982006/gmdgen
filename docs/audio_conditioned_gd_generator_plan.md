# Audio-Conditioned Geometry Dash Level Generator Plan

## 1. 핵심 진단

기존 editor AI는 Geometry Dash 오브젝트 배열을 그럴듯하게 샘플링하지만, 음악 시간과 플레이 시간의 관계를 직접 모델링하지 않는다. 그래서 BPM, beat, onset, song offset, start speed, speed portal, trigger duration, spawn delay, group 관계가 생성 목적 함수에 들어가지 않는다. 결과적으로 raw map generator는 음악 싱크가 아니라 오브젝트 분포 복제에 가깝다.

## 2. GD 내부 로직 기준으로 본 문제점

`base/gd_decompile` 기준으로 GD에는 `GJGameLevel`, `LevelSettingsObject`, `DrawGridLayer`, `GJBaseGameLayer`, `GameObject`, `EffectGameObject`, `LevelTools`에 나뉜 상태가 있다. 특히 `LevelTools::posForTime`, `posForTimeInternal`, `timeForPos`, `sortSpeedObjects`, `valueForSpeedMod`는 beat time을 X좌표로 변환하는 핵심 개념이다. 따라서 speed portal 이후 beat X좌표를 재계산하지 않는 생성기는 GD editor/playback 기준에서 싱크가 어긋난다.

## 3. 오디오 feature 추출 설계

Frame level feature:
- RMS
- onset strength
- spectral flux
- mel-like band energy

Beat level feature:
- beat time
- beat strength
- downbeat flag
- local energy
- rhythmic density

Section level feature:
- intro
- buildup
- drop
- bridge/high energy
- outro
- calm/silence

초기 구현은 `src/gmdgen/audio/analysis.py`의 dependency-light WAV analyzer를 사용한다. 이후 librosa/torch frontend를 선택적으로 붙일 수 있다.

## 4. song offset / speed portal / time-to-X mapping 설계

개념식:

```text
x = posForTime(t, speed_objects, start_speed, song_offset, speed_state)
```

구현 절차:

1. audio time에서 song offset을 반영한다.
2. LevelSettingsObject의 start speed를 `SpeedState`로 둔다.
3. section boundary/downbeat/drop에서 speed portal 후보를 만든다.
4. 각 speed portal의 X를 이전 speed state로 먼저 계산한다.
5. speed objects를 X 기준으로 정렬한다.
6. 모든 beat time을 현재 speed state로 다시 X 변환한다.
7. `timeForPos`식 역변환으로 sync error를 측정한다.
8. beat snap tolerance를 넘는 이벤트는 repair 대상으로 둔다.

금지:
- beat마다 고정 X 간격 사용
- BPM만으로 level length 계산
- speed portal 배치 후 beat-to-X 재계산 생략
- song offset 무시
- preview/playback roundtrip 검증 생략

## 5. 오디오 feature와 GD 오브젝트 대응표

| Audio signal | GD candidate |
|---|---|
| 강한 beat | jump pad, orb, 구조 변화, decoration accent |
| downbeat | section boundary, game mode change, speed portal, color change |
| 강한 onset | pulse, alpha, shake, short move trigger, glow accent |
| energy 상승 | object density 증가, decoration intensity 증가 |
| energy 하락 | 여백, 쉬운 패턴, transition |
| buildup | density ramp, speed 후보, trigger buildup |
| drop | mode/speed/structure change, pulse/shake 강조 |
| silence/break | 여백, sync reset, color/camera transition |
| repeated rhythm | repeated gameplay motif |
| spectral 변화 | color/pulse/background 후보 |

모든 beat에 object를 놓지 않는다. `audio_sync_strength`, `object_budget`, `difficulty`, `max_events_per_beat`로 제한한다.

## 6. ObjectPlan / TriggerPlan 중간 표현

직접 save string을 만들지 않고 다음 구조를 먼저 만든다.

- `AudioEvent`
- `GDTimeEvent`
- `SectionPlan`
- `ObjectPlan`
- `TriggerPlan`
- `ValidationReport`

구현 위치:
- `src/gmdgen/gd/plans.py`
- `src/gmdgen/generate/audio_conditioned.py`

## 7. loss 또는 scoring function

현재 시스템은 gradient descent가 아니라 candidate search다. 따라서 loss는 beam/candidate scoring objective로 사용한다.

```text
L_total =
  L_beat_sync
  + L_onset_sync
  + L_section_sync
  + L_time_to_x_consistency
  + L_speed_portal_consistency
  + L_energy_density
  + L_style_consistency
  + L_playability
  + L_trigger_validity
  + L_group_validity
  + L_editor_validity
  + L_object_budget
```

구현 위치:
- `src/gmdgen/generate/scoring.py`

## 8. 생성 알고리즘

1. `audio_file`을 로드한다.
2. `song_offset`을 반영한다.
3. BPM, beat, downbeat, onset, RMS, section boundary를 추출한다.
4. LevelSettingsObject 대응 시작 상태를 만든다.
5. section feature를 `SectionPlan`으로 바꾼다.
6. section별 gameplay mode, speed, density, decoration intensity를 할당한다.
7. `allow_speed_portals`가 true이면 speed portal 후보를 만든다.
8. speed objects를 만들고 정렬한다.
9. beat time을 `posForTime`식 X좌표로 변환한다.
10. `timeForPos`식 역검증을 수행한다.
11. beat-aligned gameplay event 후보를 만든다.
12. onset-aligned trigger 후보를 만든다.
13. reference level style profile을 추출해 Y/object class sampling에 반영한다.
14. ObjectPlan / TriggerPlan을 생성한다.
15. group ID를 할당한다.
16. trigger target 존재 여부를 검사한다.
17. spawn delay, duration, multi-trigger를 검사한다.
18. object_budget에 맞게 pruning한다.
19. rule-based playability validation을 수행한다.
20. editor-safe repair를 수행한다.
21. conservative save string encoder로 최종 string을 만든다.
22. import/export parser roundtrip을 검증한다.
23. ValidationReport를 출력한다.

## 9. 후처리 및 검증 규칙

- beat에서 먼 주요 event snap/경고
- speed portal 이후 time-X 재계산
- beat당 이벤트 과밀 제한
- density smoothing
- grid/x monotone repair
- speed objects sorting
- orphan trigger 제거
- group ID bounds 검사
- spawn delay/duration 검사
- stop trigger target 검사
- safe_mode에서 위험한 trigger/high detail 축소
- object count/capacity 고려
- editor preview guideline과 event alignment 검증
- save string 좌표계와 key를 보수적으로 encode

## 10. 기존 파라미터 재해석

- `generation_passes`: gradient descent가 아니라 candidate 반복 생성 횟수
- `candidate_pool_size`: 후보 유지 수
- `deliberation_width`: beam width
- `safe_simplify_objects`: regularization
- `chunk_transition_counts`: local transition statistics, 이후 audio-conditioned transition으로 확장

## 11. 신규 파라미터

- `audio_sync_strength`
- `beat_snap_tolerance`
- `onset_event_threshold`
- `section_change_sensitivity`
- `energy_density_scale`
- `drop_emphasis`
- `speed_portal_policy`
- `max_events_per_beat`
- `trigger_safety_level`
- `group_id_policy`
- `editor_roundtrip_check`

## 12. 구현 우선순위

1. audio analysis + beat grid + song offset
2. GD식 time-X mapper
3. speed portal 없는 normal speed placement
4. speed portal 추가와 beat-to-X 재계산
5. energy envelope 기반 density
6. onset 기반 trigger 후보
7. group allocator와 trigger validator
8. ObjectPlan / TriggerPlan 중간 표현
9. scoring objective 확장
10. safe_mode/editor repair 강화
11. reference motif 변형
12. CNN/Transformer audio encoder spec
13. 데이터가 충분할 때 supervised/imitation learning

## 13. 의사코드

```python
def generate_audio_synced_level(args):
    audio = load_audio(args.audio_file)
    features = analyze_audio(audio, args.song_offset)

    level_settings = build_level_settings(
        start_speed=args.start_speed,
        song_offset=args.song_offset,
        custom_song=True,
    )

    sections = segment_audio(features)
    section_plans = plan_sections(sections, args.difficulty)

    speed_plan = plan_speed_portals(
        section_plans,
        policy=args.speed_portal_policy,
        allow_speed_portals=args.allow_speed_portals,
    )

    speed_objects = build_and_sort_speed_objects(speed_plan)

    beat_x_map = {}
    for beat in features.beats:
        x = pos_for_time_like_gd(
            beat.time,
            speed_objects,
            level_settings.start_speed,
            level_settings.song_offset,
        )
        beat_x_map[beat.index] = x

    gameplay_events = plan_gameplay_events(
        beats=features.beats,
        beat_x_map=beat_x_map,
        sync_strength=args.audio_sync_strength,
        difficulty=args.difficulty,
    )

    trigger_events = plan_trigger_events(
        onsets=features.onsets,
        energy=features.energy,
        beat_x_map=beat_x_map,
        allow_triggers=args.allow_triggers,
    )

    object_plans = generate_objects_from_style(
        style_reference_level=args.style_reference_level,
        section_plans=section_plans,
        gameplay_events=gameplay_events,
    )

    trigger_plans = generate_triggers(trigger_events)
    assign_group_ids(object_plans, trigger_plans, args.group_id_policy)
    repair_result = repair_and_validate(object_plans, trigger_plans, speed_objects)
    score = evaluate_candidate(repair_result, features)
    level_string = encode_to_gd_save_string(repair_result)
    return level_string, score, validation_report
```

## 14. 한 줄 요약

Geometry Dash 음악 싱크 생성은 단순히 beat마다 오브젝트를 놓는 문제가 아니다. song offset, start speed, speed portal, posForTime/timeForPos식 변환, editor guideline, trigger/group 관계, playability를 모두 반영하는 audio-conditioned structured generation 문제다. 초기 구현은 딥러닝보다 GD 로직 기반 time-X mapper, beat/onset scoring, trigger/group validator, editor-safe repair부터 만드는 것이 가장 현실적이다.
