# AI GMD Generation Guide (Technical Implementation)

이 문서는 `gmdgen` 프로젝트의 내부 아키텍처와 머신러닝 모델 설계 방안을 딥러닝 전문가의 관점에서 설명합니다. 본 프로젝트는 단순히 객체를 배치하는 것을 넘어, 음악과 레벨 구조 간의 정교한 동기화와 물리적 타당성(Playability)을 보장하기 위한 고도의 모델링 기법을 사용합니다.

---

## 1. 데이터 표현 및 시간 매핑 (Time Mapping & Representation)

Geometry Dash 레벨 생성에서 가장 큰 기술적 도전 과제는 **시간(Audio Time)과 위치(X-Position)의 비선형적 관계**를 처리하는 것입니다.

### 1.1 Speed State & Segment
`src/gmdgen/gd/time_mapping.py`에 구현된 바와 같이, 레벨 내의 Speed Portal은 시간당 이동 거리를 변화시킵니다.
- **Positional Encoding**: 단순한 좌표 대신, 속도 변화를 고려한 `pos_for_time_like_gd` 함수를 통해 오디오 비트와 정확히 일치하는 X 좌표를 계산해야 합니다.
- **Consistency Constraint**: 모델은 `L_time_to_x_consistency` 손실 함수를 통해 생성된 객체의 위치가 물리적인 속도 세그먼트와 일치하는지 강제적으로 학습해야 합니다.

### 1.2 Feature-Aware Tokenization (Representation Learning)
단순한 객체 ID 시퀀스를 넘어, `src/gmdgen/representation/tokenizer.py`는 Goodfellow의 "Deep Learning" (Ch.15) 이론에 기반한 **인과 요인 분리(Disentangling Causal Factors)** 기법을 사용합니다.

각 객체는 다음과 같은 복합 토큰(Composite Token)으로 인코딩됩니다:
- **Format**: `OBJ:{id}|CLS:{class}|DX:{bucket}|Y:{bucket}|SEC:{id}`
- **DX Buckets**: 이전 객체와의 상대적 X 거리를 로그 스케일 빈(Bin)으로 구분하여 학습 효율 극대화.
- **Y Bands**: 절대적 Y 좌표를 8개의 수직 존(Zone)으로 매핑하여 공간적 층(Level) 인지.
- **Semantic Class**: 객체의 역할(Solid, Damage, Trigger, Portal 등)을 명시적으로 구분.

---

## 2. 모델 아키텍처: Audio-Conditioned Model


본 프로젝트의 핵심 모델 사양(`AudioConditionedModelSpec`)은 다음과 같은 구성 요소를 가집니다.

### 2.1 Audio Encoder (CNN-Transformer)
음악에서 추출된 Mel-spectrogram, Onset curve, Beat grid를 입력으로 받아 리듬과 섹션 변화를 임베딩합니다.
- **Layers**: 2D CNN (Spectral pattern), Temporal CNN (Onset/Energy), Transformer Encoder (Long-range beat dependencies).

### 2.2 Fusion Mechanism (Cross-Attention)
오디오 임베딩과 레벨 논리 임베딩을 결합하여 다음을 생성합니다:
- **Section Plan**: 레벨의 전체적인 밀도 및 분위기 구성.
- **Gameplay Event Plan**: 비트에 맞춘 점프 및 비행 이벤트 배치.

### 2.3 Structured Generator
최종 출력은 단순한 텐서가 아닌, `SectionPlan`, `SpeedPlan`, `ObjectPlan` 등의 구조화된 객체 시퀀스입니다. 
- **Decoding Strategy**: Beam Search 또는 에디터 제약 조건을 반영한 Candidate Search를 사용합니다.

### 2.4 오디오 피처와 GD 오브젝트 대응 (Domain-Specific Mapping)
단순한 확률적 생성을 넘어, 음악의 물리적 특성에 따라 다음과 같은 GD 객체 배치를 우선적으로 고려해야 합니다.

| Audio Signal | GD Object Candidate |
| :--- | :--- |
| **강한 비트 (Strong Beat)** | Jump Pad, Orb, 구조적 변화, Deco Accent |
| **다운비트 (Downbeat)** | Section Boundary, Game Mode Change, Speed Portal |
| **강한 온셋 (Onset)** | Pulse, Alpha, Shake, Short Move Trigger |
| **에너지 상승 (Buildup)** | Object Density 증가, Trigger Buildup |
| **멜로디 피치 (Melody Pitch)** | `pitch_height_hint` 기반 Y축 (Height) 배치 |
| **드롭 (Drop)** | Mode/Speed/Structure 변화, Pulse/Shake 강조 |

---

## 3. 하이브리드 전략: Heuristic + Machine Learning

완전한 End-to-End 학습보다는, GD의 물리 엔진 특성을 반영한 하이브리드 접근법이 권장됩니다.

### 3.1 Candidate Search & Scoring
현재 시스템은 단순한 생성 모델이 아닌, 여러 후보를 생성하고 `scoring.py`를 통해 최적의 후보를 선택하는 **Candidate Search** 방식을 병행합니다.
- **Generation Passes**: 생성 모델의 반복 횟수.
- **Deliberation Width**: 최적의 경로를 찾기 위한 빔 서치(Beam Search) 너비.

### 3.2 물리 기반 제약 조건 (Hard Constraints)
모델이 생성한 결과물에 대해 다음과 같은 규칙 기반 수리(Repair)가 수반됩니다.
- **Beat Snap**: 비트 오차 허용 범위(`beat_snap_tolerance`) 내로 객체 강제 정렬.
- **Speed Portal Sorting**: 속도 포탈 배치 후 모든 비트-X 좌표의 즉각적인 재계산.

### 3.3 RAG 기반의 검색 증강 계획 (RAG-Augmented Planning)
`src/gmdgen/ai/context_index.py` 및 `LocalKeywordRetriever`를 통해, 모델은 생성 과정에서 실시간으로 프로젝트 내의 지식 베이스를 참조합니다.
- **Context Retrieval**: `docs/` 폴더의 트리거 스키마(`trigger_schema.md`)나 플레이 가능성 규칙(`code_validation.md`) 중 현재 생성 구간에 가장 관련 있는 내용을 검색하여 프롬프트에 주입.
- **Reference Level Injection**: `dataset/`에 포함된 고품질 레벨의 특정 패턴을 검색하여 모델이 유사한 구조를 생성하도록 유도.

### 3.4 계층적 멀티 에이전트 계획 (Hierarchical Multi-Agent Planning)
`src/gmdgen/ai/prompts.py`에 정의된 프롬프트 체계는 복잡한 레벨 생성 과정을 여러 단계의 전문화된 에이전트로 분할하여 처리합니다.

1.  **Global Planner**: 오디오 전체 구조를 분석하여 섹션(Intro, Drop 등) 및 밀도 계획 수립.
2.  **Section Planner**: 특정 섹션 내의 리드미컬한 게임플레이 뼈대(Skeleton) 구축.
3.  **Object/Trigger Planner**: 게임플레이 이벤트를 실제 객체와 트리거 계획으로 구체화.
4.  **Critic & Revisionist**: 생성된 계획의 논리적 오류(Sync Error, Empty Drop 등)를 검토하고 수정하는 사후 교정 루프 수행.

---

## 4. 손실 함수 설계 (Loss Functions)




학습 시 단순한 Cross-Entropy 외에도 GD의 특성을 반영한 다차원 손실 함수가 적용됩니다.

| Loss Term | Purpose | Signal |
| :--- | :--- | :--- |
| `L_beat_sync` | 게임플레이 이벤트가 비트에 정확히 배치되는지 확인 | Nearest beat error |
| `L_playability` | 점프 불가능한 간격이나 안전하지 않은 전환 방지 | Rule simulation penalties |
| `L_trigger_validity` | 트리거 타겟 및 지속 시간의 유효성 검증 | Trigger graph checks |
| `L_editor_validity` | 에디터에서 로드 및 저장이 가능한 규격인지 확인 | Roundtrip/import checks |
| `L_energy_density` | 음악의 에너지 엔벨로프에 따른 객체 밀도 조절 | Density-energy correlation |

---

## 4. 플레이 가능성 검증 (Playability & Validation)

AI가 생성한 데이터는 `ValidationReport`를 통해 검증됩니다. 
- **Editor-Safe Encoding**: 생성된 데이터는 GD 에디터에서 문제없이 열려야 하며, `GeodeBridge`를 통해 실제 게임 엔진과의 호환성을 테스트합니다.
- **Physics Simulation**: 플레이어의 점프 궤적을 시뮬레이션하여 "Dead-zone"이 발생하는지 사전에 차단합니다.

---

## 5. 학습 데이터 파이프라인

1.  **Extraction**: `.gmd` 파일에서 `ObjectPlan` 및 `TriggerPlan` 추출.
2.  **Normalization**: 속도 상태별로 정규화된 시간축 데이터로 변환.
3.  **Augmentation**: 오디오 피치 변경 및 레벨 섹션 재배치를 통한 데이터 증강.

---

## 6. 출력 직렬화 및 검증 (Output Serialization & Validation)

모델이 생성한 구조화된 데이터(Plan)는 최종적으로 게임이 읽을 수 있는 `.gmd` 파일로 변환되어야 합니다.

### 6.1 GMD Writer (IO Pipeline)
`src/gmdgen/io/gmd_writer.py`와 `src/gmdgen/output/save.py`를 통해 직렬화가 수행됩니다.
- **Level String Generation**: 모델이 출력한 객체 토큰들은 다시 `1,1,2,50,3,60;` 형태의 구분자 문자열로 복원되어야 합니다.
- **Plist/XML Mapping**: `.gmd` 파일은 XML 기반의 Plist 규격을 따르며, 핵심 레벨 데이터는 특정 키(k4 등) 아래에 인코딩되어 저장됩니다.

### 6.2 자동화된 검증 스크립트
생성된 레벨의 품질을 확인하기 위해 다음 프로젝트 도구들을 활용하십시오:
- `scripts/validate.py`: 생성된 `.gmd` 파일의 구문 오류 및 플레이 가능성 검증.
- `scripts/generate.py --audit`: 생성된 결과물에 대한 AI 기반의 사후 감사(Audit) 리포트 생성.

---

## 7. 데이터셋 수집 및 파인튜닝 (Dataset & Fine-Tuning)

`gmdgen`은 지속적인 모델 개선을 위한 데이터 수집 파이프라인을 포함하고 있습니다.

### 7.1 High-Quality 데이터 추출
`src/gmdgen/ai/fine_tune_export.py`를 통해 실제 생성 결과물 중 품질이 검증된 사례를 학습 데이터로 변환할 수 있습니다.
- **Selection Criteria**: `validation_score`가 높고, `repair_loss`(수정 손실)가 적으며, 사용자 평점(`user_rating`)이 높은 데이터 위주로 선별.
- **Structured JSONL**: 학습 데이터는 원시 문자열이 아닌, 모델이 이해하기 쉬운 **구조화된 계획(Structured Plans)** 형태로 내보내집니다.

### 7.2 자가 학습 루프 (Self-Improvement Loop)
1.  **Generate**: 현재 모델을 사용하여 다양한 레벨 생성.
2.  **Evaluate**: `scripts/validate.py` 및 사용자 피드백을 통해 품질 평가.
3.  **Export**: 고품질 데이터를 `fine_tune_export.py`로 추출하여 새로운 학습셋 구축.
4.  **Train**: 추출된 데이터로 모델 재학습(Fine-tuning).

---

## 8. 학술적 배경 및 심화 학습 (Academic Foundations)

`docs/AI/` 디렉토리에 포함된 학술 논문 및 서적들은 `gmdgen` 프로젝트의 설계 철학과 AI 아키텍처에 직접적인 이론적 근거를 제공합니다. AI 엔지니어는 시스템 구조를 개선할 때 다음 문헌의 핵심 개념을 참조해야 합니다.

### 8.1 딥러닝 기초 및 최적화 메커니즘
*   **참고 문헌**: *The Little Book of Deep Learning (François Fleuret)*
*   **프로젝트 연관성**: 
    *   **Transformer 아키텍처 (Chapter 5.3)**: `AudioConditionedModel`이 음악의 긴 시퀀스(Long-range dependencies)와 레벨 구조를 매핑하는 데 필수적인 Multi-Head Attention 메커니즘의 기반을 제공합니다.
    *   **SGD & Adam Optimizer (Chapter 3.3)**: 강화학습 기반의 Playability Validation 모델 및 수리(Repair) 모듈의 파라미터 최적화에 적용됩니다. 특히 희소한 보상 환경에서의 그래디언트 업데이트 전략은 이 문헌을 기초로 합니다.

### 8.2 파운데이션 모델 및 프롬프트 엔지니어링
*   **참고 문헌**: *Foundation Models for Natural Language Processing (Gerhard Paaß & Sven Giesselbach)*
*   **프로젝트 연관성**:
    *   **Autoregressive Models & GPT (Chapter 2.2)**: `gmdgen`의 객체 및 트리거 시퀀스 생성 방식은 이전 객체들의 상태를 기반으로 다음 객체를 예측하는 GPT의 자동 회귀 방식을 차용하고 있습니다.
    *   **Retrieval-Augmented Generation (RAG) (Chapter 6.2)**: Gemini 기반 프롬프팅 파이프라인에서 `LocalKeywordRetriever`가 `dataset/` 폴더 내의 기존 레벨 패턴과 튜토리얼을 주입하는 구조는 최신 RAG 모델의 컨텍스트 강화 기법과 일치합니다.
    *   **Chain-of-Thought (CoT) Prompting**: 시스템 프롬프트에 포함된 계층적 계획(Global -> Section -> Object)은 언어 모델의 논리적 추론 능력을 극대화하기 위한 CoT 방법론의 적용 사례입니다.

### 8.3 컴퓨터 비전 및 다중 모달리티
*   **참고 문헌**: *Deep Learning in Computer Vision (Mahmoud Hassaballah et al.)*
*   **프로젝트 연관성**:
    *   **CNN 기반 특징 추출 (Chapter 2)**: 음악 파일에서 변환된 스펙트로그램(Spectrogram) 2D 이미지 데이터로부터 드롭(Drop), 비트(Beat), 싱크(Sync) 등의 핵심 오디오 특징을 추출하는 `CNNTransformerAudioEncoder`의 1D/2D Convolutional 설계에 이론적 배경을 제공합니다.

---

이 지침서는 AI 전문가가 지오메트리 대쉬 레벨의 **내부 논리(Internal Logic)**와 **이론적 근간(Theoretical ML Foundations)**을 이해하고, 이를 통해 단순한 복제가 아닌 **창의적이고 구조적으로 완전하며 실행 가능한 레벨**을 생성하는 모델을 구축하는 데 초점을 맞추고 있습니다. 프로젝트의 `src/gmdgen/gd/` 및 `src/gmdgen/ml/` 모듈을 수시로 참조하여 아키텍처 사양을 최신으로 유지하십시오.



