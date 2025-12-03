# Global Fine-tuning: 다중 사용자 GUI Agent 학습

## 1. 개요 (Overview)

Global Fine-tuning은 **다수의 사용자 데이터를 통합하여 하나의 범용 GUI Agent 모델을 학습**하는 실험입니다. Qwen3-VL-8B-Instruct 모델을 기반으로, 73명의 사용자(User 11-83) 데이터를 사용하여 Supervised Fine-tuning(SFT)을 수행합니다.

이 실험의 목적은 다음과 같습니다:
- **범용 GUI Agent 구축**: 다양한 사용자 패턴을 학습하여 새로운 사용자에게도 적용 가능한 모델 개발
- **Personalized Fine-tuning의 기반 모델 생성**: 개인화 학습의 시작점(Base Checkpoint)으로 활용
- **Zero-Shot 대비 성능 향상 검증**: Vanilla 모델 대비 Fine-tuning의 효과 측정

---

## 2. 학습 방법론 (Training Methodology)

### 2.1 두 가지 Fine-tuning 방식

본 실험에서는 두 가지 Fine-tuning 방식을 지원합니다:

| 방식 | 설명 | 학습 파라미터 | 메모리 사용량 |
|------|------|--------------|--------------|
| **LoRA** | Low-Rank Adaptation | ~0.5% | 낮음 |
| **Full Fine-tuning** | 전체 LLM 파라미터 학습 | ~99% (Vision 제외) | 높음 |

### 2.2 공통 설계 원칙

Qwen3-VL 공식 가이드라인에 따라 다음 원칙을 적용합니다:

1. **Vision Encoder Freezing**: 사전 학습된 시각 인코더의 파라미터 고정
2. **Multi-modal Projector Freezing**: Vision-Language 정렬 레이어 고정
3. **Assistant-only Label Masking**: 어시스턴트 응답 토큰만 학습 대상으로 설정
4. **Gradient Checkpointing**: 메모리 효율적 학습을 위한 기법 적용

---

## 3. 데이터 구성 (Data Configuration)

### 3.1 사용자 분할

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER DATA SPLIT                           │
├─────────────────────────────────────────────────────────────────┤
│  Training Users    │  User 11 - 83  │  73명  │  90,727 샘플    │
│  Validation Users  │  (비활성화)     │  0명   │  0 샘플         │
│  Test Users        │  User 1 - 10   │  10명  │  10,063 샘플    │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 데이터 형식

각 샘플은 대화 형식으로 구성됩니다:

```json
[
  {
    "role": "user",
    "content": [
      {"type": "image", "image": "스크린샷 경로"},
      {"type": "text", "text": "GUI 명령어 프롬프트"}
    ]
  },
  {
    "role": "assistant",
    "content": [
      {"type": "text", "text": "referring + grounding 응답"}
    ]
  }
]
```

---

## 4. 학습 워크플로우 (Training Workflow)

### 4.1 전체 프로세스 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      GLOBAL FINE-TUNING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   환경 설정   │     │   Qwen3-VL 모델  │     │   LoRA/Full FT      │
  │   (GPU, Log) │────▶│   로드           │────▶│   설정 적용          │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                          │
                                                          ▼
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   체크포인트  │     │   Trainer        │     │   데이터셋 로드      │
  │   저장       │◀────│   학습 수행      │◀────│   (73명 유저)        │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
```

### 4.2 단계별 상세 설명

#### **Step 1: 환경 설정 및 로깅 초기화**

사용자가 명령줄에서 GPU와 학습 방식을 지정합니다.

```bash
python 2_finetune_global.py 0 --lora    # GPU 0, LoRA 학습
python 2_finetune_global.py 0,1         # GPU 0,1, Full Fine-tuning
```

**로깅 시스템 (TeeLogger):**
- 표준 출력과 로그 파일에 동시 기록
- 타임스탬프가 포함된 로그 파일 자동 생성
- 경로: `./logs/finetune_global_{method}_{timestamp}.log`

#### **Step 2: 모델 및 프로세서 로드**

Hugging Face에서 Qwen3-VL-8B-Instruct 모델을 로드합니다.

**프로세서 설정:**
- `Qwen3VLProcessor`: 이미지와 텍스트 전처리 담당
- `max_pixels = 1280 * 28 * 28`: 이미지 해상도 설정

**모델 설정:**
- `torch_dtype=torch.bfloat16`: 메모리 효율적인 16비트 부동소수점
- `device_map="auto"`: 다중 GPU 자동 분산

#### **Step 3: Fine-tuning 방식 적용**

**LoRA 학습의 경우:**
```python
LoraConfig(
    r=8,                          # Rank: 저랭크 분해 차원
    lora_alpha=32,                # 스케일링 팩터
    target_modules="all-linear",  # 모든 Linear 레이어에 적용
    lora_dropout=0.1,             # 드롭아웃 비율
    bias="none",                  # Bias 학습 안함
    task_type="CAUSAL_LM"         # 인과적 언어 모델링
)
```

**Full Fine-tuning의 경우:**
- LoRA 없이 LLM 전체 파라미터 학습
- Vision Encoder와 Projector는 동일하게 Freeze

#### **Step 4: Vision 컴포넌트 Freezing**

공식 Best Practice에 따라 다음 컴포넌트를 고정합니다:

1. **Vision Encoder (vision_tower/visual)**
   - 사전 학습된 이미지 특성 추출기 보존
   - 대규모 이미지 데이터로 학습된 표현력 유지

2. **Multi-modal Projector**
   - Vision-Language 정렬 레이어
   - Vision과 LLM 사이의 연결고리 역할

**Freezing 효과:**
- 학습 파라미터 수 감소
- 과적합 방지
- 학습 안정성 향상

#### **Step 5: 데이터셋 및 Data Collator 준비**

**Qwen3VLTrainDataset:**
- 지정된 유저들의 pickle 파일 로드
- 모든 샘플을 하나의 리스트로 통합
- 'system' 역할을 'user'로 변환 (Qwen3-VL 호환성)

**Qwen3VLDataCollator:**
- `apply_chat_template(tokenize=True)`: 공식 토크나이징 방식
- **Assistant-only Label Masking**: 핵심 학습 기법

#### **Step 6: Label Masking 상세 설명**

Label Masking은 어떤 토큰을 학습 대상으로 삼을지 결정합니다.

**Qwen3-VL Chat Template 구조:**
```
<|im_start|>user
[이미지 + 프롬프트]<|im_end|>
<|im_start|>assistant
[응답 - 이 부분만 학습!]<|im_end|>
```

**마스킹 로직:**
1. `<|im_start|>assistant` 패턴 탐색
2. `<|im_end|>` 까지의 토큰을 학습 대상으로 설정
3. 나머지 토큰은 `IGNORE_INDEX(-100)`으로 마스킹

**수식 표현:**
$$
\text{labels}_i = 
\begin{cases}
\text{input\_ids}_i & \text{if } i \in \text{assistant\_response} \\
-100 & \text{otherwise}
\end{cases}
$$

#### **Step 7: 학습 인자 설정**

| 하이퍼파라미터 | 값 | 설명 |
|--------------|-----|------|
| `learning_rate` | 1e-6 | 공식 권장 범위 (1e-6 ~ 2e-7) |
| `num_train_epochs` | 1 | 에폭 수 |
| `per_device_train_batch_size` | 1 | GPU당 배치 크기 |
| `gradient_accumulation_steps` | 8 | 유효 배치 크기 = 8 |
| `warmup_steps` | 50 | 워밍업 스텝 |
| `weight_decay` | 0.01 | 가중치 감쇠 |
| `bf16` | True | BFloat16 혼합 정밀도 |
| `gradient_checkpointing` | True | 메모리 최적화 |
| `save_steps` | 500 | 체크포인트 저장 주기 |

#### **Step 8: Trainer 학습 실행**

Hugging Face `Trainer`를 사용하여 학습을 수행합니다.

**주요 특징:**
- 자동 체크포인트 저장 (매 500 스텝)
- 최근 3개 체크포인트만 유지 (디스크 절약)
- Safetensors 형식 사용 (안정성)
- 검증 비활성화 (OOM 방지)

#### **Step 9: 모델 저장**

학습 완료 후 최종 모델을 저장합니다.

**저장 경로:**
- LoRA: `/workspace/PIGG_checkpoints/global_agent_lora`
- Full: `/workspace/PIGG_checkpoints/global_agent_full`

**저장 항목:**
- 모델 가중치 (LoRA의 경우 어댑터만)
- Processor (토크나이저 포함)
- 학습 설정

---

## 5. LoRA vs Full Fine-tuning 비교

### 5.1 학습 파라미터 비교

**LoRA Fine-tuning:**
```
trainable params: 41,943,040 / 8,294,967,296 (0.51%)
```

**Full Fine-tuning:**
```
trainable params: 8,222,674,944 / 8,294,967,296 (99.13%)
(Vision Encoder + Projector 제외)
```

### 5.2 장단점 비교

| 특성 | LoRA | Full Fine-tuning |
|------|------|------------------|
| **메모리 사용량** | 낮음 (~16GB) | 높음 (~40GB+) |
| **학습 속도** | 빠름 | 느림 |
| **성능** | 준수 | 최고 |
| **저장 용량** | ~100MB | ~16GB |
| **유연성** | 어댑터 교체 가능 | 전체 모델 필요 |

---

## 6. 메모리 최적화 기법

### 6.1 Gradient Checkpointing

중간 활성화값을 저장하지 않고 역전파 시 재계산하여 메모리 절약.

**적용 방법:**
```python
model.gradient_checkpointing_enable()
```

### 6.2 Mixed Precision Training

BFloat16을 사용한 혼합 정밀도 학습.

**장점:**
- 메모리 사용량 50% 감소
- 학습 속도 향상
- 수치 안정성 유지 (bf16의 넓은 dynamic range)

### 6.3 Device Map Auto

다중 GPU 환경에서 자동으로 모델 분산.

```python
model = Model.from_pretrained(..., device_map="auto")
```

---

## 7. 실행 예시

### 7.1 LoRA Fine-tuning

```bash
# GPU 0에서 LoRA 학습
python 2_finetune_global.py 0 --lora

# 출력 예시
================================================================================
Qwen3-VL Global GUI Agent Fine-tuning
================================================================================
Training users: 73 users
  User IDs: 11-83
Method: LoRA
...
Training samples: 90727
...
Starting training...
```

### 7.2 Full Fine-tuning

```bash
# GPU 0,1에서 Full Fine-tuning
python 2_finetune_global.py 0,1

# Multi-GPU 자동 분산
```

---

## Appendix A: 함수별 상세 설명

### A.1 메인 스크립트 (`2_finetune_global.py`)

#### `finetune_global_agent(use_lora=True, log_file=None)`
- **목적**: Global GUI Agent Fine-tuning 수행
- **입력**:
  - `use_lora`: LoRA 사용 여부
  - `log_file`: 로그 파일 경로
- **동작**:
  1. 학습 방식에 따라 `train_lora()` 또는 `train_full()` 호출
  2. 체크포인트 경로 반환
- **출력**: 저장된 체크포인트 경로

#### `TeeLogger`
- **목적**: 표준 출력과 파일에 동시 로깅
- **메서드**:
  - `write(message)`: 터미널과 파일에 동시 출력
  - `flush()`: 버퍼 플러시
  - `close()`: 파일 핸들 닫기

---

### A.2 학습 스크립트 (`scripts/train_qwen3.py`)

#### `train_lora(checkpoint_dir, user_train, user_val=None)`
- **목적**: LoRA를 사용한 Qwen3-VL 학습
- **입력**:
  - `checkpoint_dir`: 체크포인트 저장 경로
  - `user_train`: 학습 유저 ID 리스트
  - `user_val`: 검증 유저 ID 리스트 (선택)
- **동작**:
  1. Processor 및 모델 로드
  2. LoRA 설정 적용 (`get_peft_model`)
  3. Vision 컴포넌트 Freezing
  4. 데이터셋 및 Collator 생성
  5. Trainer로 학습 수행
  6. 최종 모델 저장

#### `train_full(checkpoint_dir, user_train, user_val=None)`
- **목적**: 전체 파라미터 Fine-tuning
- **동작**: LoRA 설정 제외하고 `train_lora()`와 유사

---

### A.3 유틸리티 (`utils/train_utils.py`)

#### `Qwen3VLDataCollator`
- **목적**: 배치 데이터 전처리 및 Label 생성
- **핵심 메서드**:
  - `__call__(instances)`: 배치 collation 수행
  - `_create_labels_with_assistant_masking(input_ids)`: Assistant 토큰만 학습 대상으로 마스킹

#### `get_training_arguments(output_dir, **kwargs)`
- **목적**: TrainingArguments 생성
- **기본값**: config.py의 하이퍼파라미터 사용
- **kwargs**: 추가 인자로 기본값 오버라이드 가능

#### `prepare_model_for_training(model)`
- **목적**: 학습을 위한 모델 준비
- **동작**:
  - Gradient checkpointing 활성화
  - Input gradients 활성화 (PEFT 호환)

---

### A.4 데이터셋 클래스 (`dataset.py`)

#### `Qwen3VLTrainDataset`
- **목적**: 학습용 데이터셋
- **초기화**: 지정된 유저들의 pickle 파일 로드
- **`__getitem__(idx)`**: 
  - Raw 메시지 반환 (전처리는 Collator에서)
  - 'system' → 'user' 역할 변환

---

### A.5 설정 상수 (`config.py`)

| 상수 | 값 | 설명 |
|------|-----|------|
| `LR` | `1e-6` | 학습률 |
| `EPOCHS` | `1` | 에폭 수 |
| `BATCH_SIZE` | `1` | 배치 크기 |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | 기울기 누적 |
| `SAVE_STEPS` | `500` | 저장 주기 |
| `WARMUP_STEPS` | `50` | 워밍업 |
| `WEIGHT_DECAY` | `0.01` | 가중치 감쇠 |
| `USER_TRAIN` | `[11, ..., 83]` | 학습 유저 (73명) |
| `IGNORE_INDEX` | `-100` | 마스킹 인덱스 |

---

## Appendix B: 데이터 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRAINING DATA FLOW                                 │
└─────────────────────────────────────────────────────────────────────────────┘

user_11.pkl ─┐
user_12.pkl ─┤
    ...      ├──▶ Qwen3VLTrainDataset ──▶ DataLoader ──┐
    ...      │         │                                │
user_83.pkl ─┘         │                                │
                       ▼                                │
               ┌───────────────┐                        │
               │ Raw Messages  │                        │
               │ (image + text │                        │
               │  + response)  │                        │
               └───────────────┘                        │
                                                        ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA COLLATION PROCESS                               │
│                                                                              │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────┐ │
│  │ apply_chat_     │───▶│ Label Masking   │───▶│ Batch Tensors           │ │
│  │ template        │    │ (Assistant-only)│    │ - input_ids             │ │
│  │ (tokenize=True) │    │                 │    │ - labels                │ │
│  └─────────────────┘    └─────────────────┘    │ - attention_mask        │ │
│                                                 │ - pixel_values          │ │
│                                                 │ - image_grid_thw        │ │
│                                                 └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TRAINING LOOP                                   │
│                                                                              │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐ │
│  │   Forward   │───▶│   Loss       │───▶│  Backward   │───▶│  Optimizer  │ │
│  │   Pass      │    │  Computation │    │   Pass      │    │   Step      │ │
│  │             │    │  (CE Loss)   │    │             │    │             │ │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘ │
│                                                                              │
│                     Loss = CrossEntropy(logits, labels)                      │
│                     (labels == -100인 토큰은 무시)                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │   CHECKPOINT SAVING     │
                            │   - Every 500 steps     │
                            │   - Final model         │
                            │   - Processor           │
                            └─────────────────────────┘
```

---

## Appendix C: 실행 명령어

```bash
# LoRA Fine-tuning (권장: 단일 GPU)
python 2_finetune_global.py 0 --lora

# Full Fine-tuning (권장: 다중 GPU)
python 2_finetune_global.py 0,1

# 특정 GPU 지정
python 2_finetune_global.py 2 --lora

# 로그 확인
tail -f ./logs/finetune_global_lora_*.log
```

---

## Appendix D: 트러블슈팅

### D.1 CUDA Out of Memory

**증상:** GPU 메모리 부족 오류

**해결책:**
1. LoRA 사용 (`--lora` 플래그)
2. `gradient_accumulation_steps` 증가
3. 다중 GPU 사용 (`device_map="auto"`)
4. `per_device_train_batch_size=1` 유지

### D.2 학습 손실이 감소하지 않음

**증상:** Loss가 고정되거나 증가

**확인 사항:**
1. Label masking 로그 확인: "Trainable tokens found" > 0
2. 학습률이 너무 높지 않은지 확인 (권장: 1e-6)
3. 데이터 형식 검증

### D.3 체크포인트 저장 실패

**증상:** 디스크 공간 부족

**해결책:**
1. `save_total_limit=3`으로 체크포인트 수 제한
2. 충분한 디스크 공간 확보 (Full: ~50GB, LoRA: ~1GB)

---

*문서 작성일: 2025년 12월 3일*
*프로젝트: PIGG (Personalized Interactive GUI Grounding)*
