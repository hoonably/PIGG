# Vanilla Evaluation: Zero-Shot GUI Agent 성능 평가

## 1. 개요 (Overview)

Vanilla Evaluation은 **사전 학습된 Qwen3-VL-8B-Instruct 모델을 어떠한 fine-tuning 없이 그대로 사용**하여, GUI 화면에서 사용자의 자연어 명령에 해당하는 클릭 좌표를 예측하는 Zero-Shot 성능을 측정하는 실험입니다.

이 실험의 목적은 다음과 같습니다:
- **Baseline 성능 확립**: Fine-tuning 전 모델의 기본 성능을 측정하여, 이후 Global/Personalized fine-tuning의 효과를 비교할 수 있는 기준점 마련
- **Zero-Shot Capability 검증**: Vision-Language Model이 GUI 이해 태스크에 대해 가지는 기본적인 능력 평가

---

## 2. 실험 데이터 (Dataset)

### 2.1 데이터 구조

각 샘플은 **대화 형식(Conversational Format)**으로 구성되어 있으며, 다음과 같은 구조를 가집니다:

```
[
  {
    "role": "system",
    "content": [
      {"type": "image", "image": "경로/스크린샷.jpg"},
      {"type": "text", "text": "프롬프트 및 지시사항"}
    ]
  },
  {
    "role": "assistant", 
    "content": [
      {"type": "text", "text": "정답 응답 (referring + grounding)"}
    ]
  }
]
```

### 2.2 입력 프롬프트 형식

모델에게 전달되는 프롬프트는 다음과 같은 형태입니다:

```
Given a GUI image, what are the relative (0-1000) pixel point coordinates 
for the element corresponding to the following instruction: [사용자 명령]

Think step-by-step, provide referring for the element first 
and then the grounded point coordinates.

### Output (Example)
```referring
"left arrow"
```
```grounding
(100, 200)
```


### 2.3 정답(Ground Truth) 형식

정답은 두 부분으로 구성됩니다:
1. **Referring**: 클릭해야 할 UI 요소에 대한 설명 (예: "파란색 더하기 버튼")
2. **Grounding**: 해당 요소의 좌표 [x, y] (0-1000 범위의 상대 좌표)

예시:

```referring
파란색 더하기 버튼
```
```grounding
[311, 541]
```


### 2.4 평가 대상 사용자

| 구분 | User ID | 샘플 수 |
|------|---------|---------|
| Test User 1 | 1 | 268 |
| Test User 2 | 2 | 1,310 |
| Test User 3 | 3 | 561 |
| Test User 4 | 4 | 1,058 |
| Test User 5 | 5 | 457 |
| Test User 6 | 6 | 305 |
| Test User 7 | 7 | 3,700 |
| Test User 8 | 8 | 255 |
| Test User 9 | 9 | 530 |
| Test User 10 | 10 | 1,619 |
| **합계** | - | **10,063** |

---

## 3. 실험 워크플로우 (Workflow)

### 3.1 전체 프로세스 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         VANILLA EVALUATION PIPELINE                     │
└─────────────────────────────────────────────────────────────────────────┘

  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   사용자     │     │   데이터셋       │     │    Qwen3-VL-8B      │
  │   지정       │────▶│   로드           │────▶│    모델 로드        │
  │   (1-10)     │     │   (pkl files)    │     │    (HuggingFace)    │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                          │
                                                          ▼
  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
  │   결과       │     │   좌표 비교      │     │    모델 추론        │
  │   저장       │◀────│   및 평가        │◀────│    (Generation)     │
  │   (JSON)     │     │   (L2, Accuracy) │     │                      │
  └──────────────┘     └──────────────────┘     └──────────────────────┘
```

### 3.2 단계별 상세 설명

#### **Step 1: 실험 환경 설정**

사용자가 명령줄에서 GPU와 평가할 유저 목록을 지정합니다.

```bash
python 1_vanilla_eval.py 0 --u 1 10  # GPU 0번 사용, User 1-10 평가
```

- GPU 환경변수 `CUDA_VISIBLE_DEVICES`를 설정하여 특정 GPU에서 실험 실행
- 유저 리스트 파싱: 두 개의 숫자가 주어지면 범위로 해석 (예: `1 10` → `[1,2,3,...,10]`)

#### **Step 2: 평가 데이터셋 로드**

`Qwen3VLEvalDataset` 클래스를 통해 지정된 사용자들의 데이터를 로드합니다.

**로드 과정:**
1. 각 유저별 pickle 파일(`user_{id}.pkl`)을 읽어옴
2. 모든 샘플을 하나의 리스트로 병합
3. Ground Truth 추출: 각 샘플의 assistant 메시지에서 정답 좌표와 referring content 파싱

**Ground Truth 추출 로직:**
- Assistant 메시지에서 ````referring` 블록과 ````grounding` 블록을 정규표현식으로 파싱
- 좌표는 다양한 형식 지원: `(x, y)`, `[x, y]`, `x, y` 등

#### **Step 3: 모델 로드**

Hugging Face에서 `Qwen/Qwen3-VL-8B-Instruct` 모델을 로드합니다.

**모델 설정:**
- `dtype="auto"`: 자동으로 적절한 데이터 타입 선택 (일반적으로 bfloat16)
- `trust_remote_code=True`: Qwen3-VL의 커스텀 코드 실행 허용
- 모델을 evaluation mode로 설정 (`model.eval()`)
- CUDA 디바이스로 이동 (`model.to("cuda")`)

#### **Step 4: 입력 전처리**

각 샘플에 대해 다음 전처리를 수행합니다:

1. **Chat Template 적용**: 
   - 시스템 메시지를 user 역할로 변환 (Qwen3-VL 호환성)
   - `add_generation_prompt=True`로 응답 생성 준비

2. **이미지 처리**:
   - 이미지 경로 추출
   - `AutoProcessor`를 통해 이미지를 모델 입력 형식으로 변환
   - `max_pixels = 1280 * 28 * 28` 해상도로 리사이즈

3. **텐서 변환**:
   - 텍스트와 이미지를 PyTorch 텐서로 변환
   - 배치 차원 처리 (squeeze)

#### **Step 5: 모델 추론 (Inference)**

`torch.inference_mode()` 컨텍스트에서 추론을 수행합니다.

**생성 설정:**
- `max_new_tokens=512`: 최대 512 토큰까지 생성
- 모델이 referring과 grounding을 포함한 응답 생성

**출력 디코딩:**
- `tokenizer.decode()`로 토큰 시퀀스를 텍스트로 변환
- `skip_special_tokens=True`로 특수 토큰 제거

#### **Step 6: 예측 좌표 파싱**

생성된 텍스트에서 예측 좌표를 추출합니다.

**파싱 로직 (`parse_inference` 함수):**
1. Assistant 응답 부분만 추출 (`text.split("assistant")[-1]`)
2. Referring content 추출: ````referring` 블록 파싱
3. Grounding 좌표 추출: 다양한 패턴 순차 적용
   - 4값 좌표: `(x1, y1, x2, y2)`
   - to 형식: `(x1, y1) to (x2, y2)`
   - 키워드 형식: `Top-left: (x, y)`
   - 단순 형식: `[x, y]` 또는 `(x, y)`

#### **Step 7: 평가 메트릭 계산**

예측 좌표와 정답 좌표를 비교하여 성능을 측정합니다.

**평가 지표:**

1. **Click Accuracy (δ=140)**
   - L2 거리가 δ(140픽셀) 이하인 샘플의 비율
   - $\text{Accuracy} = \frac{1}{N}\sum_{i=1}^{N} \mathbb{1}[\|pred_i - gt_i\|_2 \leq \delta]$

2. **Mean L2 Error**
   - 모든 샘플의 L2 거리 평균
   - $\text{Mean L2} = \frac{1}{N}\sum_{i=1}^{N} \|pred_i - gt_i\|_2$

3. **Median L2 Error**
   - L2 거리의 중앙값 (이상치에 강건)

#### **Step 8: 결과 저장**

각 유저별로 결과를 저장합니다.

**저장 경로:** `./eval_results/qwen3_vanilla/user_{id}/`

**저장 파일:**
- `out.json`: 각 샘플별 상세 결과
  - `pred_raw`: 모델의 원본 출력
  - `pred_coord`: 파싱된 예측 좌표
  - `gt_coord`: 정답 좌표
  - `pred_content`: 예측된 referring content
  - `gt_content`: 정답 referring content
  
- `results.txt`: 요약 메트릭 (accuracy, mean_l2, median_l2)

---

## 4. 핵심 설계 결정 (Key Design Decisions)

### 4.1 Zero-Shot 설정

- **Fine-tuning 없음**: 모델의 사전 학습된 지식만으로 GUI 이해 능력 평가
- **프롬프트 엔지니어링**: 명확한 출력 형식 예시 제공으로 일관된 응답 유도

### 4.2 좌표 표현

- **상대 좌표 (0-1000)**: 다양한 해상도의 스크린샷에 대응 가능
- **단일 포인트**: 클릭 위치를 하나의 (x, y) 좌표로 표현

### 4.3 평가 기준

- **δ=140 픽셀**: 일반적인 UI 요소 크기를 고려한 허용 오차
  - 대부분의 버튼, 아이콘이 이 범위 내에 포함됨
  - 상대 좌표 1000 기준, 14%의 허용 범위

---

## 5. 실험 결과 예시

실행 후 다음과 같은 형태의 결과가 출력됩니다:

```
===== User 1 =====
{
    'accuracy': 0.4328,
    'mean_l2': 187.23,
    'median_l2': 142.56
}
```

---

## Appendix A: 함수별 상세 설명

### A.1 메인 스크립트 (`1_vanilla_eval.py`)

#### `vanilla_eval(user_list)`
- **목적**: 지정된 유저 리스트에 대해 vanilla 모델 평가 수행
- **입력**: `user_list` - 평가할 유저 ID 리스트
- **동작**:
  1. 각 유저에 대해 반복
  2. `eval_user()` 호출하여 평가 수행
  3. 결과 출력 및 저장
- **출력**: 각 유저별 결과를 콘솔에 출력하고 파일로 저장

---

### A.2 평가 스크립트 (`scripts/eval_qwen3.py`)

#### `eval_user(checkpoint_path, user_eval, save_root, is_lora=False, eval_dataset=None)`
- **목적**: 특정 체크포인트로 지정된 유저들을 평가
- **입력**:
  - `checkpoint_path`: 모델 경로 (vanilla의 경우 `"Qwen/Qwen3-VL-8B-Instruct"`)
  - `user_eval`: 평가할 유저 ID 리스트
  - `save_root`: 결과 저장 경로
  - `is_lora`: LoRA 어댑터 사용 여부
  - `eval_dataset`: 사전 생성된 평가 데이터셋 (선택)
- **동작**:
  1. 체크포인트 경로 확인 및 최신 체크포인트 탐색
  2. 모델 로드 (vanilla/fine-tuned/LoRA 구분)
  3. DataLoader를 통해 배치 단위 추론
  4. 각 샘플의 예측 결과를 JSON으로 실시간 저장
  5. 전체 평가 메트릭 계산
- **출력**: 평가 결과 딕셔너리 (`accuracy`, `mean_l2`, `median_l2`)

---

### A.3 데이터셋 클래스 (`dataset.py`)

#### `Qwen3VLEvalDataset`
- **목적**: 평가용 데이터셋 - 전처리된 입력과 Ground Truth 반환
- **초기화 (`__init__`)**:
  - 유저별 pickle 파일 로드
  - AutoProcessor 초기화 (이미지 전처리용)
  - Ground Truth 사전 추출
- **`_extract_ground_truth()`**:
  - 각 샘플의 assistant 메시지에서 정답 파싱
  - `parse_inference()` 함수로 좌표 추출
- **`__getitem__(idx)`**:
  - 입력 메시지를 chat template으로 변환
  - 이미지와 텍스트를 processor로 전처리
  - 반환: `(processed_inputs, gt_content, gt_coord)`

---

### A.4 유틸리티 함수 (`utils/eval.py`)

#### `parse_inference(text, is_gt=True)`
- **목적**: 모델 출력 또는 정답 텍스트에서 좌표 추출
- **입력**:
  - `text`: 파싱할 텍스트
  - `is_gt`: 정답 텍스트 여부 (False면 assistant 부분만 추출)
- **동작**:
  1. Referring content 추출 (```referring 블록)
  2. Grounding 좌표 추출 (다양한 정규표현식 패턴 순차 적용)
- **출력**: `(content, x, y)` 튜플

#### `evaluate_coordinates(pred_coords, gt_coords, delta=140)`
- **목적**: 예측 좌표와 정답 좌표 비교하여 메트릭 계산
- **입력**:
  - `pred_coords`: 예측 좌표 리스트 `[(x1,y1), (x2,y2), ...]`
  - `gt_coords`: 정답 좌표 리스트
  - `delta`: Click Accuracy 임계값 (기본 140)
- **동작**:
  1. NumPy 배열로 변환
  2. L2 거리 계산: $\|pred - gt\|_2$
  3. 각 메트릭 계산
- **출력**: `{"accuracy": float, "mean_l2": float, "median_l2": float}`

#### `save_results(result, save_path)`
- **목적**: 평가 결과를 파일로 저장
- **입력**: `result` - 결과 딕셔너리, `save_path` - 저장 경로
- **동작**: `results.txt` 파일에 결과 기록

---

### A.5 설정 상수 (`config.py`)

| 상수 | 값 | 설명 |
|------|-----|------|
| `MODEL_NAME` | `"Qwen/Qwen3-VL-8B-Instruct"` | 기본 모델 |
| `MAX_PIXELS` | `1280 * 28 * 28` | 이미지 최대 해상도 |
| `DATASET_ROOT` | `"./dataset_pkl"` | 데이터셋 경로 |
| `NUM_WORKERS` | `4` | DataLoader 워커 수 |
| `USER_TEST` | `[1, 2, ..., 10]` | 테스트 유저 ID |

---

## Appendix B: 데이터 흐름도

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW DIAGRAM                               │
└─────────────────────────────────────────────────────────────────────────────┘

user_1.pkl ─┐
user_2.pkl ─┼──▶ Qwen3VLEvalDataset ──▶ DataLoader ───┐
   ...      │         │                               │
user_10.pkl─┘         │                               │
                      ▼                               │
              ┌───────────────┐                       │
              │ Ground Truth  │                       │
              │ Extraction    │                       │
              │ - gt_coords   │                       │
              │ - gt_contents │                       │
              └───────────────┘                       │
                                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           INFERENCE LOOP                                    │
│                                                                             │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   Batch     │───▶│  Processor   │───▶│   Model     │───▶│  Tokenizer  │  │
│  │   (image +  │    │  (encode)    │    │  .generate()│    │  .decode()  │  │
│  │   prompt)   │    │              │    │             │    │             │  │
│  └─────────────┘    └──────────────┘    └─────────────┘    └─────────────┘  │
│                                                                    │        │
│                                                                    ▼        │
│                                                          ┌─────────────────┐│
│                                                          │ parse_inference ││
│                                                          │ - pred_coord    ││
│                                                          │ - pred_content  ││
│                                                          └─────────────────┘│
└─────────────────────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │  evaluate_coordinates   │
                            │  - Click Accuracy@140   │
                            │  - Mean L2 Error        │
                            │  - Median L2 Error      │
                            └─────────────────────────┘
                                         │
                                         ▼
                            ┌─────────────────────────┐
                            │      RESULTS            │
                            │  - out.json (per sample)│
                            │  - results.txt (summary)│
                            └─────────────────────────┘
```

---

## Appendix C: 실행 명령어 예시

```bash
# 기본 실행 (GPU 0, User 1-10)
python 1_vanilla_eval.py 0

# 특정 GPU 지정
python 1_vanilla_eval.py 1

# 특정 유저 범위 지정
python 1_vanilla_eval.py 0 --u 1 5    # User 1-5만 평가

# 개별 유저 지정
python 1_vanilla_eval.py 0 --u 1 3 7  # User 1, 3, 7만 평가
```

---

*문서 작성일: 2025년 12월 3일*
*프로젝트: PIGG (Personalized Interactive GUI Grounding)*
