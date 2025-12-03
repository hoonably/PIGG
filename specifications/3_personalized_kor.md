# Personalized Fine-tuning: 개인화 GUI Agent 학습 및 평가

## 1. 개요 (Overview)

Personalized Fine-tuning은 **Global Fine-tuning으로 학습된 모델을 기반으로, 개별 사용자의 데이터를 활용하여 해당 사용자에게 최적화된 GUI Agent를 학습**하는 실험입니다.

이 실험의 핵심 특징:
- **K-Fold Cross Validation**: 각 사용자의 데이터를 K개의 fold로 분할하여 학습 및 평가의 신뢰성 확보
- **전이 학습 (Transfer Learning)**: Global 모델의 지식을 기반으로 개인화 수행
- **Full Fine-tuning**: LLM 전체 파라미터를 학습하여 최대 성능 확보
- **학습 후 즉시 평가**: 각 fold 학습 완료 후 바로 해당 fold의 테스트셋으로 평가

### 1.1 실험 목적

1. **개인화 효과 검증**: Global 모델 대비 개인화 모델의 성능 향상 정도 측정
2. **사용자별 특성 학습**: 개별 사용자의 앱 사용 패턴, 선호 UI 요소 등을 학습
3. **과적합 방지**: K-Fold를 통해 작은 데이터셋에서도 신뢰할 수 있는 성능 평가

---

## 2. 실험 구조 (Experimental Structure)

### 2.1 전체 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERSONALIZED FINE-TUNING ARCHITECTURE                     │
└─────────────────────────────────────────────────────────────────────────────┘

                        ┌─────────────────────────┐
                        │   Global Full Model     │
                        │   (Pre-trained on       │
                        │    73 users' data)      │
                        └───────────┬─────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
            ┌───────────┐   ┌───────────┐   ┌───────────┐
            │  User 1   │   │  User 2   │   │  User 10  │
            │Personalized│   │Personalized│   │Personalized│
            │ Full Model│   │ Full Model│...│ Full Model│
            └───────────┘   └───────────┘   └───────────┘
                 │               │               │
         ┌───────┴───────┐      ...             ...
         ▼       ▼       ▼
     ┌──────┐┌──────┐┌──────┐
     │Fold 0││Fold 1││Fold 2│
     └──────┘└──────┘└──────┘
```

### 2.2 K-Fold Cross Validation 구조

각 사용자에 대해 K=3 fold로 데이터를 분할합니다:

```
User Data (예: 268 samples)
     │
     ▼
┌────────────────────────────────────────────────────────┐
│                    K-Fold Split (K=3)                   │
├──────────────────┬─────────────────┬───────────────────┤
│     Fold 0       │     Fold 1      │     Fold 2        │
│   (90 samples)   │  (89 samples)   │  (89 samples)     │
└──────────────────┴─────────────────┴───────────────────┘

Iteration 1: Train on Fold 1+2 (178 samples), Test on Fold 0 (90 samples)
Iteration 2: Train on Fold 0+2 (179 samples), Test on Fold 1 (89 samples)
Iteration 3: Train on Fold 0+1 (179 samples), Test on Fold 2 (89 samples)
```

### 2.3 데이터 분할

| 구분 | User ID | 역할 |
|------|---------|------|
| **Training (Global)** | 11-83 | Global 모델 학습에 사용 |
| **Test (Personalized)** | 1-10 | 개인화 학습 및 평가 대상 |

---

## 3. 실험 데이터 (Dataset)

### 3.1 테스트 사용자별 데이터 현황

| User ID | Total Samples | Fold 0 | Fold 1 | Fold 2 | Train (per fold) |
|---------|---------------|--------|--------|--------|------------------|
| 1 | 268 | 90 | 89 | 89 | ~178 |
| 2 | 1,310 | 437 | 437 | 436 | ~873 |
| 3 | 561 | 187 | 187 | 187 | ~374 |
| 4 | 1,058 | 353 | 353 | 352 | ~705 |
| 5 | 457 | 153 | 152 | 152 | ~304 |
| 6 | 305 | 102 | 102 | 101 | ~203 |
| 7 | 3,700 | 1,234 | 1,233 | 1,233 | ~2,466 |
| 8 | 255 | 85 | 85 | 85 | ~170 |
| 9 | 530 | 177 | 177 | 176 | ~353 |
| 10 | 1,619 | 540 | 540 | 539 | ~1,079 |
| **합계** | **10,063** | - | - | - | - |

### 3.2 Fold 분할 로직

```python
def create_k_folds(samples, k=3, seed=42):
    random.seed(seed)  # 재현성을 위한 고정 시드
    shuffled_samples = samples.copy()
    random.shuffle(shuffled_samples)
    
    fold_size = len(shuffled_samples) // k
    folds = []
    
    for i in range(k):
        if i == k - 1:
            fold = shuffled_samples[i * fold_size:]  # 마지막 fold는 나머지 포함
        else:
            fold = shuffled_samples[i * fold_size:(i + 1) * fold_size]
        folds.append(fold)
    
    return folds
```

**핵심 설계**:
- **고정 시드 (seed=42)**: 학습과 평가에서 동일한 fold 분할 보장
- **균등 분할**: 가능한 한 동일한 크기로 분할
- **마지막 fold**: 나머지 샘플을 모두 포함하여 데이터 손실 방지

---

## 4. 실험 워크플로우 (Workflow)

### 4.1 전체 프로세스 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERSONALIZED FINE-TUNING PIPELINE                         │
└─────────────────────────────────────────────────────────────────────────────┘

For each User (1-10):
  For each Fold (0, 1, 2):
    
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
    │  사용자      │     │   K-Fold         │     │    Global Full      │
    │  데이터 로드 │────▶│   데이터 분할    │────▶│    모델 로드        │
    │              │     │                  │     │                      │
    └──────────────┘     └──────────────────┘     └──────────────────────┘
                                                            │
                                                            ▼
    ┌──────────────┐     ┌──────────────────┐     ┌──────────────────────┐
    │  Fold 평가   │     │   개인화 학습    │     │    LoRA 어댑터      │
    │  (Hold-out)  │◀────│   (K-1 folds)    │◀────│    추가 (선택)      │
    │              │     │                  │     │                      │
    └──────────────┘     └──────────────────┘     └──────────────────────┘
           │
           ▼
    ┌──────────────────────────────────────────────────────────────────────┐
    │                         결과 저장 및 집계                             │
    │  - 각 Fold별 결과: ./eval_results/personalized/user_{id}/fold_{idx}  │
    │  - 사용자별 평균: user_{id}_summary.json                              │
    └──────────────────────────────────────────────────────────────────────┘
```

### 4.2 단계별 상세 설명

#### **Step 1: 환경 설정 및 인자 파싱**

```bash
python 4_finetune_person.py 0 --users 1 2 3  # GPU 0, User 1,2,3
```

- **GPU 지정**: `CUDA_VISIBLE_DEVICES` 환경변수 설정
- **사용자 선택**: 특정 사용자 또는 전체 테스트 사용자 (1-10)
- **Fold 선택**: 특정 fold만 학습하거나 전체 K개 fold 학습

#### **Step 2: 사용자 데이터 로드 및 K-Fold 분할**

1. **데이터 로드**: `user_{id}.pkl` 파일에서 해당 사용자의 전체 샘플 로드
2. **K-Fold 생성**: 고정 시드로 일관된 fold 분할 생성
3. **학습/테스트 분리**: 
   - 학습 데이터: 현재 fold를 제외한 (K-1)개 fold
   - 테스트 데이터: 현재 hold-out fold

#### **Step 3: 모델 로드**

**Base 모델**: Global Full Fine-tuned 모델을 기반으로 사용

```python
model = Qwen3VLForConditionalGeneration.from_pretrained(
    CHECKPOINT_GLOBAL_FULL,  # Global Full 체크포인트
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
```

#### **Step 4: Vision Encoder 및 Aligner 동결**

Qwen3-VL 공식 권장 사항에 따라:

```python
# Vision Encoder 동결
for param in vision_tower.parameters():
    param.requires_grad = False

# Multi-modal Projector (Aligner) 동결
for param in multi_modal_projector.parameters():
    param.requires_grad = False
```

**이유**: 
- Vision 이해 능력은 사전 학습으로 이미 충분히 습득
- Language Model 부분만 fine-tuning하여 과적합 방지

#### **Step 5: 개인화 학습**

**학습 하이퍼파라미터**:

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `PERSON_EPOCHS` | 3 | 학습 에폭 수 |
| `PERSON_BATCH_SIZE` | 1 | 배치 크기 |
| `PERSON_GRADIENT_ACCUMULATION_STEPS` | 4 | 그래디언트 누적 (effective batch=4) |
| `PERSON_LR` | 2e-5 | 학습률 (Global보다 높음) |
| `PERSON_WARMUP_STEPS` | 5 | 워밍업 스텝 |
| `PERSON_WEIGHT_DECAY` | 0.01 | 가중치 감쇠 |

**Global vs Personalized 하이퍼파라미터 비교**:

| 파라미터 | Global | Personalized | 이유 |
|---------|--------|--------------|------|
| Learning Rate | 1e-6 | 2e-5 | 작은 데이터셋에서 빠른 적응 필요 |
| Epochs | 1 | 3 | 데이터 크기가 작아 더 많은 반복 필요 |
| Effective Batch | 8 | 4 | 데이터 다양성이 낮아 더 빈번한 업데이트 |
| Warmup Steps | 50 | 5 | 작은 데이터셋에 맞게 축소 |

#### **Step 6: 모델 저장**

학습 완료 후 체크포인트 저장:

```
/workspace/PIGG_checkpoints/personalized_full/user_{id}/fold_{idx}/
  ├── config.json               # 모델 설정
  ├── model.safetensors         # 모델 가중치
  └── ...
```

#### **Step 7: Fold 평가**

학습 직후 해당 fold의 테스트셋으로 평가:

1. **테스트 데이터셋 생성**: Hold-out fold 샘플 사용
2. **추론 수행**: 각 샘플에 대해 좌표 예측
3. **메트릭 계산**: Accuracy, Mean L2, Median L2

#### **Step 8: 메모리 정리**

OOM 방지를 위한 적극적인 메모리 관리:

```python
del trainer, model
torch.cuda.empty_cache()
gc.collect()
```

각 fold 사이에 충분한 GPU 메모리 확보를 위해 다중 정리 라운드 수행

#### **Step 9: 결과 집계**

**Fold별 결과 저장**:
```
./eval_results/personalized/user_{id}/fold_{idx}/
  ├── out.json      # 샘플별 상세 결과
  └── results.txt   # 요약 메트릭
```

**사용자별 K-Fold 평균 결과**:
```json
{
    "user_id": 1,
    "n_folds": 3,
    "avg_accuracy": 0.7234,
    "avg_mean_l2": 89.45,
    "avg_median_l2": 67.23,
    "std_accuracy": 0.0156,
    "std_mean_l2": 5.67,
    "std_median_l2": 4.32
}
```

---

## 5. 핵심 설계 결정 (Key Design Decisions)

### 5.1 Global Full → Personalized Full 전이 학습

**왜 Global Full을 base로 사용하는가?**

1. **더 강력한 기반 성능**: Full fine-tuning은 모든 파라미터가 업데이트되어 더 많은 지식 습득
2. **최대 표현력 확보**: 개인화 단계에서도 전체 파라미터 학습으로 사용자 특성 완전 반영
3. **일관된 학습 방식**: Global → Personalized 전 과정에서 동일한 Full Fine-tuning 적용

### 5.2 K-Fold Cross Validation 사용 이유

1. **작은 데이터셋**: 개인별 데이터가 255~3,700 샘플로 상대적으로 작음
2. **신뢰성 있는 평가**: 단일 train/test split의 편향 방지
3. **모든 데이터 활용**: 모든 샘플이 한 번씩 테스트에 사용됨

### 5.3 K=3 선택 이유

- **K=5 대비 장점**: 각 fold의 학습 데이터 비율이 높음 (67% vs 80%)
- **데이터 효율성**: 작은 데이터셋에서 충분한 학습 데이터 확보
- **계산 비용**: 3회 학습으로 적절한 시간 내 완료

### 5.4 Full Fine-tuning 선택 이유

| 장점 | 설명 |
|------|------|
| **최대 표현력** | 전체 파라미터 학습으로 사용자 특성 완전 반영 |
| **성능 최적화** | 개인화 데이터에 완전히 적응 가능 |
| **일관성** | Global과 동일한 학습 방식 유지 |

**Full Fine-tuning 특성**:
- **학습 파라미터**: ~99% (Vision Encoder 제외)
- **메모리 사용**: 높음 (~40GB+)
- **저장 용량**: ~16GB per model

---

## 6. 평가 메트릭 (Evaluation Metrics)

### 6.1 개별 Fold 메트릭

각 fold에서 계산되는 메트릭:

1. **Click Accuracy (δ=140)**
   - L2 거리가 140픽셀 이하인 비율
   
2. **Mean L2 Error**
   - 모든 샘플의 L2 거리 평균
   
3. **Median L2 Error**
   - L2 거리의 중앙값

### 6.2 K-Fold 집계 메트릭

모든 fold 결과를 집계하여:

- **평균 (Mean)**: 각 메트릭의 K-fold 평균
- **표준편차 (Std)**: 성능 분산 측정

### 6.3 전체 사용자 집계

여러 사용자 평가 시 전체 평균 계산:

```python
overall_acc = np.mean([r['avg_accuracy'] for r in all_user_results])
overall_mean_l2 = np.mean([r['avg_mean_l2'] for r in all_user_results])
```

---

## 7. 실험 결과 예시

### 7.1 단일 Fold 결과

```
Fold 0 Results:
  Accuracy: 0.7556
  Mean L2: 78.34
  Median L2: 52.18
```

### 7.2 사용자별 K-Fold 요약

```
USER 1 K-FOLD SUMMARY
================================================================================
Evaluated 3/3 folds:
  Accuracy:   0.7234 ± 0.0156
  Mean L2:    89.45 ± 5.67
  Median L2:  67.23 ± 4.32
```

### 7.3 전체 사용자 요약

```
OVERALL SUMMARY ACROSS ALL USERS
================================================================================
Evaluated 10 users:
  Overall Accuracy:   0.6892
  Overall Mean L2:    102.34
  Overall Median L2:  78.56

Per-user breakdown:
  User 1: Acc=0.7234, Mean L2=89.45, Median L2=67.23
  User 2: Acc=0.6543, Mean L2=115.23, Median L2=89.12
  ...
```

---

## Appendix A: 함수별 상세 설명

### A.1 메인 함수

#### `finetune_personalized(user_id, fold_idx, log_file)`
- **목적**: 특정 사용자의 특정 fold에 대해 개인화 학습 및 평가 수행
- **입력**:
  - `user_id`: 대상 사용자 ID (1-10)
  - `fold_idx`: Hold-out할 fold 인덱스 (0 to K-1)
  - `log_file`: 로그 파일 경로
- **동작**:
  1. 사용자 데이터 로드 및 K-fold 분할
  2. Global Full 모델 로드
  3. Vision encoder/aligner 동결
  4. Trainer로 학습 수행
  5. 모델 저장
  6. Hold-out fold로 평가
- **출력**: `(checkpoint_path, fold_result)` 튜플

---

### A.2 데이터셋 클래스

#### `PersonalizedDataset`
- **목적**: 개인화 학습을 위한 PyTorch Dataset
- **초기화**: 샘플 리스트를 받아 저장
- **`__getitem__`**: Qwen3-VL 형식으로 메시지 반환 (system → user 역할 변환)

#### `FoldEvalDataset`
- **목적**: 특정 fold의 테스트셋을 위한 평가 데이터셋
- **초기화**: 샘플 로드, Ground Truth 추출, Processor 초기화
- **`__getitem__`**: 전처리된 입력 + Ground Truth 반환

---

### A.3 K-Fold 유틸리티 (`utils/kfold.py`)

#### `create_k_folds(samples, k, seed)`
- **목적**: 샘플을 K개의 fold로 분할
- **핵심**: 고정 시드로 재현 가능한 분할 보장

#### `get_training_folds(folds, test_fold_idx)`
- **목적**: 테스트 fold를 제외한 학습 샘플 반환

#### `get_test_fold(folds, test_fold_idx)`
- **목적**: 특정 fold의 테스트 샘플 반환

---

### A.4 평가 함수

#### `eval_fold(user_id, fold_idx, checkpoint_path, folds)`
- **목적**: 학습된 모델로 해당 fold의 테스트셋 평가
- **동작**:
  1. 테스트 fold 샘플로 FoldEvalDataset 생성
  2. `eval_user()` 호출하여 평가 수행
  3. 결과 저장
  4. 메모리 정리
- **출력**: 평가 결과 딕셔너리

---

### A.5 설정 상수 (Personalized)

| 상수 | 값 | 설명 |
|------|-----|------|
| `K_FOLDS` | 3 | K-Fold의 K 값 |
| `PERSON_EPOCHS` | 3 | 개인화 학습 에폭 |
| `PERSON_BATCH_SIZE` | 1 | 배치 크기 |
| `PERSON_GRADIENT_ACCUMULATION_STEPS` | 4 | 그래디언트 누적 |
| `PERSON_LR` | 2e-5 | 학습률 |
| `PERSON_WARMUP_STEPS` | 5 | 워밍업 스텝 |
| `PERSON_WEIGHT_DECAY` | 0.01 | 가중치 감쇠 |
| `PERSON_LORA_RANK` | 8 | LoRA rank |
| `PERSON_LORA_ALPHA` | 32 | LoRA alpha |

---

## Appendix B: 실행 명령어 예시

```bash
# 전체 테스트 사용자 (1-10), 모든 fold, LoRA 방식
python 4_finetune_person.py 0 --lora

# 특정 사용자만 학습
python 4_finetune_person.py 0 --users 1 2 3 --lora

# 특정 fold만 학습
python 4_finetune_person.py 0 --users 1 --fold 0 --lora

# Full Fine-tuning 방식
python 4_finetune_person.py 0,1 --users 1

# 멀티 GPU 사용
python 4_finetune_person.py 0,1,2 --lora
```

---

## Appendix C: 디렉토리 구조

```
/workspace/PIGG_checkpoints/
├── global_agent_full/           # Global Full 체크포인트 (Base)
├── personalized_lora/           # 개인화 LoRA 체크포인트
│   ├── user_1/
│   │   ├── fold_0/
│   │   ├── fold_1/
│   │   └── fold_2/
│   ├── user_2/
│   │   └── ...
│   └── ...
└── personalized_full/           # 개인화 Full 체크포인트
    └── ...

./eval_results/personalized/
├── user_1/
│   ├── fold_0/
│   │   ├── out.json
│   │   └── results.txt
│   ├── fold_1/
│   │   └── ...
│   └── fold_2/
│       └── ...
├── user_1_summary.json          # 사용자 1의 K-fold 평균 결과
├── user_2/
│   └── ...
└── ...
```

---

*문서 작성일: 2025년 12월 3일*
*프로젝트: PIGG (Personalized Interactive GUI Grounding)*
