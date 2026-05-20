# KLUE-RoBERTa Distill — 댓글 5-class 분류기

유튜브 테크리뷰 댓글을 `klue/roberta-base` + GPT-4.1 teacher 라벨 6,375건으로 distill한 5-class 분류기.
운영에서는 **local 1차 → 저신뢰만 GPT-4.1 API fallback** 의 cascade router 로 호출비 절감.

GPU는 **NVIDIA A40 (48GB, Ampere)** 기준 hyperparameter 셋업 (bf16 + TF32 + batch 64).

---

## 한 번에 복붙해서 돌리기 (A40 서버)

```bash
# 1) clone
git clone https://github.com/MeDeoDuck/KLUE_BERT_DISTILL.git
cd KLUE_BERT_DISTILL

# 2) 가상환경 (선택)
python -m venv .venv && source .venv/bin/activate

# 3) torch (CUDA 12.x). 다른 CUDA 버전이면 인덱스 URL만 조정
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

# 4) 전 파이프라인 한 방 실행: 데이터 정제 → 학습 → 평가
bash run_all.sh
```

산출물은 모두 `local_classifier/artifacts/` 아래:

| 경로 | 내용 |
|---|---|
| `data/train.jsonl` / `val.jsonl` / `test.jsonl` | video_id 그룹 split 결과 |
| `data/class_weights.json` | train 빈도 역수 |
| `model/best/` | best val_acc 체크포인트 (HF format) |
| `logs/train_metrics.json` | epoch별 loss/acc |
| `logs/test_summary.json` · `test_predictions.jsonl` | 평가 결과 |

---

## 단계별로 돌리고 싶을 때

```bash
python -m local_classifier.prepare_dataset   # 1. JSONL → 정제 → split
python -m local_classifier.train             # 2. 학습 (5 epoch, ~몇 분 on A40)
python -m local_classifier.evaluate          # 3. test 셋 평가
```

---

## 데이터셋

`comment_labels/labeled_gpt41_azure.jsonl` — GPT-4.1 teacher 라벨 6,375건 (Azure PG `agent_decisions` export, 2026-05-20).

| label | n | % |
|---|---:|---:|
| PRODUCT_OPINION | 3,824 | 60% |
| VIDEO_REACTION  | 1,087 | 17% |
| QUESTION        |   814 | 13% |
| CHATTER         |   405 |  6% |
| OFF_TOPIC       |   245 |  4% |

전처리 (`local_classifier/prepare_dataset.py`):
- `teacher_model == "gpt-4.1-mini"` 제외 (119건)
- `confidence < 0.85` 제외
- PII scrub (이메일/URL/멘션/전화 → 마스크)
- NFKC 정규화 + `ㅋㅋㅋㅋㅋ` 3자 압축
- alnum 정규화 후 md5 near-dup dedup
- **video_id 단위** train 80 / val 10 / test 10 그룹 split

---

## 학습 설정 (A40 default)

| 항목 | 값 |
|---|---|
| 모델 | `klue/roberta-base` (110M) |
| Seq len | 128 |
| Batch (train / eval) | 64 / 128 |
| Optimizer | AdamW lr=2e-5, weight_decay=0.01 |
| Schedule | linear warmup 10%, 5 epoch |
| Loss | CE × class_weight × per-example confidence, label_smoothing=0.05 |
| Mixed precision | bf16 autocast + TF32 matmul |
| Grad clip | 1.0 |

`local_classifier/config.py` 에서 조정. `klue/roberta-large` 도 A40에 충분히 들어가 — `BASE_MODEL` 만 바꾸면 됨.

---

## 운영 통합 (Cascade Router)

기존 `comment_filtering_agent` 의 GPT-4.1 분류기와 호환되도록 `ClassificationResult` 인터페이스 1:1 매칭.

```python
from local_classifier.classifier import LocalRobertaClassifier
from local_classifier.router import CascadeRouter

local = LocalRobertaClassifier(use_gpu=True)
# api = OptimizedBatchClassifier(...)  # 기존 GPT-4.1 batch classifier
router = CascadeRouter(local=local, api=api, tau_high=0.85, tau_low=0.55)

results = router.classify_batch(comments)
print(router.get_stats())  # local_rate, api_rate, low_conf_rate
```

Shadow 모드 (운영 영향 없이 disagreement 수집):

```python
from local_classifier.shadow import ShadowLogger
shadow = ShadowLogger(api_classifier=api, local_classifier=local)
results = shadow.classify_batch(comments)  # API 결과 반환, local은 비교만 로깅
```

---

## 폴더 구조

```
KLUE_BERT_DISTILL/
├── README.md                 # ← 지금 보고 있는 파일
├── run_all.sh                # prepare → train → evaluate 원샷
├── requirements.txt
├── comment_labels/
│   ├── README.md             # 데이터 출처·스키마
│   └── labeled_gpt41_azure.jsonl
└── local_classifier/
    ├── MODULE.md             # 모듈 상세 (튜닝 가이드 등)
    ├── config.py             # 모든 하이퍼파라미터
    ├── preprocess.py         # PII scrub / lang detect / dedup
    ├── prepare_dataset.py    # JSONL → split
    ├── dataset.py            # torch Dataset
    ├── train.py              # fine-tune entry
    ├── evaluate.py           # test 평가
    ├── classifier.py         # LocalRobertaClassifier
    ├── router.py             # CascadeRouter
    └── shadow.py             # ShadowLogger
```

---

## KPI 목표

- macro F1 ≥ 0.85 (GPT-4.1 대비 −3%p 이내)
- 라우터 운영 시 API 호출률 ≤ 20%
- local-API disagreement ≤ 5%

---

## 라이선스 / 주의

- 데이터셋은 유튜브에 공개된 댓글에서 수집되었으나, **상업 이용 전 YouTube API ToS / GPT-4.1 teacher 라벨에 대한 OpenAI 정책 확인 필요**.
- 사용자 식별 가능 정보(@멘션·전화 등)는 `preprocess.scrub_pii` 가 학습 직전 마스킹하지만, **원본 JSONL 에는 그대로 남아있음**. 외부 공유 시 추가 scrub 권장.
