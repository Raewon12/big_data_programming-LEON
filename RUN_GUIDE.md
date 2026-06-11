# RunPod 실행 가이드 (BERT+MLP / TextCNN / Late Fusion)

담당: 김홍근 | **브랜치: `main`** (최종 실험 결과 반영)

## 0. 노트북 구성

```
preprocess/make_processed_data.py  # JSON → CSV (원본 데이터 별도 다운로드 필요)
textcnn/textcnn.ipynb              # 딥러닝 비교군 (fastText 임베딩)
bert/extract_embeddings.ipynb      # KLUE-BERT frozen → 임베딩 npy 추출 (가장 먼저)
bert/bert_mlp.ipynb                # 임베딩 → MLP 학습/평가
hybrid/late_fusion.ipynb           # TF-IDF+LR 확률 + BERT+MLP 확률 가중합
```

공통: 입력은 결합(`jdgmn + ' ' + summ_pass`), 주 지표 F1-macro, 베이스라인 0.6347 돌파 목표.

## 1. RunPod Pod 접속 후 셋업

Jupyter Lab → Terminal:

```bash
cd /workspace
git clone https://github.com/Raewon12/big_data_programming-LEON.git
cd big_data_programming-LEON
pip install -r requirements-runpod.txt
```

> **의존성 안내**: 본 프로젝트 본 실험은 **RunPod PyTorch 템플릿**에서 수행하였다. `torch`는 템플릿에 사전 설치되어 있어 `requirements-runpod.txt`에 포함하지 않았다. 로컬 CPU 환경은 `requirements-cpu.txt`를 사용한다.

### fastText 한국어 임베딩 다운로드 (TextCNN용, 약 1.3GB)

```bash
mkdir -p artifacts
cd artifacts
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz
gunzip cc.ko.300.vec.gz
cd ..
```

### 데이터 확인 (저장소에 포함된 전처리 CSV)

```bash
python -c "import pandas as pd; print([(s, len(pd.read_csv(f'processed_data/{s}.csv'))) for s in ['train','val','test']])"
# [('train', 34080), ('val', 4260), ('test', 4261)] 이어야 함
```

### (선택) 원본 JSON에서 전처리 재생성

AI Hub에서 「상황에 따른 판례」 라벨링 JSON을 다운로드한 뒤:

```bash
python preprocess/make_processed_data.py --json-dir /path/to/라벨링데이터 --seed 42
```

## 2. 스모크 테스트 (선택, 5분)

각 노트북 상단 `SMOKE_TEST = True`로 바꿔 약 1,000건만 end-to-end 확인 → 에러 없으면 `False`로 본 실험.

## 3. 실험 순서 (순서 지킬 것)

1. `bert/extract_embeddings.ipynb` 전체 실행 → `artifacts/bert_*_X.npy`, `bert_*_y.npy` 생성
2. `bert/bert_mlp.ipynb` 전체 실행 → test F1-macro 측정 + `mlp_*_proba.npy` 저장 (**test 4,261건** 확인)
3. `textcnn/textcnn.ipynb` 전체 실행 → test F1-macro 측정 (fastText 파일 필요)
4. `hybrid/late_fusion.ipynb` 전체 실행 → best alpha + Fusion test F1-macro

> **주의**: `mlp_test_proba.npy`가 스모크(927건) 버전이면 Late Fusion에서 shape 오류 발생. `bert_mlp.ipynb`를 `SMOKE_TEST=False`로 재실행할 것.

## 4. 재현성 (seed)

| 항목 | 설정 |
|---|---|
| train/val/test 분할 | `train_test_split(..., stratify=label, random_state=42)` |
| 베이스라인 GridSearchCV | `random_state=42` |
| 스모크 샘플링 | `random_state=42` |
| PyTorch / NumPy | 노트북별 `torch.manual_seed(42)`, `np.random.seed(42)` 권장 |

## 5. 결과 정리 / 백업 (Pod 종료 전 필수)

- `artifacts/`의 대용량 npy/pt는 `.gitignore` 처리됨 → 수치·그래프는 별도 백업
- 공식 PNG(`artifacts/confusion_matrix_bert_mlp.png`, `artifacts/late_fusion_alpha.png`)는 저장소에 포함

```bash
git add textcnn bert hybrid preprocess requirements*.txt *.md
git commit -m "docs: experiment notes"
git push origin main
```

## 6. 산출물 (산출물/ 폴더)

- `모델_구성_v2.md` — 교수 피드백 반영 설계
- `최종보고서_레옹.md` — 양식 결과보고서
- `제출전_체크리스트.md` — 제출 전 팀 확인용
