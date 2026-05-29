# RunPod 실행 가이드 (BERT+MLP / TextCNN / Late Fusion)

담당: 김홍근 | 브랜치: `feature/bert-mlp`

## 0. 노트북 구성

```
textcnn/textcnn.ipynb          # 딥러닝 비교군 (fastText 임베딩)
bert/extract_embeddings.ipynb  # KLUE-BERT frozen -> 임베딩 npy 추출 (가장 먼저)
bert/bert_mlp.ipynb            # 임베딩 -> MLP 학습/평가
hybrid/late_fusion.ipynb       # TF-IDF+LR 확률 + BERT+MLP 확률 가중합
```

공통: 입력은 결합(`jdgmn + ' ' + summ_pass`), 주 지표 F1-macro, 베이스라인 0.6347 돌파 목표.

## 1. RunPod Pod 접속 후 셋업

Jupyter Lab -> Terminal 에서:

```bash
cd /workspace
git clone -b feature/bert-mlp https://github.com/Raewon12/big_data_programming-LEON.git
cd big_data_programming-LEON
pip install -r requirements.txt
```

### fastText 한국어 임베딩 다운로드 (TextCNN용, 약 1.3GB)

```bash
mkdir -p artifacts
cd artifacts
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ko.300.vec.gz
gunzip cc.ko.300.vec.gz
cd ..
```

데이터 확인:

```bash
python -c "import pandas as pd; print([ (s, len(pd.read_csv(f'processed_data/{s}.csv'))) for s in ['train','val','test'] ])"
# [('train', 34080), ('val', 4260), ('test', 4261)] 이어야 함
```

## 2. 스모크 테스트 (선택, 5분)

각 노트북 상단 `SMOKE_TEST = True`로 바꿔 1000건만 빠르게 end-to-end 확인 -> 에러 없으면 다시 `False`.

## 3. 실험 순서 (순서 지킬 것)

1. `bert/extract_embeddings.ipynb` 전체 실행 -> `artifacts/bert_*_X.npy`, `bert_*_y.npy` 생성
2. `bert/bert_mlp.ipynb` 전체 실행 -> test F1-macro 측정 + `mlp_*_proba.npy` 저장
3. `textcnn/textcnn.ipynb` 전체 실행 -> test F1-macro 측정 (fastText 파일 필요)
4. `hybrid/late_fusion.ipynb` 전체 실행 -> best alpha + Fusion test F1-macro

## 4. 결과 정리 / 백업 (Pod 종료 전 필수)

- `artifacts/`의 png, npy 는 `.gitignore` 처리됨 -> 결과 수치/그래프는 따로 백업
- 노트북(실행 결과 포함)과 phase 보고서는 커밋:

```bash
git add textcnn bert hybrid 산출물 *.md
git commit -m "experiment results: bert+mlp / textcnn / late fusion"
git push origin feature/bert-mlp
```

- npy 임베딩은 Network volume(`/workspace/artifacts`)에 남으므로 Pod 재시작해도 유지

## 5. 산출물 (산출물/ 폴더에 작성)

- `phase1_환경및스모크.md`
- `phase3_1_textcnn.md`, `phase3_2_bert_embedding.md`, `phase3_3_bert_mlp.md`, `phase3_4_late_fusion.md`
- `phase4_최종통합.md` (5모델 비교표 + H1/H2/H3 결론)
