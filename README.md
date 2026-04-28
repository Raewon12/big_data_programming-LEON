# 법률 판례 텍스트 기반 사건 대분류 분류 모델 성능 비교

2026-1학기 빅데이터프로그래밍 | 팀 레옹

## 프로젝트 소개

법원 판례 텍스트를 활용하여 사건 유형을 자동으로 분류하는 모델을 구축하고, 서로 다른 텍스트 표현 방식 간 성능 차이를 비교 분석하는 프로젝트입니다.

## 데이터

- **출처**: AI Hub - 상황에 따른 판례 데이터
- **원본 규모**: 약 66,511건 (Training 53,209건 + Validation 6,651건)
- **분류 대상**: 민사, 가사, 형사, 행정, 기업, 근로자, 특허/저작권, 금융조세, 개인정보/ICT 등 10개 카테고리
- **입력 텍스트**: 판시사항 / 요약문 / 두 텍스트 결합 (3가지 조건 실험)

---

## 데이터 전처리

### 1. JSON → CSV 변환
- `json_train`, `json_val` 폴더의 JSON 59,860건을 반복문으로 읽어 `jdgmn`, `summ_pass`, `class_name` 세 필드만 추출
- 파일 구조가 불완전한 경우 에러 처리

### 2. 결측치 제거
- `jdgmn`, `summ_pass`, `class_name` 중 하나라도 비어있는 행 제거
- 59,860건 → **42,601건** (17,259건 제거)

### 3. 텍스트 정제
- 반복 줄바꿈 → 공백 변환
- 연속 공백 → 공백 하나로 정리
- 앞뒤 공백 제거

### 4. 라벨 처리
- 10개 카테고리 모두 존재 확인
- 문자열 카테고리(민사, 가사 등) → 숫자(0~9)로 인코딩
- 매핑 표 `label_mapping.csv`로 저장

### 5. 데이터 분할
- train 80% / val 10% / test 10%
- `stratify` 옵션으로 각 세트에서 카테고리 비율 동일하게 유지

### 처리 결과
| 단계 | 건수 |
|------|------|
| 원본 JSON | 59,860 |
| 결측치 제거 후 | 42,601 |
| Train | 34,080 |
| Val | 4,260 |
| Test | 4,261 |

### 카테고리별 분포 (Train)
| 카테고리 | 건수 | 비율 |
|---------|-----:|-----:|
| 민사 | 12,195 | 35.8% |
| 형사B(일반형) | 4,907 | 14.4% |
| 행정 | 4,713 | 13.8% |
| 형사A(생활형) | 4,238 | 12.4% |
| 금융조세 | 2,164 | 6.4% |
| 특허/저작권 | 2,136 | 6.3% |
| 근로자 | 1,368 | 4.0% |
| 기업 | 1,110 | 3.3% |
| 가사 | 1,030 | 3.0% |
| 개인정보/ICT | 219 | 0.6% |

> ⚠️ **클래스 불균형 확인**: 민사(35.8%) ↔ 개인정보/ICT(0.6%) → 약 56배 차이. 모델 학습 시 `class_weight='balanced'` 적용

---

## EDA 분석 결과

> 위치: [`EDA/EDA.ipynb`](EDA/EDA.ipynb)

### 분석 항목
1. **클래스 분포** ([class_distribution.png](EDA/class_distribution.png))
   - 민사가 압도적 다수, 개인정보/ICT가 최소
2. **텍스트 길이 분포** ([text_length_distribution.png](EDA/text_length_distribution.png))
   - 판시사항이 요약문보다 평균적으로 길음
   - **BERT 512 토큰 제한 초과 케이스 존재** → 향후 입력 처리 전략 필요
3. **카테고리별 텍스트 길이** ([length_by_class.png](EDA/length_by_class.png))
   - 카테고리별로 판시사항 길이 편차 확인
4. **카테고리 간 유사도 히트맵** ([category_similarity_heatmap.png](EDA/category_similarity_heatmap.png))
   - TF-IDF 코사인 유사도 기반 → **형사A ↔ 형사B 등 유사 카테고리 식별**
5. **카테고리별 대표 명사 Top 10** ([class_keywords.png](EDA/class_keywords.png))
   - KoNLPy(Okt)로 명사 추출, 불용어 제거
   - 도메인 특화 키워드 확인 (예: 특허/저작권 → 도메인 용어 변별력 ↑)

### 주요 인사이트
- **클래스 불균형 심각** → `class_weight='balanced'` 또는 샘플링 전략 필요
- **BERT 토큰 한계 이슈** → 판시사항 일부 truncation 또는 분할 처리 검토
- **유사 카테고리 존재** → 형사A/B, 기업/민사 등 변별력 확보 어려움 예상

---

## Baseline 모델 결과

> 위치: [`baseline/baseline.ipynb`](baseline/baseline.ipynb)

### 실험 설계
- **입력 3종 × 모델 2종 = 6개 실험**
- 입력: 판시사항 / 요약문 / 결합
- 모델: TF-IDF + Logistic Regression / TF-IDF + LinearSVC
- 공통: `class_weight='balanced'`, `max_features=10000`

### 1차 실험 결과 (F1-macro 기준)

| 모델 | 판시사항 | 요약문 | 결합 |
|------|--------:|-------:|-----:|
| TF-IDF + LR | 0.5893 | 0.5207 | **0.6237** |
| TF-IDF + SVM | 0.5783 | 0.5061 | 0.5971 |

→ **결합 입력**이 모든 모델에서 우위 → 가설 H3(결합 > 단일) **부분 검증**

### GridSearchCV 튜닝 결과 (결합 입력 기준)

| 모델 | Best params | Accuracy | F1-macro |
|------|------------|---------:|---------:|
| TF-IDF + LR (tuned) | C=1, ngram=(1,1) | 0.689 | 0.6253 |
| **TF-IDF + SVM (tuned)** | **C=0.1, ngram=(1,1)** | **0.724** | **0.6347** |

→ 베이스라인 최종 성능: **Accuracy 72.4%, F1-macro 0.635 (TF-IDF + SVM)**

### Confusion Matrix 분석 ([confusion_matrix_baseline.png](baseline/confusion_matrix_baseline.png))

**잘 분류되는 카테고리** ✅
- 특허/저작권: **0.96** (도메인 용어 변별력 압도적)
- 금융조세: 0.77, 가사: 0.73, 근로자: 0.73

**취약한 카테고리** ⚠️
- **기업: 0.46** → 민사로 0.37 오분류 (유사 도메인)
- **형사A ↔ 형사B**: 상호 0.23~0.26 혼동 (같은 형사 도메인)
- **개인정보/ICT: 0.52** (샘플 부족 + 다양한 클래스로 분산)

→ **BERT 문맥 모델로 개선 여지 존재** (다음 단계 동기 확보)

---

## 모델 구성 (전체 계획)

| 모델 | 설명 | 상태 |
|------|------|:---:|
| TF-IDF + Logistic Regression | Baseline | ✅ 완료 |
| TF-IDF + SVM (LinearSVC) | Baseline | ✅ 완료 |
| BERT 임베딩 + MLP | 문맥 기반 분류 | 🔜 진행 예정 |
| TF-IDF + BERT 하이브리드 | 단어 빈도 + 문맥 결합 | 🔜 진행 예정 |

## 평가 지표

- Accuracy, **Macro F1-score (주요)**, Precision, Recall
- 클래스별 Confusion Matrix 분석

## 기술 스택

- Python, PyTorch, Scikit-learn
- BERT (사전학습 가중치 frozen, 임베딩 추출용)
- KoNLPy (Okt) - 한국어 명사 추출
- Google Colab / RunPod (GPU)

## 폴더 구조
```
.
├── EDA/                     # EDA 분석 노트북 + 시각화
├── baseline/                # Baseline 모델 + Confusion Matrix
├── processed_data/          # 전처리 완료 데이터 (train/val/test/label_mapping)
├── 산출물/                  # 수행계획서, 발표자료
├── troubleshooting.md       # 트러블슈팅 기록
└── README.md
```

## 팀원

| 이름 | 역할 |
|------|------|
| 정래원 | 결과 분석 및 보고서 |
| 김동현 | Baseline 모델 (TF-IDF + LR/SVM) |
| 김홍근 | BERT + MLP / 하이브리드 모델 |
| 이승준 | 데이터 전처리 |
| 이윤수 | EDA 및 시각화 |
