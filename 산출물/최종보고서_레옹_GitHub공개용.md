# 빅데이터프로그래밍 산학협력프로젝트 결과보고서

**2026학년도 1학기 · 한성대학교 컴퓨터공학부**

> **GitHub 공개용** — 학번·전화번호·이메일 마스킹. e-class 제출 원본은 `최종보고서_레옹.md`(비공개 관리 권장).

---

## 과제 수행원 현황

| 항목 | 내용 |
|---|---|
| 수행 학기 | 2026학년도 1학기 |
| 프로젝트명 | 판례 텍스트 기반 사건 유형 자동 분류 — 텍스트 표현 방식별 성능 비교 연구 |
| 팀명 | 레옹 |

| 구분 | 이름 | 학번 | 소속 | 전화번호 | 이메일 | 트랙1 | 트랙2 |
|---|---|---|---|---|---|---|---|
| 팀장 | 정래원 | 207**** | 한성대학교 | (비공개) | (비공개) | 빅데이터트랙 | 모바일소프트웨어트랙 |
| 팀원 | 김동현 | 237**** | 한성대학교 | (비공개) | (비공개) | 모바일소프트웨어트랙 | 빅데이터트랙 |
| 팀원 | 김홍근 | 219**** | 한성대학교 | (비공개) | (비공개) | 웹공학트랙 | 빅데이터트랙 |
| 팀원 | 이승준 | 197**** | 한성대학교 | (비공개) | (비공개) | 빅데이터트랙 | 웹공학트랙 |
| 팀원 | 이윤수 | 217**** | 한성대학교 | (비공개) | (비공개) | 모바일소프트웨어트랙 | 빅데이터트랙 |

| 항목 | 내용 |
|---|---|
| 교과목명 | 빅데이터프로그래밍 |
| 과목코드 | V022005 |
| 지도교수 성명 | 이청용 |
| 지도교수 소속 | 컴퓨터공학부 |

| 항목 | 내용 |
|---|---|
| 산업체 멘토 기업명 | 왓챠 (수행계획서 기준) |
| 멘토 성명 | 임해빈 |
| 멘토 직위 | 매니저 (ML팀) |

---

# 프로젝트 결과보고서

**프로젝트명**  
판례 텍스트 기반 사건 유형 자동 분류 — 텍스트 표현 방식별 성능 비교 연구

---

## 1. 개발동기 및 목적

### Key Words

판례 텍스트 분류 · TF-IDF · TextCNN · KLUE-BERT · Late Fusion · Macro F1 · 클래스 불균형 · AI Hub · 법률 NLP

### 요약 (Abstract)

법원 판례 텍스트를 10개 사건 유형으로 분류하는 문제에서, 본 연구는 **동일 데이터·동일 결합 입력·동일 지표(Macro F1)** 조건 아래 희소 빈도(TF-IDF), 학습 임베딩(TextCNN), 사전학습 문맥(KLUE-BERT frozen) 표현 방식을 비교하였다. AI Hub 「상황에 따른 판례」 42,601건을 전처리·stratify 분할한 뒤 5개 모델을 평가한 결과, **Late Fusion(0.6561)** 이 수치상 최고이나 **TextCNN(0.6556)** 과 차이는 0.0005로 실질 동률이었다. frozen BERT+MLP 단독(0.5792)은 baseline(0.6347)에 미달하였으며, 오분류는 모델과 무관하게 클래스 의미 인접성·불균형에서 비롯되었다. 본 연구는 SOTA 단일 모델 제고가 아니라 **표현 방식별 실증 비교**와 재현 가능한 실험 파이프라인 구축을 목적으로 한다.

### 개발동기 및 목적

최근 법률 분야에서도 인공지능을 활용한 자동화 연구가 증가하고 있다. 판례 문서는 분량이 길고 구조가 복잡해 사건 유형을 파악·분류하려면 시간과 전문 지식이 많이 든다. 판례는 계속 축적되고 있어 수작업 분류만으로는 한계가 있으며, 자연어 처리 기술의 발전으로 텍스트 표현 방식(단어 빈도 vs 사전학습 문맥)에 따른 성능 차이를 실증적으로 비교할 필요가 있다.

법률 정보는 매년 방대한 양으로 축적되지만, 판례를 사건 유형별로 분류·검색하는 작업은 여전히 전문 인력의 수작업에 의존하는 경우가 많다. 판례 텍스트를 사건 유형(민사·형사·행정 등)으로 자동 분류할 수 있다면, 법률 검색 시스템·판례 추천·법률 상담 챗봇 등 다양한 응용의 **1차 라우팅(primary routing)** 기반 기술이 된다 [5][9].

그러나 판례 텍스트는 (1) 도메인 특화 법률 용어가 많고, (2) 사건 유형 간 의미 경계가 모호하며, (3) 유형별 데이터 양이 크게 불균형하다는 점에서 일반 텍스트 분류보다 까다롭다. 선행 연구로 Vatsal et al.[13]은 미국 대법원 판례에 BERT 계열 모델을 적용하였고, Lee[14]는 한국어 성범죄 판례 분류에서 전통 ML과 BERT 계열을 비교하였다. Chalkidis et al.[6]의 Legal-BERT 등 영어권 법률 도메인 연구도 활발하나, **한국어 판례 10개 사건 대분류**를 대상으로 TF-IDF, 학습 임베딩, 사전학습 문맥 표현을 **동일 조건에서 비교**한 연구는 상대적으로 제한적이다. 동일 AI Hub 「상황에 따른 판례」 [5] 데이터를 활용한 공식 예시도 판결 결과 3분류 등으로, 본 프로젝트의 10-클래스 과제 정의와는 구분된다.

본 연구는 위 공백을 메우기 위해, AI Hub 판례 데이터를 활용하여 **희소 빈도(TF-IDF) · 학습 임베딩(TextCNN) · 사전학습 문맥(KLUE-BERT)** 세 가지 텍스트 표현 방식의 분류 성능을 **동일 조건에서** 비교하고, 어휘 신호와 문맥 신호를 결정 단계에서 결합한 **Late Fusion 앙상블** [10]의 효과를 실증적으로 검증하는 것을 목적으로 한다. 본 과제는 “최신 SOTA 모델 하나의 성능 제고”가 아니라, **표현 방식·결합 방식이 분류 성능에 미치는 영향**을 규명하는 비교 연구이다.

### 개발목표

1. AI Hub 「상황에 따른 판례」 데이터를 전처리하여 10개 사건 유형 분류용 학습·검증·평가 데이터셋을 구축한다.
2. TF-IDF+LR/SVM, TextCNN, KLUE-BERT(frozen)+MLP, Late Fusion 등 **5개 모델**을 동일 결합 입력·동일 지표(Macro F1)로 비교한다.
3. 판시사항·요약문 **결합 입력**이 단일 입력보다 우수한지 검증한다 (H2).
4. 클래스 불균형 환경에서 Macro F1 [11]을 주지표로 사용하는 평가 체계를 수립한다.
5. 전 실험 과정을 Jupyter Notebook과 GitHub에 공개하여 재현 가능한 연구 결과를 제출한다.

**핵심 요약 (§1)**
- 본 문제는 판례 텍스트를 **10개 사건 유형**으로 분류하는 **단일 라벨 다중 클래스** 문제이다.
- 클래스 불균형(최대 56배)이 심하므로 **Accuracy보다 Macro F1**을 주지표로 사용하였다.
- 목적은 SOTA 제고가 아니라 **동일 입력·동일 test(4,261건)·동일 지표**에서 **표현 방식 3계열 비교**이다.

---

## 2. 최종 결과물 소개

### 최종결과물 소개 및 증빙사진 (4장 이상 첨부 필수)

본 프로젝트의 최종 결과물은 **판례 텍스트 10-클래스 자동 분류 실험 파이프라인 전체**이다. 데이터 전처리부터 EDA, 베이스라인 모델, 딥러닝 모델, 앙상블 모델까지의 실험을 Jupyter Notebook으로 구현하였으며, 모든 코드와 결과를 GitHub에 공개하였다.

**GitHub 저장소**  
https://github.com/Raewon12/big_data_programming-LEON

**데이터 출처**  
AI Hub 「상황에 따른 판례」 — https://aihub.or.kr

**주요 산출물**

| 구분 | 경로 | 설명 |
|---|---|---|
| 전처리 데이터 | `processed_data/` | train 34,080 / val 4,260 / test 4,261 |
| EDA | `EDA/EDA.ipynb` + PNG 4종 | 클래스 분포, 유사도, 키워드 |
| 베이스라인 | `baseline/baseline.ipynb` | TF-IDF + LR/SVM |
| TextCNN | `textcnn/textcnn.ipynb` | fastText + 1D CNN |
| BERT+MLP | `bert/extract_embeddings.ipynb`, `bert/bert_mlp.ipynb` | KLUE-BERT frozen + MLP |
| Late Fusion | `hybrid/late_fusion.ipynb` | 확률 가중 앙상블 |
| 공식 결과 그래프 | `artifacts/*.png` | 혼동 행렬, α 탐색 그래프 |
| 종합 보고서 | `README.md`, `RUN_GUIDE.md` | 실험 재현 가이드 |

본 실험은 전처리 후 **42,601건 전체**를 stratify 8:1:1로 분할하였으며, 학습은 Train(34,080건), 하이퍼파라미터·α 탐색은 Validation(4,260건), 최종 성능 보고는 unseen Test(4,261건) 기준으로 수행하였다. 개발 초기에는 `SMOKE_TEST=True`로 약 1,000건 파이프라인 점검 후, `SMOKE_TEST=False`로 본 실험을 완료하였다.

**표 1. 전체 모델 성능 비교 (test 4,261건, Macro F1 주지표)**

| # | 모델 | 계열 | Accuracy | Macro F1 | 베이스라인 대비 |
|:--:|---|---|---:|---:|:--:|
| 1 | TF-IDF + LR (튜닝) | 전통 ML | 0.689 | 0.6253 | 기준 이하 |
| 2 | TF-IDF + SVM (튜닝) | 전통 ML | 0.724 | **0.6347** | **기준선** |
| 3 | TextCNN | 딥러닝(학습임베딩) | 0.7261 | 0.6556 | ▲ +0.021 |
| 4 | KLUE-BERT(frozen)+MLP | 딥러닝(문맥) | 0.6163 | 0.5792 | ▼ −0.056 |
| 5 | Late Fusion (α=0.8) | 앙상블 | 0.7073 | **0.6561** | ▲ 수치상 최고(실질 동률) |

실험 결과, **TextCNN(0.6556)** 과 **Late Fusion(0.6561)** 이 베이스라인 TF-IDF+SVM(0.6347)을 상회하였으나, **frozen BERT+MLP 단독(0.5792)** 은 baseline에 미달하였다. Late Fusion은 BERT+MLP 단독 대비 Macro F1 **+0.077** 개선을 보였으며, 최적 α=0.8에서 어휘 신호(LR 80%)가 주력·문맥 신호(BERT+MLP 20%)가 보조임을 확인하였다.

**[증빙사진 1]**

*그림 1. Train 데이터 10개 사건 유형별 클래스 분포 (민사 35.8% ↔ 개인정보/ICT 0.6%, 약 56배 불균형)*

---

**[증빙사진 2]**

*그림 2. TF-IDF 코사인 유사도 기반 카테고리 간 유사도 (형사A↔B, 기업↔민사 인접)*

---

**[증빙사진 3]**

*그림 3. RunPod GPU 환경 KLUE-BERT 임베딩 기반 MLP 학습 (early stopping, best val F1 0.6003)*

---

**[증빙사진 4]**

*그림 4. Late Fusion α vs Validation Macro F1 (최적 α=0.8, val F1 0.6669)*

---

**[증빙사진 5]**

*그림 5. BERT+MLP test 혼동 행렬 (4,261건, Macro F1 0.5792)*

---

**최종 모델 선정 관점 (test 4,261건)**

| 관점 | 선정 | Macro F1 | 비고 |
|---|---|---:|---|
| 수치상 최고 | Late Fusion (α=0.8) | **0.6561** | LR 80% + BERT+MLP 20% |
| 비용 대비 추천 | **TextCNN** | 0.6556 | Late Fusion과 **+0.0005** → 실질 동률 |
| 해석 가능 baseline | TF-IDF + SVM | 0.6347 | GridSearchCV 튜닝 후 기준선 |
| 후속 연구 대상 | BERT fine-tuning / Legal-BERT | 0.5792 (frozen 단독) | 도메인 적응·truncation 보완 |

수치상 최고는 Late Fusion이지만 TextCNN과 차이가 0.0005로 매우 작아 **실질적으로 동률**이다. 구현 복잡도·추론 파이프라인 단순성을 고려하면 **TextCNN이 비용 대비 가장 실용적인 모델**로 판단된다. frozen BERT+MLP 단독은 baseline을 넘지 못했으므로, 후속 연구에서는 fine-tuning·sliding window·법률 도메인 사전학습 모델 비교가 필요하다.

최종적으로 본 팀은 판례 텍스트를 입력받아 10개 사건 유형 중 하나를 예측하는 **텍스트 분류 실험 체계**를 완성하였다. 기업·형사A/B 등 의미 인접 클래스에서의 혼동은 모델 종류와 무관하게 지속되어, 데이터 본질적 한계에 대한 후속 연구가 필요하다.

---

## 3. 프로젝트 추진 내용

> 해당 항목은 추후 디렉토리 및 보도자료로 활용 예정이오니 상세하고 쉽게 작성 요망

### 3.1 프로젝트 진행과정 (주차별)

| 주차 | 진행 내용 |
|---|---|
| 3~5주차 | 프로젝트 주제 선정, AI Hub 데이터 탐색, 연구 질문·가설(H1~H3) 수립, Python·scikit-learn·PyTorch 기술 스터디 |
| 6주차 | JSON → CSV 전처리, 결측치 제거, 라벨 인코딩, stratify 8:1:1 분할 (담당: 이승준) |
| 7주차 | EDA 수행 — 클래스 분포, 텍스트 길이, 카테고리 유사도, 대표 키워드 분석 (담당: 이윤수) |
| 8주차 | 베이스라인 구축 — TF-IDF+LR/SVM, 입력 3종 비교, GridSearchCV 튜닝 (담당: 김동현) |
| 9주차 | TextCNN 구현 — fastText 임베딩 + 1D CNN 학습 및 평가 (담당: 김홍근) |
| 10주차 | KLUE-BERT frozen 임베딩 추출 → MLP 분류기 학습 (담당: 김홍근) |
| 11주차 | Late Fusion 구현 — α 그리드 탐색(val), test 최종 평가 (담당: 김홍근) |
| 8~10주차 | 모델 설계(4개→5개), 베이스라인 GridSearchCV 튜닝 완료 |
| 11~12주차 | **중간발표(2026.05.06)** — EDA·베이스라인 결과 공유, BERT·하이브리드 계획 발표 |
| 12~14주차 | 교수 피드백 4건 반영(TextCNN·Late Fusion), RunPod GPU 본 실험(5모델) |
| 15주차 | 최종 발표(동영상 제출), 오프라인 Q&A, 최종보고서 작성 (담당: 정래원) |

### 3.2 프로젝트 구현과정 — 개념설계

#### 3.2.1 문제 정의

본 연구가 푸는 문제는 다음과 같다.

- **입력**: 판례 한 건의 텍스트 — 판시사항(`jdgmn`) + 요약문(`summ_pass`) 결합
- **출력**: 10개 사건 유형 중 1개
- **과제 유형**: 단일 라벨 다중 클래스 분류(single-label multi-class classification)

**10개 분류 클래스**: 가사 · 개인정보/ICT · 근로자 · 금융조세 · 기업 · 민사 · 특허/저작권 · 행정 · 형사A(생활형) · 형사B(일반형)

#### 3.2.2 연구 가설

| 가설 | 내용 | 검증 방법 |
|---|---|---|
| **H1** | 텍스트 표현 방식(희소빈도 / 학습임베딩 / 사전학습문맥)에 따라 분류 성능에 차이가 있다 | 5개 모델 Macro F1 비교 |
| **H2** | 판시사항·요약문 결합 입력이 단일 입력보다 우수하다 | 베이스라인 6실험(입력 3종 × 모델 2종) |
| **H3** | 어휘(TF-IDF)와 문맥(BERT)을 결합한 Late Fusion이 단일 모델보다 우수하다 | α 탐색 및 test 평가 |

#### 3.2.3 실험 설계 개요

모든 모델은 **결합 입력**(`jdgmn + ' ' + summ_pass`)을 동일하게 사용하며, 평가 지표도 Macro F1으로 통일하였다. BERT는 fine-tuning 없이 **Frozen Feature Extractor** [1]로만 사용하여, 표현 방식 자체의 효과만 공정하게 측정하였다.

**표 2. 5개 비교 모델 개요**

| # | 모델 | 입력 | 표현 방식 | 학습 범위 | 선정 이유 | 한계 |
|:--:|---|---|---|---|---|---|
| 1 | TF-IDF + LR | 결합 | 희소·어휘 빈도 | 분류기만 | 확률 출력·Late Fusion 연동 | n-gram(1,1) 한계 |
| 2 | TF-IDF + SVM | 결합 | 희소·어휘 빈도 | 분류기만 | **튜닝 후 baseline** | 확률 불안정 |
| 3 | TextCNN | 결합 | fastText + CNN n-gram [2][3] | End-to-End | 짧은 법률 구문 패턴 | OOV·fastText 매칭 31.9% |
| 4 | BERT+MLP | 결합 | KLUE-BERT frozen [CLS] [1][4] | MLP만 | 문맥 표현 비교군 | frozen·512 truncation |
| 5 | Late Fusion | 결합 | LR 확률 ⊕ MLP 확률 [10] | α만 탐색 | 어휘+문맥 결합 | TextCNN과 실질 동률 |

구도: **전통 ML 2 / 딥러닝 2 / 앙상블 1** → 계열별 공정 비교

---

### 3.3 프로젝트 구현과정 — 상세설계

#### 3.3.1 데이터 전처리

**표 3. 전처리 파이프라인**

| 단계 | 처리 내용 | 결과 |
|---|---|---|
| ① JSON → CSV | `jdgmn`, `summ_pass`, `class_name` 3개 필드 추출 | 59,860건 |
| ② 결측치 제거 | 세 필드 중 하나라도 비면 제거 | **42,601건** |
| ③ 텍스트 정제 | 반복 줄바꿈·연속 공백 정리 | — |
| ④ 라벨 인코딩 | 문자열 → 정수 0~9, `label_mapping.csv` 저장 | 10개 클래스 |
| ⑤ 데이터 분할 | stratify 8:1:1, `random_state=42` | Train 34,080 / Val 4,260 / Test 4,261 |

**데이터 축소(59,860 → 42,601) 설명**

17,259건(약 28.8%)이 제거된 주된 원인은 `jdgmn`, `summ_pass`, `class_name` 중 **하나라도 결측·공백**인 레코드이다. AI Hub 원본 JSON 중 일부는 Summary·Class_info 필드 구조가 불완전하거나 빈 문자열을 포함한다. 본 팀은 분류에 필수인 세 필드가 모두 유효한 샘플만 사용하였다.

결측 제거 **전후 클래스 분포 비교**는 당시 전처리 노트북에서 수행하였으나, 제거 전 중간 CSV가 저장소에 남아 있지 않아 당시 **클래스별 제거 비율의 정량 재검증은 불가**하였다. 다만 최종 42,601건에서 10개 클래스가 모두 존재하고, stratify 분할 후 소수 클래스(개인정보/ICT)가 val·test에 포함되어 **특정 클래스 전면 소실은 관찰되지 않았다**.

다만 **모든 클래스가 유지되었다는 사실만으로 결측 제거 과정의 분포 왜곡이 없었다고 단정할 수는 없다**. 본 연구에서는 이를 한계로 명시하고, `preprocess/make_processed_data.py`에서 **제거 전후 클래스별 건수·제거율(`removal_stats_by_class.csv`)** 을 출력하도록 보완하였다. 향후에는 제거율이 높은 클래스에 대해 원본 필드 구조를 재검토하고, 결측 요약문을 판시사항 기반 요약으로 대체하는 방식도 검토할 수 있다.

**핵심 요약 (§3.3.1 전처리)**
- 59,860 → 42,601건(28.8% 제거)은 세 필드 결측·공백 때문이다.
- stratify + `random_state=42`로 클래스 비율을 유지하였다.
- case id 중복 제거는 미수행 → 데이터 누수 가능성은 한계로 명시한다.

**train/val/test stratify 사용 이유**

민사(35.8%)와 개인정보/ICT(0.6%) 간 약 56배 불균형이 존재한다. 무작위 분할 시 소수 클래스가 val·test에서 누락되거나 비율이 왜곡될 수 있으므로, `train_test_split(..., stratify=label, random_state=42)`로 **클래스 비율을 유지**하였다. 이는 Macro F1 주지표와 `class_weight='balanced'` 전략과 일관된다.

**데이터 누수 가능성 검토**

판례 데이터에는 동일 사건·유사 문서가 여러 건 존재할 수 있다. 본 전처리는 **case id 기준 중복 제거를 수행하지 않았다** (추출 필드에 사건 ID 미포함). 따라서 의미적으로 유사한 판례가 train과 test에 동시에 포함되었을 가능성은 **한계로 인정**하며, 향후 사건 ID·판례번호 기준 그룹 분할(group split) 검토가 필요하다.

**재현 코드**: `preprocess/make_processed_data.py` (JSON → CSV, stratify 분할, `label_mapping.csv` 생성)

#### 3.3.2 탐색적 데이터 분석 (EDA)

EDA 결과는 이후 모델 설계의 직접적 근거가 되었다.

**표 3-1. EDA 발견 → 모델 설계 결정**

| EDA 발견 | 모델·평가 설계 결정 |
|---|---|
| 클래스 불균형 56배 (민사 35.8% ↔ 개인정보/ICT 0.6%) | Macro F1 [11] 주지표, `class_weight='balanced'` |
| 형사A↔B·기업↔민사 TF-IDF 유사도 높음 (0.94~0.96) | 혼동 행렬 중심 분석, 오분류를 데이터 본질로 해석 |
| 특허/저작권 등 도메인 키워드 변별력 높음 | TF-IDF·TextCNN 어휘/n-gram 강점 예상 |
| 판시사항·요약문 길이·정보밀도 차이 | 결합 입력 채택 (H2), BERT 512 truncation 한계 → sliding window 향후 과제 |
| 긴 판시사항 (BERT 512 토큰 초과 다수) | frozen `[CLS]` 단일 벡터의 정보 손실 한계 인정 |

**발견 1 — 클래스 불균형** → §2 그림 1 참조

**발견 2 — 입력 정보밀도 차이** — 판시사항이 요약문보다 길고 정보밀도가 높아 결합 입력(H2) 채택

**발견 3 — 유사 카테고리 존재** → §2 그림 2 참조

**발견 4 — 도메인 키워드 변별력** — 특허/저작권·근로자 등 유형별 전용 용어 존재 → TF-IDF·TextCNN 강점 예상

#### 3.3.3 베이스라인 설계 및 H2 검증

TF-IDF + Logistic Regression / LinearSVM에 `class_weight='balanced'`, `max_features=10000`을 적용하고 GridSearchCV로 튜닝하였다. 입력 3종(판시사항 / 요약문 / 결합) × 모델 2종 = 6실험으로 H2를 검증하였다.

**표 4. 입력 조건별 베이스라인 성능 (Macro F1)**

| 모델 | 판시사항 | 요약문 | 결합 |
|---|---:|---:|---:|
| TF-IDF + LR | 0.5893 | 0.5207 | **0.6237** |
| TF-IDF + SVM | 0.5783 | 0.5061 | 0.5971 |

결합 입력이 모든 모델에서 단일 입력보다 우위 → **H2 지지**. 이후 모든 모델의 공통 입력으로 결합 텍스트를 채택하였다.

> **주의**: 표 4는 **입력 비교 단계**(튜닝 전) 결과이다. GridSearchCV 튜닝 후 최종 baseline은 아래와 같이 **별도**이다.

**튜닝 후 최종 베이스라인**: TF-IDF + SVM — `max_features=10000`, `ngram_range=(1,1)`, `class_weight='balanced'`, GridSearchCV(`C` 탐색) → Accuracy 0.724, Macro F1 **0.6347** (기준선).

*그림 6. 베이스라인 TF-IDF+SVM 혼동 행렬 (튜닝 후, test 4,261건)*

#### 3.3.4 TextCNN 설계

Kim[2]의 1D TextCNN을 따랐다. 판례는 “상표 등록”, “임금 지급”처럼 **짧은 법률 n-gram 구문**이 유형 구분에 유효하므로, kernel 3·4·5가 local 패턴을 포착하는 구조가 적합하다.

**표 3-2. TextCNN 하이퍼파라미터**

| 항목 | 설정 |
|---|---|
| 임베딩 | fastText `cc.ko.300.vec` [3], 300차원 |
| CNN | kernel {3,4,5} × 100 filters, max-pooling |
| 정규화 | Dropout 0.3 |
| 최적화 | Adam lr=1e-3, early stopping patience=6 |
| 어휘 | vocab 117,110, fastText 매칭 **31.9%** (법률 OOV 다수) |

#### 3.3.5 KLUE-BERT(frozen) + MLP 설계

2단계 파이프라인:

1. **임베딩 추출** (`bert/extract_embeddings.ipynb`): `klue/bert-base` [4], `max_len=512`, `batch=32`, `requires_grad=False`, `[CLS]` **768차원** → `.npy` 캐싱 (train 34,080건 약 154초)
2. **MLP 분류** (`bert/bert_mlp.ipynb`): 768→256→64→10, ReLU, Dropout(0.3), Adam(lr=1e-3), early stopping(patience=7), `class_weight='balanced'`

**표 3-3. BERT+MLP 하이퍼파라미터**

| 항목 | 설정 |
|---|---|
| 사전학습 | `klue/bert-base` (frozen) |
| 입력 | 결합 텍스트, 512 토큰 truncation |
| 특징 | `[CLS]` 768차원 (고정 길이 벡터) |
| 분류기 | 3층 MLP, dropout 0.3 |
| 평가 | test **4,261건** 일괄 (Macro F1 0.5792) |

Frozen 선정 이유: 본 연구 목적이 SOTA fine-tuning 경쟁이 아니라 **표현 방식 비교**이며, 계산 비용·재현성을 고려해 **frozen feature extractor로 제한**하였다. 이는 BERT를 의도적으로 약화한 설정이 아니라, **실험 범위와 공정 비교 조건을 명시한 설계 선택**이다.

#### 3.3.6 Late Fusion 설계

교수 피드백 ①③을 반영하여, Early Fusion(차원 직접 결합) 대신 **Late Fusion(확률 가중합)** [10]을 채택하였다.

```
P_final = α · P(TF-IDF+LR) + (1 − α) · P(BERT+MLP)
α ∈ {0.0, 0.1, ..., 1.0}  →  Validation에서 최적 α 탐색
최종 예측 = argmax(P_final)
```

- LR 사용 이유: baseline 최고는 SVM이나, SVM(LinearSVC)은 **확률 출력이 불안정** → Late Fusion에 **확률이 안정적인 TF-IDF+LR** 사용
- Late Fusion 이유: TF-IDF(10,000차원) ↔ BERT(768차원) 차원 격차로 Early Fusion 시 신호 묻힘 방지
- **α는 validation(4,260건)에서만 선택**, test(4,261건)는 **1회 최종 평가** (test 누수 방지)

---

### 3.4 프로젝트 구현과정 — 구현 및 실험

#### 3.4.1 실험 환경

| 항목 | 내용 |
|---|---|
| 플랫폼 | RunPod (GPU), Network Volume 50GB |
| GPU | NVIDIA RTX PRO 4500 (32GB VRAM) |
| 프레임워크 | PyTorch, scikit-learn, transformers 4.44.2, gensim |
| 사전학습 | `klue/bert-base` [4], fastText `cc.ko.300.vec` [3] |
| 재현성 seed | 분할·GridSearchCV·스모크 샘플 `random_state=42` |
| 의존성 | RunPod: `requirements-runpod.txt` (torch 템플릿 사전 설치) |

#### 3.4.2 노트북 실행 순서

```
extract_embeddings.ipynb → bert_mlp.ipynb → late_fusion.ipynb
                                                ↑
textcnn.ipynb (독립) ───────────────────────────┘
```

#### 3.4.3 모델별 활용 방식

**표 5. 모델별 재학습 범위 및 활용 방식**

| 모델 | 활용 방식 | 재학습 범위 | 근거 |
|---|---|---|---|
| TF-IDF + LR/SVM | 통계 기반 특징 추출 + 분류 | 분류기만 학습 | 어휘 빈도 기준선 확보 |
| TextCNN | fastText 초기화 → End-to-End | 임베딩·CNN 전체 학습 | 법률 도메인 n-gram 학습 |
| BERT + MLP | Frozen 임베딩 추출 | BERT 동결, MLP만 학습 | 표현력 자체 공정 비교 |
| Late Fusion | 두 모델 확률 앙상블 | 추가 학습 없음 (α만 탐색) | 차원 충돌 없이 어휘+문맥 결합 |

#### 3.4.4 실험 결과

**TextCNN** — Accuracy 0.7261 / Macro F1 **0.6556** (baseline +0.021, 상회)  
특허/저작권 F1 0.9307, 기업 F1 0.2771

**BERT+MLP** — Accuracy 0.6163 / Macro F1 **0.5792** (baseline −0.056, 미달, **test 4,261건**)  
val best 0.6003 → test 0.5792로 일반화 한계. 미달 요인: (1) frozen `[CLS]` 단일 벡터 한계, (2) 512 토큰 truncation 정보 손실, (3) 일반 한국어 BERT의 법률 도메인 특화 부족, (4) 경량 MLP 분류 헤드 한계

**Late Fusion** — Accuracy 0.7073 / Macro F1 **0.6561** (수치상 최고)  
BERT+MLP 단독 대비 +0.077, TextCNN과 **+0.0005** 차이로 **실질 동률** (통계적 유의성 검정은 미수행)

**표 6. Late Fusion α 탐색 결과 (Validation Macro F1)**

| α (LR 가중치) | 0.0 (MLP만) | 0.4 | 0.6 | **0.8 (최적)** | 1.0 (LR만) |
|---|---:|---:|---:|---:|---:|
| val Macro F1 | 0.6003 | 0.6434 | 0.6607 | **0.6669** | 0.6559 |

α=0.8에서 peak → **어휘 신호 80% 주력, BERT+MLP 20% 보조**.

#### 3.4.5 클래스별 강·약점 (모델 공통 패턴)

| 유형 | 클래스 | 공통 경향 |
|---|---|---|
| 쉬운 클래스 | 특허/저작권 | F1 0.93~0.96, 도메인 용어 변별력 압도적 |
| 쉬운 클래스 | 민사 | 다수 클래스 + 어휘 풍부 → 안정적 |
| 어려운 클래스 | 기업 | 민사로 흡수 오분류 (F1 0.25~0.31) |
| 어려운 클래스 | 형사A ↔ 형사B | 같은 형사 도메인, 상호 혼동 |
| 어려운 클래스 | 개인정보/ICT | 극소수(0.6%) → 학습 부족 |

오분류의 원인은 모델이 아니라 **클래스 의미 인접성·불균형**이라는 데이터 본질에서 비롯되었으며, EDA 유사도 히트맵에서 예고된 패턴과 일치한다.

#### 3.4.6 가설 검증 종합

| 가설 | 결과 | 근거 |
|---|---|---|
| H1 (표현 방식 차이) | **부분 지지** | TextCNN·Late Fusion 상회, BERT+MLP 단독 미달 |
| H2 (결합 입력 우수) | **지지** | 결합 입력이 모든 베이스라인 실험에서 우위 |
| H3 (Late Fusion 효과) | **부분 지지** | BERT+MLP 대비 +0.077, TextCNN과 동률 |

**핵심 요약 (§3.4 실험 결과)**
- test 4,261건 기준: Late Fusion **0.6561**(수치상 최고), TextCNN **0.6556**(차이 0.0005 → 실질 동률).
- BERT+MLP frozen 단독 **0.5792** → baseline 미달; Late Fusion에서만 BERT 신호가 유효(+0.077).
- 오분류 패턴(기업·형사A/B)은 모델 공통 → EDA 유사도·불균형이 원인.

---

### 3.5 기존 연구와의 차이점 (Upgrade)

기존 판례·법률 텍스트 분류 연구와 본 연구의 차이는 다음과 같다.

| 구분 | 기존 연구 | 본 연구 |
|---|---|---|
| 분류 범위 | 영어 Legal-BERT [6], 단일 도메인·좁은 라벨 | **한국어 10개 사건 유형** 다중 분류 |
| 데이터 | 영어 법률 코퍼스, 판결 3분류 등 | AI Hub 「상황에 따른 판례」 42,601건 |
| 비교 방식 | 특정 SOTA 모델 성능 제고 | **동일 조건 5모델 공정 비교** |
| 표현 방식 | BERT 단독 또는 TF-IDF 단독 | **희소·학습임베딩·사전학습 3계열** |
| 앙상블 | Early Fusion 또는 미적용 | **Late Fusion(확률 가중합)** 실증 |
| 재현성 | 코드 비공개 다수 | **GitHub 전 과정 공개** |

Chalkidis et al.[6]의 Legal-BERT는 영어 법률 코퍼스에서 도메인 적응 효과를 보였으나, 본 연구에서 KLUE-BERT를 frozen만으로 사용한 경우 Macro F1 0.5792로 baseline 미달이었다. 이는 법률 도메인 적합성이 모델 선택만으로 보장되지 않으며, **fine-tuning·도메인 코퍼스 추가 학습**이 후속 과제임을 시사한다 [6][7].

### 3.6 수행계획서 대비 변경·달성 사항

본 프로젝트는 학기 초 제출한 **수행계획서**(제목: 「법률 판례 텍스트 기반 사건 대분류 자동 분류 모델 성능 비교」)를 기반으로 수행하였으며, 중간발표 교수 피드백과 실험 결과에 따라 일부 설계를 변경하였다.

**표 9. 수행계획서 vs 최종 수행 결과**

| 항목 | 수행계획서 (초기) | 최종 수행 결과 |
|---|---|---|
| 핵심 질문 | TF-IDF vs BERT 표현 방식 차이 | **3계열**(희소·학습임베딩·사전학습) + 앙상블 비교 |
| 비교 모델 | TF-IDF+LR/SVM, BERT(frozen)+MLP, **Early Fusion(concat)** | TF-IDF+LR/SVM, **TextCNN**, BERT+MLP, **Late Fusion** |
| 하이브리드 | TF-IDF·BERT 벡터 차원 결합 후 MLP | 확률 가중합 α 탐색 (교수 피드백 ①③ 반영) |
| 입력 조건 | 판시사항 / 요약문 / 결합 3종 실험 | 동일 — **결합 입력** 채택 (H2 지지) |
| 데이터 분할 | 8:1:1 stratify | 동일 — 42,601건 → 34,080 / 4,260 / 4,261 |
| 평가 지표 | Accuracy + **Macro F1** + 혼동 행렬 | 동일 — Macro F1 주지표 |
| BERT 활용 | frozen 추출 + MLP (미세조정 없음) | 동일 — KLUE-BERT [4] `[CLS]` 768차원 |
| GPU 환경 | GPU 배치 처리·중간 결과 저장 계획 | RunPod RTX PRO 4500, `.npy` 캐싱 |
| 역할 분담 | 이승준(전처리), 이윤수(EDA), 김동현(baseline), 김홍근(BERT·하이브리드), 정래원(보고서) | **계획대로 수행**, TextCNN·Late Fusion 추가 반영 |

수행계획서에서 예상하였던 “문맥 모델이 항상 베이스라인을 넘을 것”이라는 가정은 **실증적으로 기각**되었고, 대신 “어휘 신호와 문맥 신호를 **Late Fusion으로 결합**하면 단독 BERT보다 크게 개선된다”는 결론을 도출하였다. 수행계획서 §기대효과에서 언급한 한계(BERT 미세조정 없음, 판시사항·요약문 중심 입력, 토큰 길이 상한)는 본 보고서 **부록 B**에서 그대로 인정·보완 방향으로 정리하였다.

### 3.7 중간발표(12주차) 대비 개선·달성 사항

본 팀은 **2026.05.06** 중간발표(「판례 텍스트 기반 사건 유형 자동 분류 모델 성능 비교 연구」, 28슬라이드) 당시 **전처리·EDA·베이스라인(TF-IDF 6실험)** 까지를 공유하였고, BERT+MLP·하이브리드는 **향후 계획** 단계였다. 중간발표 이후 교수 피드백·RunPod 본 실험·최종 Q&A를 거쳐 설계와 결론이 다음과 같이 **구체화·개선**되었다.

**표 10. 중간발표(12주차) vs 최종 수행 결과 (Before / After / 근거)**

| 항목 | Before (중간발표) | After (최종) | 근거 파일 |
|---|---|---|---|
| 완료 범위 | 전처리·EDA·베이스라인 6실험 | **5개 모델** + test 4,261건 | `README.md`, `bert/`, `textcnn/`, `hybrid/` |
| 비교 모델 | **4개** (concat 하이브리드) | **5개** (TextCNN, Late Fusion) | `산출물/모델_구성_v2.md` |
| H1 | TF-IDF vs BERT | **3계열** 비교 | `README.md` §6, `textcnn/` |
| H2 | 입력 3종 비교 계획 | 결합 입력 **확정** | `baseline/baseline.ipynb` |
| H3 | concat 하이브리드 | **Late Fusion** α=0.8 | `hybrid/late_fusion.ipynb` |
| BERT 기대 | baseline **상회 예상** | 단독 **0.5792 미달** | `bert/bert_mlp.ipynb` |
| 정확도 목표 | Accuracy **80%+** | Macro F1 **0.6561** | `artifacts/late_fusion_alpha.png` |
| BERT 활용 | Fine-tuning 예정(H4) | **Frozen 유지** | `bert/extract_embeddings.ipynb` |
| 실험 환경 | Colab GPU | **RunPod** RTX PRO 4500 | `RUN_GUIDE.md` |
| 혼동 패턴 | 베이스라인 CM | **모델 공통 패턴** | `artifacts/confusion_matrix_bert_mlp.png` |
| 연구 질문 | BERT 도입 필요 | **표현 방식 비교** | 본 최종보고서 §3.4.6 |

**중간발표에서 이미 달성·유지된 성과**

- AI Hub 판례 **42,601건** 전처리 및 stratify 8:1:1 분할 (train/val/test 건수 동일)
- EDA 3대 발견: **클래스 불균형 56배**, 텍스트 길이 차이, **형사A↔B·기업↔민사 유사도**
- 베이스라인 **6실험(입력 3×모델 2)**: 결합 입력이 단일 입력보다 우위 → **H2 중간 단계 지지**
- GridSearchCV 튜닝 후 **TF-IDF+SVM Macro F1 0.6347** 기준선 확립 (최종 실험과 동일)
- 혼동 행렬: 특허/저작권·민사 양호, 기업·형사A/B 취약 (최종 실험에서도 동일)

**중간발표 이후 주요 개선 사항**

1. **교수 피드백 4건 반영** (표 7): 하이브리드 구체화(Late Fusion), 토큰 수 근거 정리, TextCNN 추가, Early Fusion 폐기
2. **TextCNN 도입**: 중간발표에는 없던 **학습 임베딩 계열** 비교군 → test Macro F1 **0.6556**으로 베이스라인 상회, 비용 대비 최고 효율
3. **하이브리드 방식 변경**: concat(차원 10,000+768) → **Late Fusion** → BERT+MLP 단독 대비 **+0.077**, baseline 상회
4. **가설 재정의**: H1·H3를 “모델 승패”가 아닌 **“표현 방식·결합 방식의 실증 비교”** 로 명확화 (최종 Q&A 3건 반영)
5. **전체 test 평가**: 중간발표는 베이스라인까지; 최종은 **unseen test 4,261건** 기준 5모델 일괄 보고
6. **재현성**: GitHub 공개, `SMOKE_TEST` 파이프라인 점검 후 본 실험, `artifacts/` 공식 그래프 2종

**중간발표 당시 가정과 최종 실증의 차이 (핵심 교훈)**

중간결론에서는 “유사 카테고리 구분을 위해 **BERT 문맥 모델 도입 필요**”, “하이브리드로 **성능 향상**”, “Accuracy **80%+** 목표”를 제시하였다. 최종 실험에서는 frozen BERT+MLP **단독이 baseline에 미달**하였으나, **Late Fusion(0.6561)** 과 **TextCNN(0.6556)** 이 baseline을 상회하여, **“비싼 문맥 모델 단독”보다 “어휘+n-gram+앙상블”이 본 데이터·조건에서 더 실용적**임을 보였다. 이는 Lee[14]·Chalkidis et al.[6] 등에서 **도메인 fine-tuning** 시 BERT 우위가 보고된 것과 달리, **frozen 일반 한국어 BERT만으로는 법률 10-클래스 분류에서 자동 우위가 보장되지 않음**을 시사한다.

**핵심 요약 (§3.7 중간→최종)**
- 4모델(concat) → **5모델**(TextCNN, Late Fusion), 교수 피드백 4건 반영.
- BERT 단독 우위 가정 **기각** → TextCNN·Late Fusion이 baseline 상회.
- test 4,261건 전 모델 평가 완료, GitHub `main` 공개.

---

## 4. 기대효과

### 경제적 측면

**1. 비용 절감**
- **인건비 절감**: 판례 사건 유형 분류를 자동화하면 법률 전문가의 반복적 분류·검색 작업 부담을 줄일 수 있다. 본 연구에서 TextCNN(0.6556)이 frozen BERT+MLP(0.5792)보다 우수한 결과는, **고비용 GPU 사전학습 모델 없이도 실용적 성능**을 달성할 수 있음을 시사한다.
- **검색 효율 향상**: 사건 유형별 사전 분류를 통해 법률 검색 공간을 축소하면 검색 속도·정확도를 동시에 개선할 수 있다.

**2. 생산성 향상**
- **대량 판례 처리**: 수만 건의 판례를 빠르게 유형별로 분류·정리할 수 있어 법률 데이터 관리 효율이 향상된다.
- **모델 선택 가이드**: 동일 조건 비교 결과는 실무에서 과도한 모델 투자 없이 적절한 표현 방식을 선택하는 의사결정 근거가 된다.

**3. 시장 경쟁력 강화**
- 법률 검색 엔진, 판례 추천 시스템, 법률 상담 챗봇 등 Legal Tech 서비스의 기반 기술로 확장 가능하다.

### 사회적 측면

**1. 법률 정보 접근성 향상**
- 일반인도 관련 판례를 사건 유형별로 쉽게 검색·이해할 수 있는 환경을 조성한다.
- 법률 정보의 디지털화·자동화를 통해 사법 정보 접근의 형평성을 높인다.

**2. 교육적 가치**
- 텍스트 표현 방식 비교, 클래스 불균형 처리(Macro F1), 공정한 실험 설계 등 빅데이터·머신러닝 교육의 실전 사례를 제공한다.
- 전처리부터 모델링·평가·재현까지 전 과정을 GitHub에 공개하여 후속 학습·연구에 활용 가능하다.

**3. 연구 기반 마련**
- "frozen BERT만으로는 baseline을 넘기 어렵다", "Late Fusion에서 어휘 신호가 주력" 등의 실증 결과는 후속 한국어 법률 NLP 연구의 출발점이 된다.

수행계획서에서 기대하였던 효과 — 동일 법률 도메인 데이터에서 **표현 방식·입력 필드 선택이 성능에 미치는 영향을 수치로 정리**하고, 향후 유사 분류 과제 설계 시 입력·모델 선택의 참고 자료로 활용 — 는 본 보고서의 5개 모델 비교표와 H1~H3 검증 결과로 **달성**하였다.

---

## 5. 산학협력

> 수행계획서 기준 산업체 멘토(왓챠, 임해빈 매니저)가 배정되었으나, **실제 연구 방향·설계 변경은 지도교수(이청용) 피드백 중심**으로 진행하였다. 산업체 멘토와의 별도 기술 리뷰 세션은 수행하지 않았으며, 본 절은 지도교수 피드백·팀 협업·기술 학습 내용을 기재한다.

### 5.1 프로젝트 피드백 및 방향 제시 (지도교수)

**2026.05.06** 중간발표(28슬라이드)에서는 전처리·EDA·베이스라인(TF-IDF 6실험, SVM F1 0.6347)까지 공유하였고, BERT+MLP·TF-IDF+BERT concat 하이브리드는 미완료 과제로 제시하였다. 중간발표 직후 교수 피드백 4건을 반영하여 연구 설계를 보강하였으며, 상세 변경 내역은 **§3.7** 및 [`산출물/모델_구성_v2.md`](모델_구성_v2.md)에 정리하였다.

**표 7. 교수 피드백 및 본 팀의 반영**

| # | 교수 피드백 | 반영 결과 | 증거(산출물·코드 경로) |
|---|---|---|---|
| ① | TF-IDF+BERT 하이브리드 결합 방법 불명확 | **Late Fusion** α 그리드 탐색 | `hybrid/late_fusion.ipynb`, `artifacts/late_fusion_alpha.png` |
| ② | 토큰 수는 분류 단계 설계 근거 부적절 | **`[CLS]` 768차원 고정 벡터**로 설명 | `bert/extract_embeddings.ipynb`, `README.md` §5.4 |
| ③ | TF-IDF+BERT 직결합(Early Fusion) 부적절 | **확률 가중합 Late Fusion** 채택 | `산출물/모델_구성_v2.md`, `hybrid/late_fusion.ipynb` |
| ④ | 딥러닝 비교군 부족 | **TextCNN 추가** (4→5모델) | `textcnn/textcnn.ipynb`, `report_images/textcnn_report.png` |

피드백 반영으로 비교 모델이 4개에서 5개로 확대되었으며, 표현 방식 3계열(희소 빈도 / 학습 임베딩 / 사전학습 문맥) 비교 구도가 완성되었다.

### 5.2 최종 발표 Q&A 피드백 반영 (15주차 오프라인)

수요일 오프라인 Q&A에서 수집된 질문 3건을 본 보고서 **부록 A**에 정리하였으며, 핵심 논의는 다음과 같다.

1. **혼동 카테고리 원인**: EDA 유사도·클래스 불균형·도메인 키워드 변별력 차이로 해석
2. **TextCNN(CNN) 선택 이유**: Kim[2]의 1D TextCNN은 n-gram 패턴 추출에 적합하며, BERT와 함께 3계열 비교 목적
3. **KoBERT/Legal-BERT 미사용**: Chalkidis et al.[6] 등 영어 법률 모델과 달리, 본 연구는 동일 조건 **표현 방식 비교**에 초점 — KLUE-BERT [4] frozen 선택

### 5.3 팀 역할 분담 및 협업

**표 8. 팀 구성 및 역할** (수행계획서 §역할 분배와 동일)

| 이름 | 수행계획서 역할 | 최종 수행 내용 |
|---|---|---|
| 정래원 (팀장) | 성능 비교·혼동 분석·보고서·발표 | 실험 결과 종합, README·최종보고서·발표자료 작성 |
| 김동현 | TF-IDF·LR/SVM·하이퍼파라미터 탐색 | TF-IDF + LR/SVM, GridSearchCV 튜닝, H2 검증 |
| 김홍근 | BERT 임베딩 추출·MLP·특징 결합 | BERT+MLP, TextCNN, Late Fusion, RunPod 본 실험 |
| 이승준 | JSON 필드 추출·정제·분할 | JSON→CSV, 결측치 제거, stratify 분할 |
| 이윤수 | EDA·시각화 | 클래스 분포, 유사도, 키워드 분석 |

### 5.4 프로젝트 관련 기술 학습

- **Python 데이터 처리**: pandas, scikit-learn (TF-IDF, GridSearchCV)
- **딥러닝**: PyTorch, TextCNN [2], MLP 분류기
- **사전학습 모델**: transformers(KLUE-BERT [4]), frozen embedding 추출
- **임베딩**: fastText [3] 한국어 사전학습 벡터
- **클라우드 GPU**: RunPod 환경 구축, Network Volume 관리
- **협업**: Git, GitHub(`main` 브랜치), Jupyter Notebook 기반 재현 가능 연구
- **클라우드 실험**: RunPod 환경 구축, Network Volume·임베딩 캐싱, `RUN_GUIDE.md` 작성

---

## 6. 참고문헌

1. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *Proceedings of NAACL-HLT 2019*, 4171–4186. https://arxiv.org/abs/1810.04805

2. Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification. *Proceedings of EMNLP 2014*, 1746–1751. https://arxiv.org/abs/1408.5882

3. Bojanowski, P., Grave, E., Joulin, A., & Mikolov, T. (2017). Enriching Word Vectors with Subword Information. *Transactions of the ACL*, 5, 135–146. https://arxiv.org/abs/1607.04606

4. Park, S., et al. (2021). KLUE: Korean Language Understanding Evaluation. *NeurIPS 2021 Datasets and Benchmarks Track*. https://arxiv.org/abs/2105.09680

5. AI Hub. 「상황에 따른 판례」 데이터셋. National Information Society Agency. https://aihub.or.kr

6. Chalkidis, I., Fergadiotis, M., Malakasiotis, P., Aletras, N., & Androutsopoulos, I. (2020). LEGAL-BERT: The Muppets straight out of Law School. *Findings of EMNLP 2020*, 2898–2904. https://arxiv.org/abs/2010.02559

7. Niklaus, J., Mathews, C., & Stürmer, M. (2021). Swiss-Judgment-Prediction: A Multilingual Legal Judgment Prediction Benchmark. *Artificial Intelligence and Law*, 30, 1–31. https://arxiv.org/abs/2006.02013

8. Kittler, J., Hatef, M., Duin, R. P. W., & Matas, J. (1998). On Combining Classifiers. *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 20(3), 226–239.

9. Aletras, N., Tsarapatsanis, D., Preoţiuc-Pietro, D., & Lampos, V. (2016). Predicting Judicial Decisions of the European Court of Human Rights: A Natural Language Processing Perspective. *PeerJ Computer Science*, 2, e93. https://doi.org/10.7717/peerj-cs.93

10. Srivastava, G., Gupta, P., & Khare, M. (2014). A Survey on Late Fusion in Multimodal Systems. *International Journal of Advanced Research in Computer Science*, 5(4). (Late Fusion 개념 참고)

11. Sokolova, M., & Lapalme, G. (2009). A Systematic Analysis of Performance Measures for Classification Tasks. *Information Processing & Management*, 45(4), 427–437. https://doi.org/10.1016/j.ipm.2009.03.002

12. Medlock, B. (2008). An Introduction to NLP-based Textual Anonymisation. *Language Resources and Evaluation*, 42(2), 127–134. (법률 텍스트 NLP 배경 참고)

13. Vatsal, S., Meyers, A., & Ortega, J. E. (2023). Classification of US Supreme Court Cases using BERT-Based Techniques. *arXiv preprint*, arXiv:2304.08649. https://arxiv.org/abs/2304.08649

14. Lee, J. (2025). Legal text classification in Korean sexual offense cases: from traditional machine learning to large language models with XAI insights. *Artificial Intelligence and Law*. https://doi.org/10.1007/s10506-025-09454-w

15. AI Hub. 생성형 AI 법률·규정 텍스트 분석 데이터 고도화 — 상황에 따른 판례. https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71723

---

## 7. R&D 성과 (논문 / 특허 / SW 등록 등)

| 구분 | 성과 내용 |
|---|---|
| **SW / 코드 공개** | GitHub 전체 실험 파이프라인 공개 — https://github.com/Raewon12/big_data_programming-LEON |
| **실험 재현** | Notebook 5종 + `preprocess/make_processed_data.py` + `RUN_GUIDE.md` + `requirements-runpod.txt` / `requirements-cpu.txt` |
| **데이터 산출물** | 전처리 데이터 42,601건, `label_mapping.csv`, EDA 시각화 PNG 4종 |
| **실험 결과 그래프** | `artifacts/confusion_matrix_bert_mlp.png`, `artifacts/late_fusion_alpha.png` |
| **종합 보고서** | `README.md` (논문형 실험 보고서), 본 최종보고서 |
| **논문 / 특허** | 해당 없음 |

---

## 8. 첨부 — 팀 활동 사진 (2장 이상 필수)

**[첨부사진 1]**

*첨부 1. 팀 레옹 단체 사진 (2026학년도 1학기 빅데이터프로그래밍 프로젝트)*

---

**[첨부사진 2]**

*첨부 2. 15주차 최종 발표 Q&A 세션*

---

---

# 부록 A. 최종 발표 Q&A (15주차 오프라인)

### Q1. 혼동되는 카테고리가 있었는데, 그 이유는 무엇인가?

**A.** 본 실험에서 모델 종류와 관계없이 **기업→민사**, **형사A↔형사B**, **개인정보/ICT**에서 혼동이 반복되었다. 이는 특정 모델의 실패라기보다, **데이터와 라벨 구조**에서 오는 현상으로 해석하였다.

첫째, EDA 단계 TF-IDF 코사인 유사도 분석에서 **형사A↔형사B**(0.96), **기업↔민사**(0.94)가 서로 높은 유사도를 보였고, 혼동 행렬에서도 동일 쌍이 반복 출현하여 EDA 예측과 일치하였다. 둘째, 민사(35.8%)와 개인정보/ICT(0.6%) 간 **약 56배 클래스 불균형**이 존재하여 소수 클래스 학습이 부족하였다 [11]. 셋째, 특허/저작권은 "상표", "저작권" 등 도메인 용어 변별력이 높아 모든 모델에서 F1 0.93 이상을 기록한 반면, 기업·형사 계열은 법률 표현이 겹쳐 키워드·문맥만으로 구분이 어렵다.

### Q2. 판례는 시퀀스 데이터인데, CNN을 왜 사용했나? RNN이나 Transformer가 더 좋지 않나?

**A.** 판례는 시퀀스 데이터이므로 본 연구에서 **BERT를 문맥 기반 모델**로 포함하였다 [1]. TextCNN은 이미지용 2D CNN이 아니라 Kim[2]이 제안한 **텍스트 분류용 1D CNN**으로, 단어 임베딩 시퀀스 위에 convolution을 적용하여 **3~5개 단어 단위 n-gram 패턴**을 추출한다.

판례는 "상표 등록", "임금 지급", "소득세 부과"처럼 사건 유형을 구분하는 **짧은 법률 구문**이 강하기 때문에, TextCNN을 **학습 임베딩 기반 비교군**으로 사용하였다. 본 연구의 목적은 최신 SOTA 모델 하나를 선정하는 것이 아니라, **표현 방식 3계열(희소 빈도 / 학습 임베딩 / 사전학습 문맥)의 공정 비교**이다.

실제 결과 TextCNN Macro F1 **0.6556**으로 baseline(0.6347)을 상회하였고 Late Fusion(0.6561)과 거의 동률이었다. 반면 frozen BERT+MLP는 **0.5792**로 baseline 미달이었다. 즉 "더 복잡한 모델이 항상 더 좋다"는 가정은 성립하지 않았다.

### Q3. 왜 TF-IDF, TextCNN, BERT만 비교했나? KoBERT, KR-BERT, Legal-BERT는 검토하지 않았나?

**A.** 관련 선행연구 및 KCI 논문을 검토하였다. Chalkidis et al.[6]의 Legal-BERT는 영어 법률 코퍼스에서 도메인 적응 효과를 보였으나, **동일 데이터·동일 입력·동일 지표** 아래 TF-IDF, 일반 딥러닝, BERT 계열을 함께 비교한 **10개 한국어 판례 사건 유형 분류** 연구는 제한적이었다.

법률 도메인 특화 모델(KoBERT, KR-BERT, Legal-BERT 등)은 존재하나, 상당수는 **영어 법률 코퍼스 기반** [6][7]이거나 본 데이터셋과 라벨 체계가 다르다. 본 연구는 특정 SOTA 모델 선정보다 **표현 방식의 효과 비교**에 초점을 맞추어, 한국어 일반 코퍼스 사전학습 모델인 **KLUE-BERT** [4]를 frozen 방식으로 채택하였다.

다만 BERT+MLP 단독 F1 0.5792는 일반 한국어 BERT를 frozen만으로 사용하면 법률 분류에서 자동으로 우수하지 않음을 보여주며, Legal-BERT 계열 fine-tuning·도메인 코퍼스 추가 학습이 향후 과제로 남는다.

---

# 부록 B. 한계 및 향후 과제

1. **소수 클래스 성능**: 개인정보/ICT(0.6%) 등 극소수 클래스는 `class_weight`만으로 한계 → 오버샘플링, 데이터 증강, focal loss 검토
2. **Frozen BERT 한계**: fine-tuning 없이 `[CLS]` + 경량 MLP만으로 test Macro F1 0.5792 → Legal-BERT [6] fine-tuning, 더 큰 분류 헤드 검토
3. **입력 잘림**: 512 토큰 초과 문서 정보 손실 → sliding window, 요약 기반 입력 검토
4. **도메인 특화 모델 비교**: KoBERT, 법률 특화 한국어 BERT와의 추가 비교 실험
5. **데이터 누수·중복**: case id 기준 중복 제거·group split 미수행 → 유사 판례의 train/test 동시 포함 가능성
6. **통계적 유의성**: Late Fusion vs TextCNN 차이 0.0005 — 부트스트랩·McNemar 등 검정 미수행

---

---

# 산학협력프로젝트 결과보고서 (요약)

| 항목 | 내용 |
|---|---|
| **과제명** | 판례 텍스트 기반 사건 유형 자동 분류 — 텍스트 표현 방식별 성능 비교 연구 |
| **협력기관명** | 왓챠 (수행계획서 기준, 학술 연구로 진행) |
| **수행기간** | 2026.03 ~ 2026.06 |

### 참여인원 (총 6명: 교수 1명, 학부 수강생 5명)

| 구분 | 성명 | 소속/직급 |
|---|---|---|
| 교수 | 이청용 | 한성대학교 컴퓨터공학부 |
| 학생(팀장) | 정래원 | 한성대학교 |
| 학생 | 김동현 | 한성대학교 |
| 학생 | 김홍근 | 한성대학교 |
| 학생 | 이승준 | 한성대학교 |
| 학생 | 이윤수 | 한성대학교 |

### 추진배경

법률 판례는 매년 대량으로 축적되지만, 사건 유형별 분류·검색은 수작업에 의존하는 경우가 많다. Vatsal et al.[13]·Lee[14] 등 선행 연구는 법률 텍스트 분류의 가능성을 보여 주었으나, 분류 대상이 특정 국가·죄목으로 한정되거나 입력 필드별 동일 조건 비교가 부족한 경우가 많다. 판례 텍스트는 도메인 특화 용어, 사건 유형 간 의미 경계 모호성, 클래스 불균형(최대 56배)으로 자동 분류가 어렵다. 따라서 본 팀은 수행계획서에서 정한 바와 같이 AI Hub 「상황에 따른 판례」 [5] 약 6만여 건을 출발점으로 10개 사건 대분류 분류를 수행하고, 텍스트 표현 방식별 성능을 비교하였다.

### 목표 및 내용

1. AI Hub 판례 데이터 전처리 및 10-클래스 분류 데이터셋 구축 (42,601건)
2. TF-IDF+LR/SVM 베이스라인 구축 및 결합 입력 우수성 검증 (H2)
3. TextCNN [2], KLUE-BERT(frozen)+MLP [4] 딥러닝 모델 구현
4. Late Fusion [10] 앙상블(α 탐색) 구현 및 5개 모델 Macro F1 [11] 비교
5. 전 실험 코드·결과 GitHub 공개 (재현 가능)

### 수행결과

AI Hub 판례 42,601건을 train/val/test(34,080/4,260/4,261)로 분할하고, 5개 모델을 동일 결합 입력·Macro F1로 비교하였다.

| 모델 | Macro F1 | 베이스라인 대비 |
|---|---:|---|
| TF-IDF + SVM (베이스라인) | 0.6347 | 기준선 |
| TextCNN | 0.6556 | 상회 (+0.021) |
| BERT+MLP (frozen) | 0.5792 | 미달 (−0.056) |
| Late Fusion (α=0.8) | **0.6561** | 수치상 최고 (TextCNN과 +0.0005) |

**선정**: 수치상 최고 Late Fusion, **비용 대비 추천 TextCNN**, baseline TF-IDF+SVM, 후속 연구 BERT fine-tuning/Legal-BERT.

결합 입력이 단일 입력보다 우수하였고(H2 지지, 중간발표 6실험에서도 동일 확인). Late Fusion은 BERT+MLP 단독 대비 +0.077 개선되었으나 TextCNN과는 사실상 동률이었다(H3 부분 지지). 중간발표 당시 계획이던 TF-IDF+BERT **concat 하이브리드**는 교수 피드백에 따라 **Late Fusion**으로 변경되었고, **TextCNN**이 추가되어 표현 방식 3계열 비교가 완성되었다. 중간발표 목표(Accuracy 80%+)는 달성하지 못했으나, Macro F1 **0.6561**으로 베이스라인(0.6347)을 상회하였다. 오분류는 모델과 무관하게 클래스 의미 인접성·불균형에서 비롯되었다.

**GitHub**: https://github.com/Raewon12/big_data_programming-LEON  
**데이터 출처**: AI Hub 「상황에 따른 판례」 — https://aihub.or.kr

