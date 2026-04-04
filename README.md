# 법률 판례 텍스트 기반 사건 대분류 분류 모델 성능 비교

2026-1학기 빅데이터프로그래밍 | 팀 레옹

## 프로젝트 소개

법원 판례 텍스트를 활용하여 사건 유형을 자동으로 분류하는 모델을 구축하고, 서로 다른 텍스트 표현 방식 간 성능 차이를 비교 분석하는 프로젝트입니다.

## 데이터

- **출처**: AI Hub - 상황에 따른 판례 데이터
- **규모**: 약 66,511건
- **분류 대상**: 민사, 가사, 형사, 행정, 기업, 근로자, 특허/저작권, 금융조세, 개인정보/ICT 등 10개 카테고리
- **입력 텍스트**: 판시사항 / 요약문 / 두 텍스트 결합 (3가지 조건 실험)

## 모델 구성

| 모델 | 설명 |
|------|------|
| TF-IDF + Logistic Regression | Baseline |
| TF-IDF + SVM (LinearSVC) | Baseline |
| BERT 임베딩 + MLP | 문맥 기반 분류 |
| TF-IDF + BERT 하이브리드 | 단어 빈도 + 문맥 결합 |

## 평가 지표

- Accuracy, Macro F1-score, Precision, Recall
- 클래스별 Confusion Matrix 분석

## 기술 스택

- Python, PyTorch, Scikit-learn
- BERT (사전학습 가중치 frozen, 임베딩 추출용)
- Google Colab (GPU)

## 팀원

| 이름 | 역할 |
|------|------|
| 정래원 | 결과 분석 및 보고서 |
| 김동현 | Baseline 모델 (TF-IDF + LR/SVM) |
| 김홍근 | BERT + MLP / 하이브리드 모델 |
| 이승준 | 데이터 전처리 |
| 이윤수 | EDA 및 시각화 |
