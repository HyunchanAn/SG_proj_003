# V-SAMS (Surface Analysis & Measurement System)

![Status](https://img.shields.io/badge/Status-v1.2.0_Stable-brightgreen)
![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Backend](https://img.shields.io/badge/Backend-PyTorch_/_OpenCV-orange)
![Physics](https://img.shields.io/badge/Physics-Coin--Reflection-red)

V-SAMS는 산업용 스테인리스강의 표면 마감 상태를 동전 반사 원리 및 딥러닝 특징 공간을 결합하여 정밀 분석하는 하이브리드 측정 엔진입니다.

---

## 1. Key Features

1. 지능형 ROI 자동 탐지
- CLAHE 전처리 및 질감 분석 알고리즘을 통한 실시간 동전 위치 추적
- 이미지 중앙 가중치 적용으로 조명 반사 등 주변 노이즈와 실제 객체 완벽 구분

2. 물리 기반 정밀 분석 엔진
- 가변 블러링(Adaptive Blurring) 기술을 활용해 표면 결(Grain) 노이즈를 제거하고 반사된 상의 선명도만 추출
- 조도(Ra)와 광택도(Gloss) 수치를 정량화하여 산업 표준 마감(SM, BA, HL, #4) 자동 분류

3. 하이브리드 판정 엔진
- MobileSAM(Segment Anything Model 경량화 버전)을 백본으로 탑재하여 표면의 미세 질감 특징을 2048차원 벡터로 인덱싱
- 물리 추정 값과 시각 유사도를 결합한 통합 가중치 계산을 통해 오분류 차단

4. 사용자 중심 인터페이스
- 이미지 업로드 시 별도의 조작 없이 즉시 분석 결과를 도출하는 Zero-Click UX 구현
- 분석에 사용된 동전 및 반사광 영역의 크롭 이미지를 실시간으로 제공하여 측정 정밀도 확인 지원

---

## 2. Technical Stack

- Language: Python >= 3.10
- Engine: OpenCV, PyTorch, MobileSAM (vit_t)
- Static Analysis: Ruff, Mypy
- Formatting & Linting: Black, Isort, Pre-commit
- Verification: Pytest, Hypothesis
- UI: Streamlit

---

## 3. Architecture Specification

전체 시스템의 구조적 아키텍처 및 상세 컴포넌트 정보는 다음 문서들을 참조하십시오.
- [ARCHITECTURE.md](file:///e:/Github/SG_proj_003/docs/ARCHITECTURE.md): Mermaid 시스템 흐름도 및 물리 분석 수식 해설
- [PROJECT_PLAN.md](file:///e:/Github/SG_proj_003/docs/PROJECT_PLAN.md): 고도화 계획 및 004 플랫폼 병합용 API 명세

---

## 4. Installation & Setup

### 사용자 설치
```bash
git clone https://github.com/HyunchanAn/SG_proj_003.git
cd SG_proj_003
pip install .
```

### 개발자 설치 (정적 분석 및 테스트 의존성 포함)
```bash
pip install -e .[dev]
pre-commit install
```

---

## 5. Usage Guide

### Streamlit 웹 인터페이스 실행
```bash
streamlit run app.py
```
실행 후 브라우저에서 제공되는 화면에 분석 대상 스테인리스강 사진을 업로드합니다. 업로드 시 자동으로 동전을 추적하고 표면 마감을 추정합니다.

### CLI 테스트 및 정적 분석 실행
```bash
# 전체 테스트 실행
pytest tests/

# pre-commit 정적 린팅 및 포맷팅 검사
pre-commit run --all-files
```

### FastAPI 플러그인 엔드포인트 실행
R.A.D.A.R(SG_proj_004) 플랫폼과의 연동을 위한 API 서버 구동:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
실행 후 브라우저에서 `http://localhost:8000/docs` (Swagger UI)에 접속하여 엔드포인트를 검증할 수 있습니다.

---

## 6. Limitations & Future Plan
- 조명 외란 영향: 과도하게 어두운 조명 혹은 다중 난반사 광원 환경에서는 대비(Gloss) 산출의 신뢰도가 다소 감소할 수 있습니다.
- 향후 계획: 해당 V-SAMS (SG_proj_003) 독립 모듈은 004 (R.A.D.A.R) 종합 자동화 플랫폼의 플러그인 서브엔진으로 병합될 예정입니다.

