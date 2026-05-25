# V-SAMS (Visual Surface Analysis & Matching System)

[![Status](https://img.shields.io/badge/Status-Stable_v1.0-4c1)](https://github.com/HyunchanAn/SG_proj_003)
[![Python](https://img.shields.io/badge/Python-3.10+-007ec6)](https://github.com/HyunchanAn/SG_proj_003)
[![Vision](https://img.shields.io/badge/Vision-Mobile--SAM-d00)](https://github.com/HyunchanAn/SG_proj_003)
[![UI](https://img.shields.io/badge/UI-Streamlit-f39c12)](https://github.com/HyunchanAn/SG_proj_003)

🌐 **라이브 데모 (Live Demo)**: [https://sg-proj-003-vsams.streamlit.app/](https://sg-proj-003-vsams.streamlit.app/)

본 프로젝트는 100원 동전의 실제 이미지와 금속 표면에 비친 반사광 이미지를 분석하여 표면의 물리적 특성(조도, 광택도)을 추정하고, 이를 바탕으로 금속 표면의 종류(BA, HL, #4, 2B, SM 등)를 식별하는 시스템입니다.

## 🚀 주요 기능
1. **정밀 영역 추출 (SAM 기반)**: Meta의 SAM(Segment Anything Model)을 활용하여 복잡한 배경 속에서도 동전과 반사광 영역을 픽셀 단위로 정확히 분리합니다.
2. **물성 추정 알고리즘**:
    - **조도 (Roughness, Ra)**: 표면의 텍스처 밀도와 에지 강도를 물리적 조도(um)로 변환.
    - **광택도 (Glossiness)**: 실제 동전과 반사상 간의 상대적 선명도(Sharpness Ratio)를 비교하여 광택 수준 평가.
    - **방향성 (Directionality)**: 결의 정렬 상태를 분석하여 HL(긴 결)과 #4(짧은 결)를 정밀 구분.
3. **자동 표면 판별**: 수집된 데이터를 바탕으로 학습된 모델이 표면의 종류(BA, HL, #4, 2B, SM)를 실시간으로 예측합니다.
4. **Data Organizer**: 고품질 학습 데이터셋 구축을 위한 전용 관리 도구를 제공합니다. 수동 박스 마스킹을 통해 SAM 엔진의 결과를 보정할 수 있습니다.

## 📂 프로젝트 구조
- `app.py`: 메인 표면 분석 UI (Streamlit 배포용 진입점)
- `apps/`: 데이터 수집 및 관리용 보조 앱 (`data_organizer_app.py` 등)
- `vsams/`: 분석 엔진, 모델, DB 연동 등 핵심 로직 패키지
- `dataset/`: 검증된 고품질 데이터셋 및 메타데이터 저장소
- `scripts/`: 데이터 재계산, 모델 학습 등 독립 실행형 유틸리티 스크립트
- `docs/`: 개발 일지(development_log.txt) 및 실험 리포트, 메모 등 문서 보관
- `assets/`: 엑셀 등 외부 리소스 파일 보관

## 🛠 설치 및 실행
### 1. 환경 설치
```powershell
pip install -r requirements.txt
```

### 2. 메인 앱 실행 (표면 분석)
라이브 데모와 동일한 웹 앱을 로컬에서 실행합니다.
```powershell
python -m streamlit run app.py
```

### 3. 데이터 정리 도구 실행 (Data Organizer)
학습용 데이터를 구축하거나 마스킹 품질을 검증할 때 사용합니다.
```powershell
python -m streamlit run apps/data_organizer_app.py
```

## 📈 기술 스택
- **Engine**: PyTorch, Mobile-SAM (vit_t)
- **Image Processing**: OpenCV, PIL
- **UI**: Streamlit, Streamlit-Drawable-Canvas
- **Database**: Pandas, Excel

---
*마지막 업데이트: 2026-05-25*
