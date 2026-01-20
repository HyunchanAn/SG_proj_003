# V-SAMS: 시각 기반 표면 분석 및 매칭 시스템 (Visual-based Surface Analysis & Matching System)

## 개요 (Overview)
V-SAMS는 산업용 제품(피착제)의 사진을 분석하여 모재의 종류와 표면 마감 상태를 식별하는 지능형 시스템입니다. 분석된 표면 특성(표면 에너지, 거칠기 등)을 바탕으로 최적의 보호 필름을 자동으로 추천합니다.

핵심 기능:
- 시각적 분석 (Visual Analysis): 딥러닝을 활용하여 재질(예: 금속, 플라스틱)과 질감(예: 거울면, 거친면)을 동시에 분류합니다.
- 고차원 특징 추출 (Feature Extraction): 홀딩 파워 예측을 위한 2048차원 벡터 추출 기능을 제공합니다.
- 지능형 매칭 (Intelligent Matching): 시각적 특성을 물리적 속성으로 변환하여 적합한 보호 필름을 추천합니다.
- 통합 분석 (Integration): V-SAMS 비전 데이터와 DeepDrop 접촉각 데이터를 결합한 최종 성능 예측이 가능합니다.

## 기술 스택 (Tech Stack)
- AI Core: PyTorch, timm (ResNet50), Albumentations
- App/UI: Streamlit (빠른 프로토타이핑)
- Backend: Python (FastAPI 호환 구조)
- Data: 전이 학습(Transfer Learning)을 위해 MINC(재질) 및 DTD(텍스처) 오픈 데이터셋 활용

## 설치 방법 (Installation)

1. 저장소 복제 (Clone):
   ```bash
   git clone <repository-url>
   cd V-SAMS
   ```

2. 가상 환경 설정 (권장):
   ```bash
   python -m venv venv
   # Windows:
   .\venv\Scripts\activate
   # Mac/Linux:
   source venv/bin/activate
   ```

3. 의존성 패키지 설치:
   ```bash
   pip install -r requirements.txt
   pip install -e .
   ```
   * `pip install -e .` 명령어를 통해 `vsams` 패키지를 설치합니다. 이를 통해 프로젝트 내 어디서든 모듈을 import 할 수 있습니다.

## 사용법 (Usage)

### 1. 메인 데모 앱 (Prototype Demo)
피착제 분석 및 보호필름 추천 시연을 위한 메인 애플리케이션입니다.
```bash
python -m streamlit run app.py
```
* 사용자 모드 (User Demo): 사진을 업로드하고 AI 분석 및 추천 결과를 확인합니다. 
* 가중치 파일(`checkpoints/v_sams_model.pth`)이 있을 경우 실제 모델 추론을 수행하며, 없을 경우 시뮬레이션 모드로 동작합니다.

### 2. 데이터 라벨링 툴 (Labeling Tool)
AI 학습용 데이터를 쉽고 빠르게 수집/관리하기 위한 도구입니다.
```bash
python -m streamlit run labeler.py
```
* 기능: 이미지를 드래그 앤 드롭으로 업로드하고, 재질/마감 속성을 선택하면 자동으로 폴더(`dataset/train/Material_Finish`)에 분류하여 저장합니다. 한글 파일명 보안 처리가 되어 있습니다.

### 3. 모델 학습 (Training)
수집된 데이터를 기반으로 모델을 학습시킵니다.
```bash
python train.py
```
* Albumentations를 통한 데이터 증강과 Multi-task 학습을 수행합니다.

### 4. 통합 예측 파이프라인 (Integration)
비전 데이터와 물성 데이터를 결합하여 분석합니다.
```bash
python integration_pipeline.py
```

### 5. 라이브러리 사용 (Library Usage)
V-SAMS는 이제 파이썬 라이브러리로 제공됩니다. 다른 프로젝트에서 다음과 같이 사용할 수 있습니다.

```python
import vsams
from vsams.models.classifier import SurfaceClassifier

# 모델 초기화
model = SurfaceClassifier(num_materials=6, num_finishes=7)
print(f"V-SAMS Version: {vsams.__version__}")
```

## 프로젝트 구조 (Project Structure)
```text
V-SAMS/
├── vsams/                  # 메인 패키지 (Source Code)
│   ├── __init__.py         # 패키지 초기화
│   ├── models/             # AI 모델 아키텍처 (특징 추출 모드 포함)
│   └── utils/              # DB 로드/저장 및 검색 유틸리티
├── app.py                  # 메인 데모 애플리케이션 (Streamlit)
├── labeler.py              # 데이터 라벨링 도구 (Streamlit)
├── train.py                # 실전 데이터 학습 스크립트
├── integration_pipeline.py # V-SAMS + DeepDrop 통합 예측 골격
├── setup.py                # 라이브러리 설치 설정 파일
├── development_log.txt     # 프로젝트 개발 이력
├── database.json           # 제품 정보 DB
├── data_collection_guide.md # 데이터 수집 가이드라인
├── requirements.txt        # 의존성 목록
└── README.md               # 프로젝트 설명서
```

## 현재 진행 상황 (Current Status)
이 프로젝트는 "발사대(Launchpad)" 정비가 완료된 단계입니다.
- AI 모델: `models/classifier.py`에 특징 추출 모드 및 Multi-Head 구조가 완성되었습니다.
- 학습 시스템: 실제 데이터를 즉시 학습할 수 있는 `train.py`가 준비되었습니다.
- 데이터 수집: `labeler.py`를 통해 안전하게 데이터를 축적할 준비가 끝났습니다.
- 통합 준비: 타 모듈과의 데이터 융합을 위한 인터페이스가 정의되었습니다.
