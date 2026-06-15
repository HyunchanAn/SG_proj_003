# 표면 조도/광택도 평가 및 분류 기능 구현 계획

사용자가 제공한 `/test_260420_surface` 경로의 동전 반사광 이미지들을 분석하여 조도(um)와 광택도(deg)를 정밀하게 평가하고, 이를 `피착재 종류 및 물성(AI화)2.xlsx` 데이터와 비교하여 표면 타입(BA, #4, HL, SM, 2B)을 분류하는 기능을 구현합니다.

## User Review Required

> [!IMPORTANT]
> **SAM 및 위치 기반 분석 (User Feedback 반영)**
> - **SAM 활용**: 원래 동전과 반사된 상을 정밀하게 구분하기 위해 SAM(Segment Anything Model)을 사용하여 객체 마스킹을 수행합니다.
> - **위치 제약 조건**: "상단 = 실제 동전", "하단 = 반사된 상"이라는 물리적 위치를 기준으로 두 객체를 분류합니다. 특히 Super Mirror나 BA와 같은 고광택 표면에서 두 객체가 유사할 때 이 규칙을 최우선으로 적용합니다.
> - **조명 강인성**: 절대적인 밝기보다는 실제 동전 대비 반사된 상의 상대적 선명도(Contrast/Edge Ratio)를 분석하여 조명 변화에 대응합니다.

> [!NOTE]
> **새 브랜치 운용**
> 사용자의 요청에 따라 `feature/surface-analysis-260421` 브랜치에서 모든 작업을 진행하고 있습니다.

## Proposed Changes

### 1. 데이터베이스 모듈 업데이트

#### [MODIFY] [substrate_db.py](file:///c:/Users/chema/Github/SG_proj_003/vsams/utils/substrate_db.py)
- 새로운 엑셀 파일(`피착재 종류 및 물성(AI화)2.xlsx`)의 구조(7행부터 데이터 시작, 특정 컬럼 매핑)를 지원하도록 로드 로직을 개선하거나 확장합니다.
- `BA, #4, HL, SM, 2B` 다섯 가지 핵심 표면에 대한 기준 물성치를 별도로 관리하는 기능을 추가합니다.

### 2. 정밀 표면 분석 엔진 구현

#### [NEW] `surface_evaluator.py` (file:///c:/Users/chema/Github/SG_proj_003/vsams/analysis/surface_evaluator.py)
- **SAM 기반 동전/반사광 검출**:
    - MobileSAM을 사용하여 이미지 내의 모든 객체 마스크를 추출합니다.
    - 추출된 마스크 중 중심 Y좌표가 가장 높은(Y값이 작은) 객체를 **실제 동전**, 그 아래에 위치한 객체를 **반사된 상**으로 정의합니다.
- **광택도 평가 (Relative Sharpness)**:
    - 실제 동전의 엣지 선명도($S_{real}$)와 반사된 상의 엣지 선명도($S_{ref}$)를 비교합니다.
    - $Gloss \propto S_{ref} / S_{real}$ 공식을 적용하여 조명 변화에 관계없이 표면의 반사 능력을 수치화합니다.
- **조도 평가 (Scale-aware Texture)**:
    - 100원 동전의 실제 지름(24mm)을 레퍼런스로 해당 이미지의 `pixel-per-mm`을 계산합니다.
    - 표면 텍스처의 거칠기를 물리적 단위(um)로 맵핑하기 위해 spatial frequency를 분석합니다.
- **수치 매핑**: 추출된 지표를 `피착재 종류 및 물성(AI화)2.xlsx` 데이터와 유클리드 거리 기반으로 매칭하여 최종 분류를 수행합니다.

### 3. 검증 및 테스트 스크립트

#### [NEW] `test_surface_classification.py` (file:///c:/Users/chema/Github/SG_proj_003/scratch/test_surface_classification.py)
- `/test_260420_surface` 하위의 모든 폴더를 순회하며 이미지를 분석합니다.
- 분석된 조도/광택도를 바탕으로 엑셀 DB에서 가장 일치하는 제품을 찾습니다.
- 실제 폴더명(정답)과 알고리즘의 예측값(분류 결과)을 비교하여 정확도를 리포트합니다.

## Open Questions

- **동전의 종류**: 촬영에 사용된 동전이 일정한가요? (예: 100원 동전) 동전의 크기를 레퍼런스로 사용하여 픽셀당 실제 길이를 계산할 필요가 있는지 확인이 필요합니다.
- **조명 조건**: 모든 사진이 동일한 조명 환경에서 촬영되었나요? 반사광의 밝기는 조도에 민감하므로 일관된 환경일 때 분석 정확도가 높아집니다.

## Verification Plan

### Automated Tests
- `scratch/test_surface_classification.py`를 실행하여 5개 표면 타입에 대한 분류 정확도(Accuracy) 및 물성 추정 오차(MAE)를 확인합니다.
- `vsams/utils/substrate_db.py`의 새 엑셀 로드 기능을 단위 테스트합니다.

### Manual Verification
- Streamlit 앱(`app.py`)에 새로운 분석 엔진을 적용하여, 개별 이미지를 업로드했을 때 반사광 검출 결과와 추정된 수치가 타당한지 시각적으로 확인합니다.
