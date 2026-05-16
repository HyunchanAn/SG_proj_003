# V-SAMS (Surface Analysis & Measurement System)

![Status](https://img.shields.io/badge/Status-v1.2.0_Stable-brightgreen)
![Python](https://img.shields.io/badge/Python-3.14+-blue)
![Backend](https://img.shields.io/badge/Backend-PyTorch_/_OpenCV-orange)
![Physics](https://img.shields.io/badge/Physics-Coin--Reflection-red)

V-SAMS는 산업용 스테인리스강의 표면 마감 상태를 동전 반사 원리를 활용해 정밀 분석하는 물리 기반 측정 엔진입니다.

## 핵심 기능

1. 지능형 ROI 자동 탐지
- CLAHE 전처리 및 질감 분석 알고리즘을 통한 실시간 동전 위치 추적
- 이미지 중앙 가중치 적용으로 조명 반사 등 주변 노이즈와 실제 객체 완벽 구분

2. 물리 기반 정밀 분석 엔진
- 가변 블러링(Adaptive Blurring) 기술을 활용해 표면 결(Grain) 노이즈를 제거하고 반사된 상의 선명도만 추출
- 조도(Ra)와 광택도(Gloss) 수치를 정량화하여 산업 표준 마감(SM, BA, HL, #4) 자동 분류

3. 사용자 중심 인터페이스
- 이미지 업로드 시 별도의 조작 없이 즉시 분석 결과를 도출하는 Zero-Click UX 구현
- 분석에 사용된 동전 및 반사광 영역의 크롭 이미지를 실시간으로 제공하여 신뢰도 확인 가능

## 기술 스택 (Tech Stack)
- Engine: Python, OpenCV (Image Processing)
- UI: Streamlit
- Data: 산업용 금속 피착제(BA, SM, HL, #4 등) 23종의 물성치 DB 연동

## 사용 방법 (Usage)
1. `streamlit run app.py` 명령어로 메인 허브를 실행합니다.
2. 분석할 금속 표면 사진을 업로드합니다.
3. 캔버스에서 첫 번째 박스로 실제 동전 을, 두 번째 박스로 반사된 상 을 지정합니다.
4. 즉시 계산된 추정 조도(Ra)와 광택도(%)를 확인하고 DB 내 유사 제품을 매칭합니다.

## 프로젝트 구조 (Project Structure)
- vsams/analysis/surface_evaluator.py: 핵심 물리 분석 엔진
- vsams/utils/substrate_db.py: 제품 물성치 DB 핸들러
- app.py: 메인 분석 허브 (Unified Hub)
- tests/: 알고리즘 및 모듈 검증용 단위 테스트

## 현재 상태 (Current Status)
- 리팩토링 완료: 레거시 PoC 코드를 제거하고 물리 알고리즘 중심으로 엔진을 단일화하였습니다.
- 안정성 확보: 27개의 단위 테스트를 통해 코어 모듈의 무결성을 보장합니다.
