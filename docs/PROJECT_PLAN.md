# V-SAMS Project Roadmap & Integration Plan

본 문서는 V-SAMS (SG_proj_003) 프로젝트의 독립적 고도화 계획과 향후 R.A.D.A.R (SG_proj_004) 플랫폼과의 원활한 병합(Integration)을 위한 아키텍처 연동 규격을 기술합니다.

---

## 1. Project Objective

V-SAMS는 금속 및 고분자 표면의 미세 마감(BA, HL, #4 등)을 사진 한 장으로 고속 판별하는 정밀 진단 모듈입니다. 본 프로젝트는 독자적인 완성도와 무결성을 극대화하여 패키지화하되, 향후 004 종합 플랫폼의 핵심 서브엔진으로 플러그인(Plug-in) 연결이 가능하도록 입출력 레이아웃을 통일하는 것을 목표로 합니다.

---

## 2. Advanced Roadmap (Senior-Level Tasks)

- Phase 1: 아키텍처 문서화 및 strict type hints 정비 (현재 진행 중)
- Phase 2: pre-commit, ruff, mypy를 활용한 정적 분석 도구 도입 및 코드 포맷 표준화
- Phase 3: hypothesis 기반 Property-based testing을 통한 물리 엔진 엣지 케이스 수치 견고성 입증
- Phase 4: GitHub Actions CI 파이프라인 연동 (모킹 없이 실제 MobileSAM 추론 및 DB 매칭 완전성 검증)
- Phase 5: pyproject.toml 정비 및 독립 배포 패키지 빌드 검증

---

## 3. Integration Specification with SG_proj_004 (R.A.D.A.R)

SG_proj_002(DeepDrop-SFE)와 SG_proj_003(V-SAMS)은 독자적인 레포지토리 구조를 견고히 유지하며, 최종적으로 SG_proj_004(R.A.D.A.R) 플랫폼의 서브 모듈로서 연동됩니다. 이를 위해 다음과 같은 플러그인 아키텍처를 사전에 준수합니다.

```
+-------------------------------------------------------------+
|                     SG_proj_004 (R.A.D.A.R)                 |
|  - Main Platform Entrypoint (FastAPI / Streamlit)          |
|  - Substrate & Adhesive Matcher                             |
+------------------------------------+------------------------+
                                     | (Import as Plugin)
                                     v
+-------------------------------------------------------------+
|                     SG_proj_003 (V-SAMS)                    |
|  - Independent Package (vsams/)                             |
|  - input: PIL Image or Numpy Array                          |
|  - output: dict { roughness: float, gloss: float, ... }     |
+-------------------------------------------------------------+
```

### 3.1. Standardized Interface API
V-SAMS의 추론 결과를 004 플랫폼이 파싱하여 제품 자동 추천 데이터베이스와 매핑할 수 있도록 API 구조를 다음과 같이 표준화합니다.

```python
# vsams/analysis/surface_evaluator.py 의 표준 출력 인터페이스 예시
from typing import TypedDict, List

class SubstrateMatchResult(TypedDict):
    product_name: str
    category: str
    similarity: float

class VSAMSAnalysisOutput(TypedDict):
    roughness_ra: float
    gloss_percent: float
    detected_finish: str
    confidence: float
    matching_substrates: List[SubstrateMatchResult]
```

이와 같이 입출력 데이터 구조가 명확하게 정의되어 있으면, 004 플랫폼 측에서 V-SAMS의 복잡한 물리 계산 엔진이나 딥러닝 모듈의 세부적인 작동을 알 필요 없이 `vsams` 패키지를 그대로 패키지 임포트하여 표면 분석 기능을 손쉽게 가동할 수 있습니다.
