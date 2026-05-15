"""
tests/test_classifier.py
=========================
SurfaceClassifier 모델의 아키텍처 무결성을 검증하는 단위 테스트.

검증 항목:
    1. 모델 초기화 - 기본/커스텀 클래스 수로 인스턴스가 생성되는지 확인
    2. Forward Pass - 더미 텐서를 통해 재질/마감 두 헤드의 출력 Shape 검증
    3. extract_features - 피처 벡터 추출 메서드의 출력 Shape 검증
    4. return_features 모드 - forward(x, return_features=True) 경로 검증
    5. eval/train 모드 - 모델 상태 전환이 정상적으로 작동하는지 확인
"""

import pytest
import torch

from vsams.models.classifier import SurfaceClassifier


# 테스트 전반에서 공유할 기본 설정값
DEFAULT_NUM_MATERIALS = 6
DEFAULT_NUM_FINISHES = 7
BATCH_SIZE = 2
IMG_SIZE = 224


@pytest.fixture
def dummy_input():
    """224x224 크기의 더미 RGB 이미지 배치를 생성합니다."""
    return torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE)


@pytest.fixture
def model():
    """기본 설정으로 SurfaceClassifier 인스턴스를 생성합니다."""
    m = SurfaceClassifier(
        num_materials=DEFAULT_NUM_MATERIALS,
        num_finishes=DEFAULT_NUM_FINISHES,
    )
    m.eval()
    return m


class TestSurfaceClassifierInit:
    """모델 초기화 관련 테스트"""

    def test_default_initialization(self):
        """기본 파라미터로 모델이 정상 생성되는지 확인합니다."""
        m = SurfaceClassifier()
        assert m is not None

    def test_custom_class_counts(self):
        """커스텀 클래스 수(재질 4개, 마감 3개)로 모델이 생성되는지 확인합니다."""
        m = SurfaceClassifier(num_materials=4, num_finishes=3)
        assert m is not None

    def test_backbone_features_positive(self):
        """백본의 feature 차원이 양수인지 확인합니다 (ResNet50 기준 2048)."""
        m = SurfaceClassifier()
        assert m.num_features > 0


class TestSurfaceClassifierForward:
    """Forward Pass 출력 Shape 검증 테스트"""

    def test_output_shapes_default(self, model, dummy_input):
        """
        기본 forward 호출 시 재질 헤드와 마감 헤드의 출력 Shape이
        (batch_size, num_classes) 형태인지 확인합니다.
        """
        with torch.no_grad():
            mat_logits, fin_logits = model(dummy_input)

        assert mat_logits.shape == (BATCH_SIZE, DEFAULT_NUM_MATERIALS), (
            f"Material head 출력 Shape 불일치: {mat_logits.shape}"
        )
        assert fin_logits.shape == (BATCH_SIZE, DEFAULT_NUM_FINISHES), (
            f"Finish head 출력 Shape 불일치: {fin_logits.shape}"
        )

    def test_output_shapes_custom_classes(self, dummy_input):
        """커스텀 클래스 수를 적용했을 때 출력 Shape이 그에 맞게 조정되는지 확인합니다."""
        n_mat, n_fin = 4, 3
        m = SurfaceClassifier(num_materials=n_mat, num_finishes=n_fin)
        m.eval()
        with torch.no_grad():
            mat_logits, fin_logits = m(dummy_input)

        assert mat_logits.shape == (BATCH_SIZE, n_mat)
        assert fin_logits.shape == (BATCH_SIZE, n_fin)

    def test_return_features_mode(self, model, dummy_input):
        """
        return_features=True 시 두 헤드를 거치지 않고
        백본 특징 벡터만 반환하는지 확인합니다.
        """
        with torch.no_grad():
            features = model(dummy_input, return_features=True)

        # 반환값이 tuple이 아닌 단일 텐서여야 함
        assert isinstance(features, torch.Tensor), "return_features 모드는 단일 Tensor를 반환해야 합니다."
        assert features.shape[0] == BATCH_SIZE
        # 두 번째 차원이 백본의 특징 차원과 일치해야 함
        assert features.shape[1] == model.num_features

    def test_single_image_input(self, model):
        """배치 크기 1의 단일 이미지 입력이 정상 처리되는지 확인합니다."""
        single_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        with torch.no_grad():
            mat_logits, fin_logits = model(single_input)

        assert mat_logits.shape == (1, DEFAULT_NUM_MATERIALS)
        assert fin_logits.shape == (1, DEFAULT_NUM_FINISHES)


class TestSurfaceClassifierFeatureExtraction:
    """extract_features 메서드 검증 테스트"""

    def test_feature_shape(self, model, dummy_input):
        """extract_features 출력의 Shape이 (batch_size, num_features)인지 확인합니다."""
        features = model.extract_features(dummy_input)
        assert features.shape == (BATCH_SIZE, model.num_features), (
            f"extract_features 출력 Shape 불일치: {features.shape}"
        )

    def test_feature_is_detached_tensor(self, model, dummy_input):
        """extract_features가 그래디언트가 없는 Tensor를 반환하는지 확인합니다."""
        features = model.extract_features(dummy_input)
        # torch.no_grad() 블록 내에서 계산되므로 grad_fn이 없어야 함
        assert features.grad_fn is None, "extract_features 결과에 grad_fn이 있어서는 안 됩니다."

    def test_feature_dtype_float(self, model, dummy_input):
        """추출된 특징 벡터가 float32 타입인지 확인합니다."""
        features = model.extract_features(dummy_input)
        assert features.dtype == torch.float32


class TestSurfaceClassifierModeSwitch:
    """train/eval 모드 전환 테스트"""

    def test_eval_mode(self, model):
        """모델이 eval 모드로 전환되는지 확인합니다."""
        model.eval()
        assert not model.training

    def test_train_mode(self, model):
        """모델이 train 모드로 전환되는지 확인합니다."""
        model.train()
        assert model.training
