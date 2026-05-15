"""
tests/test_substrate_db.py
===========================
SubstrateDB 클래스의 핵심 계산 로직을 검증하는 단위 테스트.

실제 Excel 파일이나 .pth 파일 없이 더미 데이터를 직접 주입하여
순수 비즈니스 로직(거리 계산, 정렬, 유사도 매칭)의 정확성을 확인합니다.

검증 항목:
    1. 파일 미존재 시 경고만 출력하고 예외 없이 초기화되는지 확인
    2. find_closest - 가장 가까운 제품을 올바르게 선택하는지 확인
    3. find_closest_top_k - 상위 K개를 거리 오름차순으로 반환하는지 확인
    4. find_visual_match - visual_library가 None일 때 None을 반환하는지 확인
    5. find_visual_match - 더미 라이브러리를 주입해 코사인 유사도 기반 정렬 확인
"""

import numpy as np
import pandas as pd
import pytest

from vsams.utils.substrate_db import SubstrateDB


def make_dummy_db() -> SubstrateDB:
    """
    실제 파일 의존 없이 더미 DataFrame을 주입한 SubstrateDB를 반환합니다.
    excel_path를 존재하지 않는 경로로 설정하여 파일 로딩 로직을 우회한 뒤,
    df를 직접 교체합니다.
    """
    db = SubstrateDB(excel_path="__non_existent_path_for_testing__.xlsx")

    # 더미 제품 데이터 주입
    data = {
        "product_name": ["Sus_BA", "Sus_HL", "Sus_SM", "Sus_4"],
        "roughness_avg": [0.30, 0.50, 0.10, 0.80],
        "gloss_avg": [40.0, 20.0, 60.0, 10.0],
    }
    db.df = pd.DataFrame(data)
    db.visual_library = None  # 시각 라이브러리는 별도 테스트에서 주입
    return db


class TestSubstrateDBInit:
    """초기화 관련 테스트"""

    def test_init_with_missing_file_does_not_raise(self):
        """존재하지 않는 경로를 주면 예외 없이 초기화되어야 합니다."""
        db = SubstrateDB(excel_path="__totally_fake_path__.xlsx")
        assert db.df is None

    def test_visual_library_none_on_missing_pth(self):
        """visual_library.pth 미존재 시 visual_library가 None이어야 합니다."""
        db = SubstrateDB(excel_path="__totally_fake_path__.xlsx")
        assert db.visual_library is None


class TestFindClosest:
    """find_closest 메서드의 로직 정확성 테스트"""

    def test_returns_none_when_df_is_none(self):
        """df가 None이면 None을 반환해야 합니다."""
        db = SubstrateDB(excel_path="__fake__.xlsx")
        result = db.find_closest(0.5, 20.0)
        assert result is None

    def test_finds_exact_match(self):
        """
        데이터에 정확히 일치하는 제품(roughness=0.30, gloss=40.0)이 있을 때
        해당 제품(Sus_BA)을 반환해야 합니다.
        """
        db = make_dummy_db()
        result = db.find_closest(roughness=0.30, glossiness=40.0)
        assert result is not None
        assert result["product_name"] == "Sus_BA"

    def test_finds_nearest_product(self):
        """
        입력값이 Sus_SM(Ra=0.10, Gloss=60.0)에 가까울 때 Sus_SM을 선택해야 합니다.
        """
        db = make_dummy_db()
        result = db.find_closest(roughness=0.12, glossiness=58.0)
        assert result["product_name"] == "Sus_SM"

    def test_result_contains_distance_key(self):
        """반환된 딕셔너리에 'distance' 키가 있어야 합니다."""
        db = make_dummy_db()
        result = db.find_closest(roughness=0.5, glossiness=20.0)
        assert "distance" in result


class TestFindClosestTopK:
    """find_closest_top_k 메서드의 로직 정확성 테스트"""

    def test_returns_empty_list_when_df_is_none(self):
        """df가 None이면 빈 리스트를 반환해야 합니다."""
        db = SubstrateDB(excel_path="__fake__.xlsx")
        result = db.find_closest_top_k(0.5, 20.0, k=3)
        assert result == []

    def test_returns_k_results(self):
        """k개의 결과를 반환해야 합니다."""
        db = make_dummy_db()
        result = db.find_closest_top_k(0.5, 20.0, k=3)
        assert len(result) == 3

    def test_returns_all_if_k_exceeds_data(self):
        """k가 데이터 개수보다 크면 전체를 반환해야 합니다."""
        db = make_dummy_db()
        result = db.find_closest_top_k(0.5, 20.0, k=100)
        assert len(result) == len(db.df)

    def test_results_are_sorted_by_distance_ascending(self):
        """반환 결과가 거리(distance) 오름차순으로 정렬되어 있어야 합니다."""
        db = make_dummy_db()
        result = db.find_closest_top_k(0.5, 20.0, k=4)
        distances = [r["distance"] for r in result]
        assert distances == sorted(distances), "결과가 거리 오름차순으로 정렬되지 않았습니다."

    def test_result_keys_are_correct(self):
        """반환된 딕셔너리 항목에 필수 키가 모두 포함되어 있어야 합니다."""
        db = make_dummy_db()
        result = db.find_closest_top_k(0.5, 20.0, k=1)
        required_keys = {"product_name", "distance", "roughness", "gloss"}
        assert required_keys.issubset(result[0].keys())

    def test_closest_product_is_first(self):
        """Sus_HL(Ra=0.50, Gloss=20.0)과 정확히 일치하는 입력 시 첫 번째 결과가 Sus_HL이어야 합니다."""
        db = make_dummy_db()
        result = db.find_closest_top_k(0.50, 20.0, k=4)
        assert result[0]["product_name"] == "Sus_HL"


class TestFindVisualMatch:
    """find_visual_match 메서드의 로직 정확성 테스트"""

    def test_returns_none_when_library_is_none(self):
        """visual_library가 None이면 None을 반환해야 합니다."""
        db = make_dummy_db()
        dummy_features = np.random.rand(2048).astype(np.float32)
        result = db.find_visual_match(dummy_features, k=3)
        assert result is None

    def test_finds_most_similar_product(self):
        """
        특정 제품과 동일한 특징 벡터를 입력했을 때
        해당 제품이 1위로 매칭되어야 합니다.
        """
        db = make_dummy_db()

        # 더미 visual library 생성 (각 제품마다 1장씩의 참조 이미지 특징 벡터)
        n_features = 128
        feat_a = np.random.rand(1, n_features).astype(np.float32)
        feat_b = np.random.rand(1, n_features).astype(np.float32)

        db.visual_library = [
            {"product_name": "Sus_BA", "features": feat_a, "ref_image": "dummy_a.jpg"},
            {"product_name": "Sus_HL", "features": feat_b, "ref_image": "dummy_b.jpg"},
        ]

        # feat_a[0]를 쿼리로 주면 Sus_BA가 1위여야 함
        query = feat_a[0]
        result = db.find_visual_match(query, k=2)

        assert result is not None
        assert len(result) == 2
        assert result[0]["product_name"] == "Sus_BA"

    def test_similarity_range_is_valid(self):
        """코사인 유사도 값이 -1 ~ 1 범위 안에 있어야 합니다."""
        db = make_dummy_db()
        n_features = 64

        db.visual_library = [
            {"product_name": "Sus_SM", "features": np.random.rand(3, n_features).astype(np.float32), "ref_image": "sm.jpg"},
        ]
        query = np.random.rand(n_features).astype(np.float32)
        result = db.find_visual_match(query, k=1)

        sim = result[0]["similarity"]
        assert -1.0 <= sim <= 1.0, f"유사도 값이 범위 밖에 있습니다: {sim}"
