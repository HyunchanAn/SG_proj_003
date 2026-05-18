import numpy as np
from hypothesis import given, settings
import hypothesis.strategies as st
from vsams.analysis.surface_evaluator import SurfaceEvaluator


@given(
    h=st.integers(min_value=20, max_value=300),
    w=st.integers(min_value=20, max_value=300),
    val_coin=st.integers(min_value=0, max_value=255),
    val_ref=st.integers(min_value=0, max_value=255),
)
@settings(max_examples=50, deadline=None)
def test_physical_evaluation_properties(
    h: int, w: int, val_coin: int, val_ref: int
) -> None:
    """Verifies that physical estimators return valid bounded results for any constant pixel values."""
    evaluator = SurfaceEvaluator()
    # Create constant dummy BGR images
    coin_img = np.full((h, w, 3), val_coin, dtype=np.uint8)
    ref_img = np.full((h, w, 3), val_ref, dtype=np.uint8)

    # Run roughness estimation
    ra = evaluator._estimate_roughness(coin_img, ref_img)
    assert isinstance(ra, float)
    assert 0.0 <= ra <= 1.02

    # Run gloss estimation
    gloss = evaluator._estimate_gloss(coin_img, ref_img)
    assert isinstance(gloss, float)
    assert 0.0 <= gloss <= 600.0


@given(
    h=st.integers(min_value=15, max_value=150),
    w=st.integers(min_value=15, max_value=150),
)
@settings(max_examples=30, deadline=None)
def test_random_image_stability(h: int, w: int) -> None:
    """Verifies that physical estimation does not throw exceptions on random images."""
    evaluator = SurfaceEvaluator()
    coin_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    ref_img = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)

    ra = evaluator._estimate_roughness(coin_img, ref_img)
    gloss = evaluator._estimate_gloss(coin_img, ref_img)

    assert isinstance(ra, float)
    assert isinstance(gloss, float)
    assert not np.isnan(ra)
    assert not np.isnan(gloss)
