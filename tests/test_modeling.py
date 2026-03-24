from src.modeling.thresholding import top_k_threshold


def test_top_k_threshold():
    threshold = top_k_threshold([0.1, 0.2, 0.9, 0.8], 0.5)
    assert 0.2 <= threshold <= 0.9
