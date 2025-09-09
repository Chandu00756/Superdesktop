import math
import types

import numpy as np


def test_extract_features_handles_heterogeneous_inputs():
    # Import locally to avoid heavy module import side effects in other tests
    from storage_node.main import IsolationForestAnomalyDetector

    det = IsolationForestAnomalyDetector()

    class Obj:
        def __init__(self, value):
            self.value = value

    values = [
        1, 2.5, "3", Obj(4.5), {"not": "numeric"}, "nan", 6, 7.0, "8.25", Obj(9)
    ]

    feats = det._extract_features(values)  # private method by design; tested for robustness
    assert isinstance(feats, list)
    assert len(feats) == 8
    # All numeric and finite
    for f in feats:
        assert isinstance(f, (int, float))
        assert not (isinstance(f, float) and (math.isnan(f) or math.isinf(f)))


def test_extract_features_with_minimal_and_empty_inputs():
    from storage_node.main import IsolationForestAnomalyDetector

    det = IsolationForestAnomalyDetector()

    # Empty -> zeros
    feats_empty = det._extract_features([])
    assert feats_empty == [0.0] * 8

    # Single value -> pads and computes stats deterministically
    feats_one = det._extract_features([5])
    assert len(feats_one) == 8
    # mean, std, min, max, median, delta, count_above_mean, avg_abs_diff_norm
    mean = feats_one[0]
    std = feats_one[1]
    min_v = feats_one[2]
    max_v = feats_one[3]
    median = feats_one[4]
    assert mean == 5.0
    assert min_v == 5.0 and max_v == 5.0 and median == 5.0
    # std could be 0 due to padding with edge value
    assert std == 0.0
    # Count above mean should be 0 when all equal
    assert feats_one[6] == 0

    # Two values
    feats_two = det._extract_features([1, 3])
    assert len(feats_two) == 8
    assert feats_two[0] == 2.0  # mean
    assert feats_two[2] == 1.0 and feats_two[3] == 3.0
    # last - first
    assert feats_two[5] == 2.0
