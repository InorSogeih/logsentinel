from logsentinel.evaluation import binary_metrics, choose_threshold


def test_threshold_selection_and_metrics():
    labels = [0, 0, 0, 1, 1, 1]
    scores = [0.05, 0.10, 0.20, 0.70, 0.80, 0.95]

    threshold, _ = choose_threshold(labels, scores)
    metrics = binary_metrics(labels, scores, threshold=threshold)

    assert 0.0 <= threshold <= 1.0
    assert metrics["f1"] == 1.0
    assert metrics["recall"] == 1.0
