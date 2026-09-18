import numpy as np

from trajectory_clustering import evaluate_clustering, fit_candidate


def _two_cluster_matrix():
    return np.asarray(
        [
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1],
        ]
    )


def test_kmeans_candidate_returns_two_clusters():
    result = fit_candidate(_two_cluster_matrix(), "kmeans", n_clusters=2)
    assert result.metrics["n_clusters_excl_noise"] == 2
    assert result.metrics["silhouette"] is not None


def test_gmm_candidate_returns_two_clusters():
    result = fit_candidate(_two_cluster_matrix(), "gmm", n_components=2)
    assert result.metrics["n_clusters_excl_noise"] == 2


def test_density_noise_is_excluded_from_metrics():
    X = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 5.0],
            [20.0, 20.0],
        ]
    )
    labels = np.asarray([0, 0, 1, 1, -1])

    metrics = evaluate_clustering(X, labels, algorithm="dbscan")

    assert metrics["n_noise"] == 1
    assert metrics["n_samples_valid_for_metrics"] == 4
    assert metrics["n_clusters_excl_noise"] == 2


def test_hdbscan_candidate_runs():
    result = fit_candidate(
        _two_cluster_matrix(),
        "hdbscan",
        min_cluster_size=2,
        min_samples=1,
    )
    assert len(result.labels) == 6
