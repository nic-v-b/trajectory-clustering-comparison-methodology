"""Comparable internal metrics for clustering candidates."""

import numpy as np
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


def evaluate_clustering(
    X: np.ndarray,
    labels: np.ndarray,
    *,
    algorithm: str,
    noise_label: int = -1,
) -> dict:
    """Evaluate a clustering assignment with noise handled explicitly.

    Noise points are retained in the reported totals but excluded from the
    silhouette, Davies-Bouldin, and Calinski-Harabasz calculations.
    """
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if len(X) != len(labels):
        raise ValueError("X and labels must contain the same number of samples.")

    has_noise_semantics = algorithm.lower() in {"dbscan", "hdbscan"}
    valid_mask = labels != noise_label if has_noise_semantics else np.ones(len(labels), dtype=bool)

    X_valid = X[valid_mask]
    labels_valid = labels[valid_mask]
    unique_valid = np.unique(labels_valid)

    output = {
        "algorithm": algorithm.lower(),
        "n_samples_total": int(len(labels)),
        "n_samples_valid_for_metrics": int(valid_mask.sum()),
        "n_noise": int(np.sum(labels == noise_label)) if has_noise_semantics else 0,
        "noise_fraction": (
            float(np.mean(labels == noise_label)) if has_noise_semantics else 0.0
        ),
        "n_clusters_excl_noise": int(len(unique_valid)),
        "silhouette": None,
        "davies_bouldin": None,
        "calinski_harabasz": None,
    }

    if len(unique_valid) >= 2 and len(X_valid) > len(unique_valid):
        output["silhouette"] = float(
            silhouette_score(X_valid, labels_valid)
        )
        output["davies_bouldin"] = float(
            davies_bouldin_score(X_valid, labels_valid)
        )
        output["calinski_harabasz"] = float(
            calinski_harabasz_score(X_valid, labels_valid)
        )

    return output
