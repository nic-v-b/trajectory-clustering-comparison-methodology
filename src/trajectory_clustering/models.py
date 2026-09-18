"""Common fitting interface for the four comparison algorithms."""

from dataclasses import dataclass
import time
from typing import Any

import hdbscan
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.mixture import GaussianMixture

from .evaluation import evaluate_clustering


@dataclass
class CandidateResult:
    """Model, labels, metrics, and runtime for one clustering candidate."""

    algorithm: str
    model: Any
    labels: np.ndarray
    metrics: dict
    fit_runtime_s: float


def _build_model(algorithm: str, params: dict) -> Any:
    name = algorithm.lower()

    if name == "kmeans":
        defaults = {"n_init": 10, "random_state": 0}
        defaults.update(params)
        return KMeans(**defaults)

    if name == "dbscan":
        return DBSCAN(**params)

    if name == "hdbscan":
        return hdbscan.HDBSCAN(**params)

    if name == "gmm":
        defaults = {"random_state": 0}
        defaults.update(params)
        return GaussianMixture(**defaults)

    raise ValueError(f"Unsupported algorithm: {algorithm}")


def fit_candidate(
    X: np.ndarray,
    algorithm: str,
    **params,
) -> CandidateResult:
    """Fit one clustering candidate and return comparable evaluation metadata."""
    X = np.asarray(X, dtype=float)
    model = _build_model(algorithm, params)

    start = time.perf_counter()
    if algorithm.lower() == "gmm":
        model.fit(X)
        labels = model.predict(X)
    elif hasattr(model, "fit_predict"):
        labels = model.fit_predict(X)
    else:
        model.fit(X)
        labels = model.predict(X)
    runtime_s = float(time.perf_counter() - start)

    labels = np.asarray(labels)
    metrics = evaluate_clustering(
        X,
        labels,
        algorithm=algorithm,
    )
    metrics["fit_runtime_s"] = runtime_s

    return CandidateResult(
        algorithm=algorithm.lower(),
        model=model,
        labels=labels,
        metrics=metrics,
        fit_runtime_s=runtime_s,
    )
