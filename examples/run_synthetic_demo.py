"""Run all four clustering algorithms on deterministic synthetic trajectories."""

import numpy as np
import pandas as pd

from trajectory_clustering import TrajectoryPreprocessor, fit_candidate


def _synthetic_trajectories(seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []

    for group in range(4):
        for member in range(12):
            n_points = int(rng.integers(45, 76))
            times = np.sort(rng.uniform(0.0, 300.0, n_points))
            u = times / times.max()

            if group == 0:
                x = 1000.0 + 5000.0 * u
                y = 1000.0 + 500.0 * u
                z = 300.0 + 700.0 * u
            elif group == 1:
                x = 6000.0 - 5000.0 * u
                y = 5000.0 - 700.0 * u
                z = 1800.0 - 800.0 * u
            elif group == 2:
                x = 1200.0 + 3500.0 * u
                y = 6000.0 - 4200.0 * u
                z = 700.0 + 120.0 * np.sin(np.pi * u)
            else:
                x = 5500.0 - 3200.0 * u
                y = 1200.0 + 4000.0 * u
                z = 2300.0 - 250.0 * u

            x = x + rng.normal(0.0, 80.0, n_points)
            y = y + rng.normal(0.0, 80.0, n_points)
            z = z + rng.normal(0.0, 25.0, n_points)

            trajectory_id = f"G{group}_T{member:02d}"
            for t, xv, yv, zv in zip(times, x, y, z):
                rows.append(
                    {
                        "trajectory_id": trajectory_id,
                        "time": float(t),
                        "X": float(xv),
                        "Y": float(yv),
                        "Z": float(zv),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    observations = _synthetic_trajectories()

    preprocessor = TrajectoryPreprocessor(
        id_column="trajectory_id",
        feature_columns=("X", "Y", "Z"),
        time_column="time",
        sequence_length=60,
        scale=True,
    )
    X, metadata = preprocessor.fit_transform(observations)

    candidates = [
        ("kmeans", {"n_clusters": 4}),
        ("gmm", {"n_components": 4}),
        ("dbscan", {"eps": 5.0, "min_samples": 3}),
        ("hdbscan", {"min_cluster_size": 4, "min_samples": 2}),
    ]

    summaries = []
    for algorithm, params in candidates:
        result = fit_candidate(X, algorithm, **params)
        summaries.append(result.metrics)

    summary_df = pd.DataFrame(summaries)
    columns = [
        "algorithm",
        "n_clusters_excl_noise",
        "n_noise",
        "noise_fraction",
        "silhouette",
        "davies_bouldin",
        "calinski_harabasz",
        "fit_runtime_s",
    ]

    print(f"Prepared trajectories: {len(metadata)}")
    print(f"Feature matrix shape: {X.shape}")
    print(summary_df[columns].to_string(index=False))


if __name__ == "__main__":
    main()
