"""Equal-length trajectory interpolation and feature construction."""

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class TrajectoryPreprocessor:
    """Convert long-form trajectory observations into fixed-length vectors."""

    id_column: str = "icao24"
    feature_columns: Sequence[str] = ("X", "Y", "Z")
    time_column: str = "time"
    sequence_length: int = 60
    scale: bool = True
    min_points_per_trajectory: int = 2

    def __post_init__(self) -> None:
        self.scaler_: StandardScaler | None = None
        self.meta_: pd.DataFrame | None = None

    def _validate(self, df: pd.DataFrame) -> None:
        required = [self.id_column, self.time_column, *self.feature_columns]
        missing = [name for name in required if name not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        if self.sequence_length < 2:
            raise ValueError("sequence_length must be at least 2.")

    def _numeric_copy(self, df: pd.DataFrame) -> pd.DataFrame:
        work = df.copy()
        for col in [self.time_column, *self.feature_columns]:
            work[col] = pd.to_numeric(work[col], errors="coerce")
        return work.dropna(
            subset=[self.id_column, self.time_column, *self.feature_columns]
        ).copy()

    def _interpolate_one(
        self,
        trajectory: pd.DataFrame,
    ) -> tuple[np.ndarray, float, int] | None:
        trajectory = trajectory.sort_values(self.time_column).copy()

        t = trajectory[self.time_column].to_numpy(dtype=float)
        feature_arrays = [
            trajectory[col].to_numpy(dtype=float)
            for col in self.feature_columns
        ]

        t_unique, unique_idx = np.unique(t, return_index=True)
        t = t_unique
        feature_arrays = [values[unique_idx] for values in feature_arrays]

        n_points = len(t)
        if n_points < self.min_points_per_trajectory:
            return None

        t_rel = t - t[0]
        duration_s = float(t_rel[-1])

        if duration_s <= 0:
            sequences = [
                np.full(self.sequence_length, values[0], dtype=float)
                for values in feature_arrays
            ]
        else:
            normalized_t = t_rel / duration_s
            target_t = np.linspace(0.0, 1.0, self.sequence_length)
            sequences = [
                np.interp(target_t, normalized_t, values)
                for values in feature_arrays
            ]

        return np.concatenate(sequences), duration_s, n_points

    def fit_transform(
        self,
        df: pd.DataFrame,
    ) -> tuple[np.ndarray, pd.DataFrame]:
        """Return a standardized fixed-length matrix and trajectory metadata."""
        self._validate(df)
        work = self._numeric_copy(df)

        vectors = []
        meta_rows = []

        for trajectory_id, sub_df in work.groupby(self.id_column, sort=True):
            result = self._interpolate_one(sub_df)
            if result is None:
                continue

            vector, duration_s, n_points = result
            vectors.append(vector)
            meta_rows.append(
                {
                    self.id_column: trajectory_id,
                    "trajectory_duration_s": duration_s,
                    "trajectory_n_points": n_points,
                }
            )

        if not vectors:
            raise ValueError("No valid trajectories remain after preprocessing.")

        X = np.vstack(vectors)
        if self.scale:
            self.scaler_ = StandardScaler()
            X = self.scaler_.fit_transform(X)
        else:
            self.scaler_ = None

        self.meta_ = pd.DataFrame(meta_rows)
        return X, self.meta_.copy()
