import pandas as pd

from trajectory_clustering import TrajectoryPreprocessor


def test_equal_length_trajectory_matrix():
    df = pd.DataFrame(
        {
            "trajectory_id": ["A", "A", "A", "B", "B", "B"],
            "time": [0, 5, 10, 0, 5, 10],
            "X": [0, 5, 10, 10, 5, 0],
            "Y": [0, 1, 2, 3, 4, 5],
            "Z": [100, 150, 200, 200, 150, 100],
        }
    )

    preprocessor = TrajectoryPreprocessor(
        id_column="trajectory_id",
        feature_columns=("X", "Y", "Z"),
        time_column="time",
        sequence_length=5,
        scale=False,
    )
    X, metadata = preprocessor.fit_transform(df)

    assert X.shape == (2, 17)
    assert metadata["trajectory_id"].tolist() == ["A", "B"]
    assert metadata["trajectory_duration_s"].tolist() == [10.0, 10.0]


def test_default_representation_matches_paper():
    df = pd.DataFrame(
        {
            "trajectory_id": ["A", "A", "A"],
            "time": [0, 5, 10],
            "X": [0, 5, 10],
            "Y": [0, 1, 2],
            "Z": [100, 150, 200],
        }
    )

    preprocessor = TrajectoryPreprocessor(
        id_column="trajectory_id",
        feature_columns=("X", "Y", "Z"),
        time_column="time",
        scale=False,
    )
    X, metadata = preprocessor.fit_transform(df)

    assert X.shape == (1, 194)
    assert X[0, -2] == 10.0
    assert X[0, -1] == 3.0
    assert metadata["trajectory_duration_s"].iloc[0] == 10.0
    assert metadata["trajectory_n_points"].iloc[0] == 3
