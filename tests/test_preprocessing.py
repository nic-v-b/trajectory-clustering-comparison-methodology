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

    assert X.shape == (2, 15)
    assert metadata["trajectory_id"].tolist() == ["A", "B"]
    assert metadata["trajectory_duration_s"].tolist() == [10.0, 10.0]
