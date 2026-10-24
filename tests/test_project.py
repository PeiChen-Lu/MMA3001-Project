"""
Tests for the missing-temperature reconstruction workflow.

The tests check the project's own functions with small synthetic
examples whose correct answers are known by hand (verification), and
check that the committed data files are consistent and reproducible.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sklearn.linear_model import LinearRegression

from src.clean_temperature import (
    count_conflicting_duplicates,
    extract_temperature_records,
)
from src.create_ml_dataset import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_windows,
)
from src.final_model_comparison import (
    NeighbourAverage,
    evaluate_model,
    load_best_depth,
    load_ml_dataset,
    split_development_test,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEMPERATURE_FILE = PROCESSED_DIR / "sensor_0869_temperature.csv"

ML_FILE = PROCESSED_DIR / "ml_temperature_dataset.csv"


def make_series(offsets_minutes, temperatures):
    """Build a timezone-aware temperature series for the tests."""

    start = pd.Timestamp("2024-05-01 00:00", tz="Australia/Melbourne")

    return pd.DataFrame({
        "timestamp": [
            start + pd.Timedelta(minutes=m) for m in offsets_minutes
        ],
        "temperature": temperatures
    })


# ============================================================
# Data files
# ============================================================

def test_processed_files_exist():
    """The committed processed data files are present."""

    assert TEMPERATURE_FILE.exists()
    assert ML_FILE.exists()


def test_temperature_data_is_valid():
    """The cleaned temperature file has the expected columns and finite values."""

    df = pd.read_csv(TEMPERATURE_FILE)

    assert len(df) > 0
    assert {"timestamp", "temperature"} <= set(df.columns)
    assert np.isfinite(df["temperature"]).all()

    timestamps = pd.to_datetime(df["timestamp"], utc=True)

    assert timestamps.is_monotonic_increasing
    assert timestamps.is_unique


def test_timestamps_with_daylight_saving_offsets_parse(tmp_path):
    """The data mixes +11:00 and +10:00 offsets (daylight saving ends)."""

    path = tmp_path / "mixed_offsets.csv"

    path.write_text(
        "timestamp,temperature\n"
        "2024-04-06 12:00:00+11:00,20.0\n"
        "2024-04-08 12:00:00+10:00,21.0\n"
    )

    df = load_ml_dataset_like(path)

    elapsed = df["timestamp"].iloc[1] - df["timestamp"].iloc[0]

    # Same wall-clock time two days later, but the clock went back 1 h.
    assert elapsed == pd.Timedelta(hours=49)


def load_ml_dataset_like(path):
    """Read a CSV with the same timestamp handling as the project."""

    df = pd.read_csv(path)

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Australia/Melbourne")
    )

    return df


def test_ml_dataset_columns_and_no_missing_values():
    """The machine-learning dataset has the required columns and no missing values."""

    df = pd.read_csv(ML_FILE)

    required = ["timestamp"] + FEATURE_COLUMNS + [TARGET_COLUMN]

    assert list(df.columns) == required
    assert df[required].isna().sum().sum() == 0


def test_ml_dataset_is_reproduced_from_sensor_data():
    """Rebuilding the windows gives the committed ML dataset."""

    sensor = load_ml_dataset_like(TEMPERATURE_FILE)

    rebuilt = build_windows(sensor)

    committed = pd.read_csv(ML_FILE)

    assert len(rebuilt) == len(committed)

    np.testing.assert_allclose(
        rebuilt[FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy(),
        committed[FEATURE_COLUMNS + [TARGET_COLUMN]].to_numpy()
    )


# ============================================================
# build_windows(): known answers
# ============================================================

def test_build_windows_values_on_a_linear_ramp():
    """On a straight line, the target equals the mean of its neighbours."""

    offsets = [10 * i for i in range(8)]
    temperatures = [20.0 + 0.5 * i for i in range(8)]

    windows = build_windows(make_series(offsets, temperatures))

    # 8 readings -> targets at positions 2, 3, 4, 5.
    assert len(windows) == 4

    first = windows.iloc[0]

    assert first["temp_minus_20"] == 20.0
    assert first["temp_minus_10"] == 20.5
    assert first[TARGET_COLUMN] == 21.0
    assert first["temp_plus_10"] == 21.5
    assert first["temp_plus_20"] == 22.0

    mean_of_neighbours = (
        windows["temp_minus_10"] + windows["temp_plus_10"]
    ) / 2

    np.testing.assert_allclose(
        mean_of_neighbours,
        windows[TARGET_COLUMN]
    )


def test_build_windows_drops_windows_that_span_a_gap():
    """A missing sample (20-minute gap) invalidates every window using it."""

    # Readings 0..9; the interval between readings 4 and 5 is 20 minutes.
    offsets = [0, 10, 20, 30, 40, 60, 70, 80, 90, 100]
    temperatures = list(range(10))

    windows = build_windows(make_series(offsets, temperatures))

    # Reading i needs the four intervals (i-2,i-1), (i-1,i), (i,i+1),
    # (i+1,i+2) to be ~10 min. Only i = 2 (uses readings 0-4) and
    # i = 7 (uses readings 5-9) avoid the gap; i = 3, 4, 5, 6 use it.
    assert list(windows[TARGET_COLUMN]) == [2, 7]


def test_build_windows_keeps_windows_away_from_the_gap():
    """Windows that do not touch a long gap are kept."""

    offsets = (
        [10 * i for i in range(8)]
        + [200 + 10 * i for i in range(8)]
    )

    windows = build_windows(make_series(offsets, list(range(16))))

    # 4 valid targets in each continuous block of 8 readings.
    assert len(windows) == 8


def test_build_windows_accepts_unsorted_input():
    """Unsorted input gives the same windows as sorted input."""

    offsets = [10 * i for i in range(8)]
    temperatures = [20.0 + 0.5 * i for i in range(8)]

    df = make_series(offsets, temperatures).sample(
        frac=1.0,
        random_state=0
    )

    assert len(build_windows(df)) == 4


def test_build_windows_rejects_missing_columns():
    """Input without the required columns raises a ValueError."""

    with pytest.raises(ValueError):
        build_windows(pd.DataFrame({"timestamp": []}))


# ============================================================
# extract_temperature_records(): invalid input handling
# ============================================================

def test_extract_temperature_records_skips_invalid_measurements():
    """Entries without a valid time or value are skipped and counted."""

    json_text = json.dumps([
        {
            "values": [
                {"time": 1700000000, "value": 23.5},
                {"time": 1700000600},
                {"value": 24.0}
            ],
            "variable": {"name": "Temperature", "unit": "C"}
        },
        {
            "values": [{"time": 1700000000, "value": 50}],
            "variable": {"name": "Humidity", "unit": "HR%"}
        }
    ])

    records, skipped = extract_temperature_records(123, json_text)

    assert len(records) == 1
    assert records[0]["temperature"] == 23.5
    assert records[0]["sensorid"] == 123
    assert skipped == 2


@pytest.mark.parametrize("bad_input", ["not json", float("nan"), "{}"])
def test_extract_temperature_records_rejects_invalid_json(bad_input):
    """Invalid or non-list JSON text raises a ValueError."""

    with pytest.raises(ValueError):
        extract_temperature_records(1, bad_input)


# ============================================================
# Chronological split: no leakage between development and test
# ============================================================

def test_split_leaves_a_purge_gap_between_sets():
    """The development and test sets are separated by the purge zone."""

    offsets = [10 * i for i in range(200)]

    df = make_series(offsets, [20.0] * 200)

    development_df, test_df = split_development_test(
        df,
        boundary_fraction=0.8,
        purge_minutes=30
    )

    assert len(development_df) > 0 and len(test_df) > 0

    separation = (
        test_df["timestamp"].min()
        - development_df["timestamp"].max()
    )

    # Every window spans +/-20 min, so the sets must be >= 40 min apart.
    assert separation >= pd.Timedelta(minutes=40)


def test_split_rejects_too_small_purge():
    """A purge zone shorter than 20 minutes raises a ValueError."""

    df = make_series([0, 10, 20, 30], [1, 2, 3, 4])

    with pytest.raises(ValueError):
        split_development_test(df, purge_minutes=10)


# ============================================================
# Metrics and models
# ============================================================

class ConstantOffsetModel:
    """Predicts y + 1 for the last sample, y otherwise (test double)."""

    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.array([1.0, 2.0, 4.0])


def test_evaluate_model_metrics_match_hand_calculation():
    """MAE, MSE, RMSE and R2 agree with a hand calculation."""

    X = pd.DataFrame({"a": [0, 0, 0]})
    y = pd.Series([1.0, 2.0, 3.0])

    result = evaluate_model(
        "test",
        ConstantOffsetModel(),
        X, y, X, y
    )

    # Errors are (0, 0, 1): MAE = 1/3, MSE = 1/3, RMSE = sqrt(1/3).
    assert result["mae"] == pytest.approx(1 / 3)
    assert result["mse"] == pytest.approx(1 / 3)
    assert result["rmse"] == pytest.approx(np.sqrt(1 / 3))
    assert result["r2"] == pytest.approx(1 - (1 / 3) / (2 / 3))


def test_linear_regression_recovers_neighbour_average():
    """Linear Regression should learn T(t) = (T(t-10) + T(t+10)) / 2."""

    rng = np.random.default_rng(0)

    X = pd.DataFrame(
        rng.normal(20, 2, size=(300, 4)),
        columns=FEATURE_COLUMNS
    )

    y = (X["temp_minus_10"] + X["temp_plus_10"]) / 2

    model = LinearRegression().fit(X, y)

    coefficients = dict(zip(FEATURE_COLUMNS, model.coef_))

    assert coefficients["temp_minus_10"] == pytest.approx(0.5)
    assert coefficients["temp_plus_10"] == pytest.approx(0.5)
    assert coefficients["temp_minus_20"] == pytest.approx(0, abs=1e-9)
    assert coefficients["temp_plus_20"] == pytest.approx(0, abs=1e-9)


def test_load_best_depth_reads_lowest_validation_rmse(tmp_path):
    """The depth with the lowest validation RMSE is selected."""

    path = tmp_path / "depth.csv"

    pd.DataFrame({
        "max_depth": [2.0, 8.0, np.nan],
        "validation_rmse": [0.30, 0.09, 0.10]
    }).to_csv(path, index=False)

    assert load_best_depth(path) == 8


def test_load_best_depth_returns_none_for_unlimited_tree(tmp_path):
    """An empty max_depth in the results means an unlimited tree (None)."""

    path = tmp_path / "depth.csv"

    pd.DataFrame({
        "max_depth": [2.0, np.nan],
        "validation_rmse": [0.30, 0.05]
    }).to_csv(path, index=False)

    assert load_best_depth(path) is None


def test_load_best_depth_reports_missing_results_file(tmp_path):
    """A missing optimisation results file raises FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_best_depth(tmp_path / "does_not_exist.csv")


def test_committed_depth_results_select_depth_8():
    """The committed optimisation results select max_depth = 8."""

    assert load_best_depth() == 8


def test_load_ml_dataset_is_chronological_and_timezone_aware():
    """The loaded dataset is in time order with Melbourne timestamps."""

    df = load_ml_dataset()

    assert df["timestamp"].is_monotonic_increasing
    assert str(df["timestamp"].dt.tz) == "Australia/Melbourne"


def test_neighbour_average_uses_only_the_two_nearest_neighbours():
    """Plain interpolation: T(t) = (T(t-10) + T(t+10)) / 2."""

    X = pd.DataFrame(
        {
            "temp_minus_20": [100.0, 0.0],
            "temp_minus_10": [20.0, 21.0],
            "temp_plus_10": [22.0, 21.5],
            "temp_plus_20": [-50.0, 99.0],
        }
    )

    prediction = NeighbourAverage().fit(X).predict(X)

    # (20 + 22) / 2 = 21 and (21 + 21.5) / 2 = 21.25
    assert prediction == pytest.approx([21.0, 21.25])


def test_count_conflicting_duplicates():
    """Only repeats with different temperatures count as conflicts."""

    df = pd.DataFrame(
        {
            "sensorid": ["a", "a", "a", "a", "a"],
            "timestamp": [1, 1, 2, 2, 3],
            "temperature": [20.0, 20.0, 21.0, 21.5, 22.0],
        }
    )

    repeated, conflicting = count_conflicting_duplicates(df)

    # Times 1 and 2 are repeated; only time 2 has different values.
    assert repeated == 2
    assert conflicting == 1