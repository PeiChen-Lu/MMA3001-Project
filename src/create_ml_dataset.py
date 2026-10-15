"""
Create the machine-learning dataset used for missing temperature
measurement reconstruction.

The input features are surrounding temperature measurements at
t-20, t-10, t+10 and t+20 minutes. The prediction target is the
temperature at time t.
"""

from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sensor_0869_temperature.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_temperature_dataset.csv"
)


FEATURE_COLUMNS = [
    "temp_minus_20",
    "temp_minus_10",
    "temp_plus_10",
    "temp_plus_20"
]

TARGET_COLUMN = "target_temperature"


def build_windows(df, min_gap_minutes=9, max_gap_minutes=11):
    """
    Build reconstruction examples from one sensor's temperature series.

    Each example is centred on one measurement (the target) and uses the
    two measurements before and the two measurements after it as inputs.
    An example is kept only if all four neighbouring time intervals are
    between ``min_gap_minutes`` and ``max_gap_minutes``, so that the
    window is a continuous, approximately 10-minute sequence.

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain ``timestamp`` (timezone-aware datetimes) and
        ``temperature`` (degrees Celsius). Rows may be in any order.
    min_gap_minutes : float, optional
        Smallest accepted interval between consecutive measurements.
    max_gap_minutes : float, optional
        Largest accepted interval between consecutive measurements.

    Returns
    -------
    pandas.DataFrame
        Columns ``timestamp``, ``temp_minus_20``, ``temp_minus_10``,
        ``temp_plus_10``, ``temp_plus_20`` and ``target_temperature``
        (all temperatures in degrees Celsius). The frame is empty if no
        valid window exists.

    Raises
    ------
    ValueError
        If ``timestamp`` or ``temperature`` is missing from ``df``.
    """

    required_columns = {"timestamp", "temperature"}

    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    data = (
        df[["timestamp", "temperature"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    # Time gap (minutes) between each measurement and the previous one.
    gap_to_previous = (
        data["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    data["temp_minus_20"] = data["temperature"].shift(2)
    data["temp_minus_10"] = data["temperature"].shift(1)
    data[TARGET_COLUMN] = data["temperature"]
    data["temp_plus_10"] = data["temperature"].shift(-1)
    data["temp_plus_20"] = data["temperature"].shift(-2)

    acceptable_gap = gap_to_previous.between(
        min_gap_minutes,
        max_gap_minutes
    )

    # Intervals (i-2 -> i-1), (i-1 -> i), (i -> i+1), (i+1 -> i+2).
    valid_sequence = (
        acceptable_gap.shift(1, fill_value=False)
        & acceptable_gap
        & acceptable_gap.shift(-1, fill_value=False)
        & acceptable_gap.shift(-2, fill_value=False)
    )

    ml_df = data.loc[
        valid_sequence,
        ["timestamp"] + FEATURE_COLUMNS + [TARGET_COLUMN]
    ]

    return ml_df.dropna().reset_index(drop=True)


def main():
    """Create the machine-learning reconstruction dataset."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load selected sensor data
    # ------------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Australia/Melbourne")
    )

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("Temperature data loaded.")
    print(f"Number of original measurements: {len(df)}")

    # ------------------------------------------------------------
    # 2. Build windows of surrounding measurements
    #
    # row i-2 -> temperature 20 min before
    # row i-1 -> temperature 10 min before
    # row i   -> target
    # row i+1 -> temperature 10 min after
    # row i+2 -> temperature 20 min after
    #
    # Only continuous 10-minute sequences (gaps of 9-11 minutes) are
    # kept; see build_windows().
    # ------------------------------------------------------------

    ml_df = build_windows(df)

    # ------------------------------------------------------------
    # 3. Show resulting dataset
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("MACHINE LEARNING DATASET")
    print("===================================")

    print(f"Number of usable ML examples: {len(ml_df)}")

    print()
    print("First 10 examples:")
    print(ml_df.head(10))

    # ------------------------------------------------------------
    # 4. Define X and y
    # ------------------------------------------------------------

    X = ml_df[FEATURE_COLUMNS]
    y = ml_df[TARGET_COLUMN]

    print()
    print("===================================")
    print("X AND y")
    print("===================================")

    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    print()
    print("Example X:")
    print(X.head())

    print()
    print("Example y:")
    print(y.head())

    # ------------------------------------------------------------
    # 5. Save ML dataset
    # ------------------------------------------------------------

    ml_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"ML dataset saved as: {OUTPUT_FILE}")

    print()
    print("Stage 3 finished!")


if __name__ == "__main__":
    main()