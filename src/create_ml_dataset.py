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
    # 2. Calculate the time gap between consecutive measurements
    # ------------------------------------------------------------

    df["gap_to_previous"] = (
        df["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    # ------------------------------------------------------------
    # 3. Create shifted temperature columns
    #
    # row i-2 -> temperature 20 min before
    # row i-1 -> temperature 10 min before
    # row i   -> target
    # row i+1 -> temperature 10 min after
    # row i+2 -> temperature 20 min after
    # ------------------------------------------------------------

    df["temp_minus_20"] = df["temperature"].shift(2)
    df["temp_minus_10"] = df["temperature"].shift(1)

    df["target_temperature"] = df["temperature"]

    df["temp_plus_10"] = df["temperature"].shift(-1)
    df["temp_plus_20"] = df["temperature"].shift(-2)

    # ------------------------------------------------------------
    # 4. Create corresponding time gaps
    # ------------------------------------------------------------

    df["gap_minus_20_to_minus_10"] = (
        df["gap_to_previous"].shift(1)
    )

    df["gap_minus_10_to_target"] = (
        df["gap_to_previous"]
    )

    df["gap_target_to_plus_10"] = (
        df["gap_to_previous"].shift(-1)
    )

    df["gap_plus_10_to_plus_20"] = (
        df["gap_to_previous"].shift(-2)
    )

    # ------------------------------------------------------------
    # 5. Define acceptable sampling interval
    # ------------------------------------------------------------

    MIN_GAP = 9
    MAX_GAP = 11

    valid_sequence = (
        df["gap_minus_20_to_minus_10"].between(
            MIN_GAP, MAX_GAP
        )
        &
        df["gap_minus_10_to_target"].between(
            MIN_GAP, MAX_GAP
        )
        &
        df["gap_target_to_plus_10"].between(
            MIN_GAP, MAX_GAP
        )
        &
        df["gap_plus_10_to_plus_20"].between(
            MIN_GAP, MAX_GAP
        )
    )

    # ------------------------------------------------------------
    # 6. Keep only valid continuous sequences
    # ------------------------------------------------------------

    ml_df = df.loc[
        valid_sequence,
        [
            "timestamp",
            "temp_minus_20",
            "temp_minus_10",
            "temp_plus_10",
            "temp_plus_20",
            "target_temperature"
        ]
    ].copy()

    ml_df = ml_df.dropna().reset_index(drop=True)

    # ------------------------------------------------------------
    # 7. Show resulting dataset
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
    # 8. Define X and y
    # ------------------------------------------------------------

    feature_columns = [
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20"
    ]

    X = ml_df[feature_columns]
    y = ml_df["target_temperature"]

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
    # 9. Save ML dataset
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