"""
Analyse the selected temperature sensor, including temperature
statistics, sampling intervals and time-series behaviour.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_temperature.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "sensor_0869_temperature.csv"
)

SELECTED_SENSOR = 6012002000869

FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


def main():
    """Analyse the selected temperature sensor."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Load cleaned temperature data
    # ------------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True
        )
        .dt.tz_convert(
            "Australia/Melbourne"
        )
    )

    print("Clean data loaded.")
    print(f"Total rows: {len(df)}")

    # ------------------------------------------------------------
    # 2. Select one sensor
    # ------------------------------------------------------------

    sensor = df[
        df["sensorid"]
        == SELECTED_SENSOR
    ].copy()

    if sensor.empty:
        raise ValueError(
            f"No data found for sensor "
            f"{SELECTED_SENSOR}."
        )

    sensor = (
        sensor
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    print()
    print("===================================")
    print("SELECTED SENSOR")
    print("===================================")

    print(
        f"Sensor ID: "
        f"{SELECTED_SENSOR}"
    )

    print(
        f"Number of measurements: "
        f"{len(sensor)}"
    )

    print(
        f"First measurement: "
        f"{sensor['timestamp'].min()}"
    )

    print(
        f"Last measurement: "
        f"{sensor['timestamp'].max()}"
    )

    # ------------------------------------------------------------
    # 3. Calculate time gap between measurements
    # ------------------------------------------------------------

    sensor["gap_minutes"] = (
        sensor["timestamp"]
        .diff()
        .dt.total_seconds()
        / 60
    )

    print()
    print("===================================")
    print("SAMPLING INTERVAL")
    print("===================================")

    print(
        sensor["gap_minutes"].describe()
    )

    # ------------------------------------------------------------
    # 4. Count approximate gap sizes
    # ------------------------------------------------------------

    print()
    print(
        "Approximate sampling gaps:"
    )

    gap_counts = (
        sensor["gap_minutes"]
        .round()
        .value_counts()
        .sort_index()
    )

    print(
        gap_counts.head(10)
    )

    # ------------------------------------------------------------
    # 5. Temperature statistics
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("TEMPERATURE STATISTICS")
    print("===================================")

    print(
        sensor["temperature"].describe()
    )

    # ------------------------------------------------------------
    # 6. Plot complete temperature record
    # ------------------------------------------------------------

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        sensor["timestamp"],
        sensor["temperature"],
        linewidth=0.8
    )

    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    plt.title(
        f"Temperature Time Series - "
        f"Sensor {SELECTED_SENSOR}"
    )

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "temperature_time_series.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    # ------------------------------------------------------------
    # 7. Plot one week
    # ------------------------------------------------------------

    start_date = pd.Timestamp(
        "2024-05-01",
        tz="Australia/Melbourne"
    )

    end_date = pd.Timestamp(
        "2024-05-08",
        tz="Australia/Melbourne"
    )

    one_week = sensor[
        (
            sensor["timestamp"]
            >= start_date
        )
        &
        (
            sensor["timestamp"]
            < end_date
        )
    ]

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        one_week["timestamp"],
        one_week["temperature"],
        marker=".",
        markersize=3,
        linewidth=1
    )

    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    plt.title(
        "Temperature - 1 to 7 May 2024"
    )

    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 8. Save selected sensor
    # ------------------------------------------------------------

    sensor.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved as: "
        f"{OUTPUT_FILE}"
    )

    print()
    print("TIME GAP SUMMARY")

    print(
        sensor["gap_minutes"].describe()
    )

    print()
    print("MOST COMMON TIME GAPS")

    print(
        sensor["gap_minutes"]
        .round()
        .value_counts()
        .sort_values(
            ascending=False
        )
        .head(10)
    )

    print()
    print("Stage 2 finished!")


if __name__ == "__main__":
    main()