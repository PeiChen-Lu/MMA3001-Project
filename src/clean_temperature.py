"""
Extract and clean temperature measurements from the raw MMA3001
environmental sensor dataset.

This module reads the raw CSV file in chunks, extracts temperature
measurements from JSON records, removes duplicated measurements,
converts timestamps, and saves the cleaned dataset.
"""

from pathlib import Path
import pandas as pd
import json
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "5EnvSensor_MayToDec2024_180kRows.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_temperature.csv"
)


def main():
    """Extract and clean temperature sensor measurements."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Start timer
    # ------------------------------------------------------------

    start_time = time.time()

    print("Starting data cleaning...")

    # ------------------------------------------------------------
    # 2. Read the large CSV in chunks
    # ------------------------------------------------------------

    chunk_size = 20000

    all_temperature_records = []

    chunk_number = 0

    skipped_invalid_measurements = 0

    # ------------------------------------------------------------
    # 3. Process each chunk
    # ------------------------------------------------------------

    for chunk in pd.read_csv(
        INPUT_FILE,
        usecols=["sensorid", "jsondata"],
        chunksize=chunk_size
    ):

        chunk_number += 1

        print(f"Processing chunk {chunk_number}...")

        # --------------------------------------------------------
        # 4. Go through each raw row
        # --------------------------------------------------------

        for row in chunk.itertuples(index=False):

            sensor_id = row.sensorid

            try:
                sensor_data = json.loads(row.jsondata)

            except (json.JSONDecodeError, TypeError):
                continue

            if not isinstance(sensor_data, list):
                continue

            # ----------------------------------------------------
            # 5. Find Temperature entries
            # ----------------------------------------------------

            for entry in sensor_data:

                if not isinstance(entry, dict):
                    continue

                variable = entry.get("variable", {})

                if not isinstance(variable, dict):
                    continue

                if variable.get("name") != "Temperature":
                    continue

                unit = variable.get("unit")

                measurements = entry.get("values", [])

                if not isinstance(measurements, list):
                    continue

                # ------------------------------------------------
                # 6. Extract Temperature measurements
                # ------------------------------------------------

                for measurement in measurements:

                    if not isinstance(measurement, dict):
                        skipped_invalid_measurements += 1
                        continue

                    timestamp = measurement.get("time")
                    value = measurement.get("value")

                    if timestamp is None or value is None:
                        skipped_invalid_measurements += 1
                        continue

                    all_temperature_records.append({
                        "sensorid": sensor_id,
                        "timestamp": timestamp,
                        "temperature": value,
                        "unit": unit
                    })

    # ------------------------------------------------------------
    # 7. Convert extracted records to DataFrame
    # ------------------------------------------------------------

    df = pd.DataFrame(all_temperature_records)

    print()
    print("Raw extracted Temperature records:")
    print(df.shape)

    # ------------------------------------------------------------
    # 8. Convert Unix timestamp to Melbourne date/time
    # ------------------------------------------------------------

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        unit="s",
        utc=True
    )

    df["timestamp"] = (
        df["timestamp"]
        .dt.tz_convert("Australia/Melbourne")
    )

    # ------------------------------------------------------------
    # 9. Ensure temperature is numeric
    # ------------------------------------------------------------

    df["temperature"] = pd.to_numeric(
        df["temperature"],
        errors="coerce"
    )

    before_invalid_numeric = len(df)

    df = df.dropna(
        subset=["temperature"]
    )

    invalid_numeric_removed = (
        before_invalid_numeric - len(df)
    )

    # ------------------------------------------------------------
    # 10. Remove duplicated measurements
    # ------------------------------------------------------------

    before_duplicates = len(df)

    df = df.drop_duplicates(
        subset=[
            "sensorid",
            "timestamp"
        ]
    )

    after_duplicates = len(df)

    duplicates_removed = (
        before_duplicates
        - after_duplicates
    )

    # ------------------------------------------------------------
    # 11. Sort data
    # ------------------------------------------------------------

    df = df.sort_values(
        [
            "sensorid",
            "timestamp"
        ]
    ).reset_index(drop=True)

    # ------------------------------------------------------------
    # 12. Print cleaning summary
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("CLEANING SUMMARY")
    print("===================================")

    print(
        f"Final number of rows: {len(df)}"
    )

    print(
        f"Duplicates removed: "
        f"{duplicates_removed}"
    )

    print(
        f"Invalid measurements skipped: "
        f"{skipped_invalid_measurements}"
    )

    print(
        f"Non-numeric temperatures removed: "
        f"{invalid_numeric_removed}"
    )

    print()
    print("Number of measurements for each sensor:")
    print(
        df["sensorid"].value_counts()
    )

    print()
    print("Temperature range:")
    print(
        df["temperature"].describe()
    )

    print()
    print("First 10 cleaned rows:")
    print(
        df.head(10)
    )

    # ------------------------------------------------------------
    # 13. Show time range for each sensor
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("TIME RANGE FOR EACH SENSOR")
    print("===================================")

    sensor_summary = (
        df.groupby("sensorid")
        .agg(
            first_measurement=(
                "timestamp",
                "min"
            ),
            last_measurement=(
                "timestamp",
                "max"
            ),
            number_of_measurements=(
                "temperature",
                "count"
            )
        )
    )

    print(sensor_summary)

    # ------------------------------------------------------------
    # 14. Save cleaned dataset
    # ------------------------------------------------------------

    df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Clean data saved as: "
        f"{OUTPUT_FILE}"
    )

    # ------------------------------------------------------------
    # 15. Finish timer
    # ------------------------------------------------------------

    elapsed_time = (
        time.time()
        - start_time
    )

    print()
    print(
        f"Total processing time: "
        f"{elapsed_time:.2f} seconds"
    )

    print("Finished!")


if __name__ == "__main__":
    main()