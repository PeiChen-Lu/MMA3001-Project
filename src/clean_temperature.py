from pathlib import Path
import pandas as pd
import json
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = PROJECT_ROOT / "5EnvSensor_MayToDec2024_180kRows.csv"

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clean_temperature.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ============================================================
# MMA3001 Project
# Stage 1: Extract and clean Temperature sensor data
# ============================================================




# ------------------------------------------------------------
# 1. Start timer
# ------------------------------------------------------------
start_time = time.time()

print("Starting data cleaning...")


# ------------------------------------------------------------
# 2. Read the large CSV in chunks
#
# We only need:
#   sensorid
#   jsondata
#
# chunksize means Python does NOT load the whole 353 MB file
# into memory at once.
# ------------------------------------------------------------
chunk_size = 20000

all_temperature_records = []

chunk_number = 0


for chunk in pd.read_csv(
    INPUT_FILE,
    usecols=["sensorid", "jsondata"],
    chunksize=chunk_size
):

    chunk_number += 1

    print(f"Processing chunk {chunk_number}...")


    # --------------------------------------------------------
    # 3. Go through each raw row
    # --------------------------------------------------------
    for row in chunk.itertuples(index=False):

        sensor_id = row.sensorid

        try:
            sensor_data = json.loads(row.jsondata)

        except (json.JSONDecodeError, TypeError):
            # If the JSON is broken, skip this row
            continue


        # ----------------------------------------------------
        # 4. Look through the variables inside jsondata
        #
        # Examples:
        # Battery
        # Carbon dioxide
        # Humidity
        # Temperature
        # Pressure
        # etc.
        #
        # For now we ONLY want Temperature.
        # ----------------------------------------------------
        for entry in sensor_data:

            variable_name = entry["variable"]["name"]

            if variable_name != "Temperature":
                continue


            unit = entry["variable"]["unit"]


            # ------------------------------------------------
            # 5. Extract every Temperature reading
            # ------------------------------------------------
            for measurement in entry["values"]:

                all_temperature_records.append({
                    "sensorid": sensor_id,
                    "timestamp": measurement["time"],
                    "temperature": measurement["value"],
                    "unit": unit
                })


# ------------------------------------------------------------
# 6. Convert the extracted data into a DataFrame
# ------------------------------------------------------------
df = pd.DataFrame(all_temperature_records)

print()
print("Raw extracted Temperature records:")
print(df.shape)


# ------------------------------------------------------------
# 7. Convert Unix timestamp into Melbourne date/time
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
# 8. Make sure temperature is numeric
# ------------------------------------------------------------
df["temperature"] = pd.to_numeric(
    df["temperature"],
    errors="coerce"
)


# Remove rows where temperature could not be converted
df = df.dropna(subset=["temperature"])


# ------------------------------------------------------------
# 9. Remove duplicated sensor measurements
#
# Same sensor + same timestamp should normally represent
# the same physical measurement.
# ------------------------------------------------------------
before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["sensorid", "timestamp"]
)

after_duplicates = len(df)

duplicates_removed = before_duplicates - after_duplicates


# ------------------------------------------------------------
# 10. Sort the data
# ------------------------------------------------------------
df = df.sort_values(
    ["sensorid", "timestamp"]
).reset_index(drop=True)


# ------------------------------------------------------------
# 11. Print useful information
# ------------------------------------------------------------
print()
print("===================================")
print("CLEANING SUMMARY")
print("===================================")

print(f"Final number of rows: {len(df)}")

print(f"Duplicates removed: {duplicates_removed}")

print()
print("Number of measurements for each sensor:")
print(df["sensorid"].value_counts())

print()
print("Temperature range:")
print(df["temperature"].describe())


print()
print("First 10 cleaned rows:")
print(df.head(10))


# ------------------------------------------------------------
# 12. Show time range for each sensor
# ------------------------------------------------------------
print()
print("===================================")
print("TIME RANGE FOR EACH SENSOR")
print("===================================")

sensor_summary = df.groupby("sensorid").agg(
    first_measurement=("timestamp", "min"),
    last_measurement=("timestamp", "max"),
    number_of_measurements=("temperature", "count")
)

print(sensor_summary)


# ------------------------------------------------------------
# 13. Save cleaned data
# ------------------------------------------------------------
df.to_csv(
    OUTPUT_FILE,
    index=False
)

print()
print(f"Clean data saved as: {OUTPUT_FILE}")


# ------------------------------------------------------------
# 14. Finish timer
# ------------------------------------------------------------
elapsed_time = time.time() - start_time

print()
print(f"Total processing time: {elapsed_time:.2f} seconds")
print("Finished!")