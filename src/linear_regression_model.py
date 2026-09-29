"""
Train and evaluate a Linear Regression model for reconstructing
missing temperature sensor measurements.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# MMA3001 Project
# Stage 4: Linear Regression
#
# Purpose:
# Reconstruct a missing temperature measurement using
# surrounding temperature measurements.
#
# Inputs:
# T(t-20), T(t-10), T(t+10), T(t+20)
#
# Output:
# Predicted T(t)
# ============================================================


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_temperature_dataset.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "linear_regression_predictions.csv"
)

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------
# 1. Load machine-learning dataset
# ------------------------------------------------------------

def main():
    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("ML dataset loaded.")
    print(f"Total ML examples: {len(df)}")


    # ------------------------------------------------------------
    # 2. Define input features X and target y
    # ------------------------------------------------------------

    feature_columns = [
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20"
    ]

    target_column = "target_temperature"


    # ------------------------------------------------------------
    # 3. Chronological Train/Test split
    #
    # First ~80% of time:
    #     Training data
    #
    # Last ~20% of time:
    #     Testing data
    #
    # A small 20-minute separation is used around the boundary.
    # ------------------------------------------------------------

    split_index = int(len(df) * 0.80)

    cutoff_time = df.iloc[split_index]["timestamp"]

    purge_time = pd.Timedelta(minutes=20)

    train_df = df[
        df["timestamp"] <= cutoff_time - purge_time
    ].copy()

    test_df = df[
        df["timestamp"] >= cutoff_time + purge_time
    ].copy()


    print()
    print("===================================")
    print("TRAIN / TEST SPLIT")
    print("===================================")

    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples:  {len(test_df)}")

    print()
    print(f"Training period:")
    print(f"{train_df['timestamp'].min()}  to  {train_df['timestamp'].max()}")

    print()
    print(f"Testing period:")
    print(f"{test_df['timestamp'].min()}  to  {test_df['timestamp'].max()}")


    # ------------------------------------------------------------
    # 4. Create X_train, y_train, X_test, y_test
    # ------------------------------------------------------------

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]


    print()
    print("===================================")
    print("DATA SHAPES")
    print("===================================")

    print(f"X_train: {X_train.shape}")
    print(f"y_train: {y_train.shape}")

    print(f"X_test:  {X_test.shape}")
    print(f"y_test:  {y_test.shape}")


    # ------------------------------------------------------------
    # 5. Create Linear Regression model
    # ------------------------------------------------------------

    model = LinearRegression()


    # ------------------------------------------------------------
    # 6. Train model
    # ------------------------------------------------------------

    start_train = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_train


    # ------------------------------------------------------------
    # 7. Make predictions on TEST data
    # ------------------------------------------------------------

    start_predict = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start_predict


    # ------------------------------------------------------------
    # 8. Calculate evaluation metrics
    # ------------------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)


    print()
    print("===================================")
    print("LINEAR REGRESSION RESULTS")
    print("===================================")

    print(f"MAE:  {mae:.4f} °C")
    print(f"MSE:  {mse:.4f} degC^2")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"R²:   {r2:.4f}")

    print()
    print(f"Training time:   {training_time:.6f} seconds")
    print(f"Prediction time: {prediction_time:.6f} seconds")


    # ------------------------------------------------------------
    # 9. Show model coefficients
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("MODEL COEFFICIENTS")
    print("===================================")

    for feature, coefficient in zip(
        feature_columns,
        model.coef_
    ):
        print(f"{feature}: {coefficient:.4f}")

    print(f"Intercept: {model.intercept_:.4f}")


    # ------------------------------------------------------------
    # 10. Create result table
    # ------------------------------------------------------------

    results = pd.DataFrame({
        "timestamp": test_df["timestamp"].values,
        "actual_temperature": y_test.values,
        "predicted_temperature": y_pred
    })

    results["error"] = (
        results["predicted_temperature"]
        - results["actual_temperature"]
    )

    results["absolute_error"] = abs(results["error"])


    print()
    print("===================================")
    print("FIRST 10 PREDICTIONS")
    print("===================================")

    print(results.head(10))


    # ------------------------------------------------------------
    # 11. Save predictions
    # ------------------------------------------------------------

    results.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(f"Predictions saved as: {OUTPUT_FILE}")

    # ------------------------------------------------------------
    # 12. Plot Actual vs Predicted temperatures
    #
    # Only show first 300 test examples so that the graph
    # remains readable.
    # ------------------------------------------------------------

    plot_data = results.head(300)

    plt.figure(figsize=(12, 5))

    plt.plot(
        plot_data["timestamp"],
        plot_data["actual_temperature"],
        label="Actual Temperature",
        linewidth=1.5
    )

    plt.plot(
        plot_data["timestamp"],
        plot_data["predicted_temperature"],
        label="Linear Regression Prediction",
        linewidth=1.2
    )

    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    plt.title(
        "Actual vs Reconstructed Temperature "
        "- Linear Regression"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    print()
    print("Stage 4 finished!")
if __name__ == "__main__":
    main()