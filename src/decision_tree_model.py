"""
Train and evaluate a Decision Tree Regression model for reconstructing
missing temperature sensor measurements.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

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
    / "decision_tree_predictions.csv"
)


def main():
    """Run the Decision Tree reconstruction workflow."""

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # 你原来从 df = pd.read_csv(INPUT_FILE)
    # 开始的所有代码
    # 全部放进这里。



# ============================================================
# MMA3001 Project
# Stage 5: Decision Tree Regression
# ============================================================



# ------------------------------------------------------------
# 1. Load dataset
# ------------------------------------------------------------

    df = pd.read_csv(INPUT_FILE)

    df["timestamp"] = (
    pd.to_datetime(df["timestamp"], utc=True)
    .dt.tz_convert("Australia/Melbourne")
)

    df = df.sort_values("timestamp").reset_index(drop=True)

    print("ML dataset loaded.")
    print(f"Total ML examples: {len(df)}")


    # ------------------------------------------------------------
    # 2. Define features and target
    # ------------------------------------------------------------

    feature_columns = [
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20"
    ]

    target_column = "target_temperature"


    # ------------------------------------------------------------
    # 3. Use the SAME chronological train/test split
    #
    # Important:
    # We must use the same split as Linear Regression
    # so that the comparison is fair.
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


    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]


    print()
    print("===================================")
    print("TRAIN / TEST SPLIT")
    print("===================================")

    print(f"Training examples: {len(train_df)}")
    print(f"Testing examples:  {len(test_df)}")


    # ------------------------------------------------------------
    # 4. Create Decision Tree model
    #
    # max_depth controls tree complexity.
    #
    # We begin with a simple tree so that the model does not
    # become unnecessarily complex.
    # ------------------------------------------------------------

    model = DecisionTreeRegressor(
        max_depth=5,
        random_state=42
    )


    # ------------------------------------------------------------
    # 5. Train model
    # ------------------------------------------------------------

    start_train = time.perf_counter()

    model.fit(X_train, y_train)

    training_time = time.perf_counter() - start_train


    # ------------------------------------------------------------
    # 6. Prediction
    # ------------------------------------------------------------

    start_predict = time.perf_counter()

    y_pred = model.predict(X_test)

    prediction_time = time.perf_counter() - start_predict


    # ------------------------------------------------------------
    # 7. Evaluation metrics
    # ------------------------------------------------------------

    mae = mean_absolute_error(y_test, y_pred)

    mse = mean_squared_error(y_test, y_pred)

    rmse = np.sqrt(mse)

    r2 = r2_score(y_test, y_pred)


    print()
    print("===================================")
    print("DECISION TREE RESULTS")
    print("===================================")

    print(f"MAE:  {mae:.4f} °C")
    print(f"MSE:  {mse:.4f} degC^2")
    print(f"RMSE: {rmse:.4f} °C")
    print(f"R²:   {r2:.4f}")

    print()
    print(f"Training time:   {training_time:.6f} seconds")
    print(f"Prediction time: {prediction_time:.6f} seconds")


    # ------------------------------------------------------------
    # 8. Feature importance
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("FEATURE IMPORTANCE")
    print("===================================")

    for feature, importance in zip(
        feature_columns,
        model.feature_importances_
    ):
        print(f"{feature}: {importance:.4f}")


    # ------------------------------------------------------------
    # 9. Save predictions
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


    results.to_csv(
        OUTPUT_FILE,
        index=False
    )


    # ------------------------------------------------------------
    # 10. Plot
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
        label="Decision Tree Prediction",
        linewidth=1.2
    )

    plt.xlabel("Time")
    plt.ylabel("Temperature (°C)")

    plt.title(
        "Actual vs Reconstructed Temperature "
        "- Decision Tree Regression"
    )

    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


    print()
    print("Stage 5 finished!")

if __name__ == "__main__":
    main()
