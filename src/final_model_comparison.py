"""
Compare the final Linear Regression and optimised Decision Tree
models on an untouched chronological test dataset.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "ml_temperature_dataset.csv"
)

COMPARISON_FILE = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "final_model_comparison.csv"
)

PREDICTIONS_FILE = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "final_test_predictions.csv"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)

# Selected from the Decision Tree
# validation optimisation stage.
BEST_TREE_DEPTH = 8


def evaluate_model(
    name,
    model,
    X_dev,
    y_dev,
    X_test,
    y_test
):
    """
    Train and evaluate a regression model.

    Parameters
    ----------
    name : str
        Name of the model.
    model :
        Scikit-learn regression model.
    X_dev : pandas.DataFrame
        Development input features.
    y_dev : pandas.Series
        Development target values.
    X_test : pandas.DataFrame
        Final test input features.
    y_test : pandas.Series
        Final test target values.

    Returns
    -------
    dict
        Model performance metrics and predictions.
    """

    start_train = (
        time.perf_counter()
    )

    model.fit(
        X_dev,
        y_dev
    )

    training_time = (
        time.perf_counter()
        - start_train
    )

    start_predict = (
        time.perf_counter()
    )

    prediction = model.predict(
        X_test
    )

    prediction_time = (
        time.perf_counter()
        - start_predict
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    mse = mean_squared_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        prediction
    )

    return {
        "model": name,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "training_time":
            training_time,
        "prediction_time":
            prediction_time,
        "prediction":
            prediction
    }


def main():
    """Run the final model comparison."""
    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    COMPARISON_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 1. Load dataset
    # ------------------------------------------------------------

    df = pd.read_csv(
        INPUT_FILE
    )

    df["timestamp"] = (
        pd.to_datetime(
            df["timestamp"],
            utc=True
        )
        .dt.tz_convert(
            "Australia/Melbourne"
        )
    )

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    feature_columns = [
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20"
    ]

    target_column = (
        "target_temperature"
    )

    # ------------------------------------------------------------
    # 2. Final chronological development/test split
    # ------------------------------------------------------------

    n = len(df)

    test_boundary_index = int(
        n * 0.80
    )

    test_boundary = (
        df.iloc[
            test_boundary_index
        ]["timestamp"]
    )

    purge_time = pd.Timedelta(
        minutes=30
    )

    development_df = df[
        df["timestamp"]
        <= test_boundary - purge_time
    ].copy()

    test_df = df[
        df["timestamp"]
        >= test_boundary + purge_time
    ].copy()

    X_dev = development_df[
        feature_columns
    ]

    y_dev = development_df[
        target_column
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        target_column
    ]

    print("===================================")
    print("FINAL DATA SPLIT")
    print("===================================")

    print(
        f"Development examples: "
        f"{len(development_df)}"
    )

    print(
        f"Final test examples:  "
        f"{len(test_df)}"
    )

    print()
    print("Development period:")

    print(
        development_df[
            "timestamp"
        ].min(),
        "to",
        development_df[
            "timestamp"
        ].max()
    )

    print()
    print("Test period:")

    print(
        test_df[
            "timestamp"
        ].min(),
        "to",
        test_df[
            "timestamp"
        ].max()
    )

    # ------------------------------------------------------------
    # 3. Linear Regression
    # ------------------------------------------------------------

    linear_model = (
        LinearRegression()
    )

    linear_result = evaluate_model(
        "Linear Regression",
        linear_model,
        X_dev,
        y_dev,
        X_test,
        y_test
    )

    # ------------------------------------------------------------
    # 4. Optimised Decision Tree
    # ------------------------------------------------------------

    tree_model = (
        DecisionTreeRegressor(
            max_depth=BEST_TREE_DEPTH,
            random_state=42
        )
    )

    tree_result = evaluate_model(
        (
            "Decision Tree "
            f"(depth={BEST_TREE_DEPTH})"
        ),
        tree_model,
        X_dev,
        y_dev,
        X_test,
        y_test
    )

    # ------------------------------------------------------------
    # 5. Results comparison table
    # ------------------------------------------------------------

    comparison = pd.DataFrame([
        {
            "Model":
                linear_result["model"],
            "MAE_C":
                linear_result["mae"],
            "MSE_C2":
                linear_result["mse"],
            "RMSE_C":
                linear_result["rmse"],
            "R2":
                linear_result["r2"],
            "Training_Time_s":
                linear_result[
                    "training_time"
                ],
            "Prediction_Time_s":
                linear_result[
                    "prediction_time"
                ]
        },
        {
            "Model":
                tree_result["model"],
            "MAE_C":
                tree_result["mae"],
            "MSE_C2":
                tree_result["mse"],
            "RMSE_C":
                tree_result["rmse"],
            "R2":
                tree_result["r2"],
            "Training_Time_s":
                tree_result[
                    "training_time"
                ],
            "Prediction_Time_s":
                tree_result[
                    "prediction_time"
                ]
        }
    ])

    print()
    print("===================================")
    print("FINAL TEST RESULTS")
    print("===================================")

    print(
        comparison.to_string(
            index=False
        )
    )

    comparison.to_csv(
        COMPARISON_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # 6. Linear Regression coefficients
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("LINEAR REGRESSION COEFFICIENTS")
    print("===================================")

    for feature, coefficient in zip(
        feature_columns,
        linear_model.coef_
    ):

        print(
            f"{feature}: "
            f"{coefficient:.4f}"
        )

    print(
        f"Intercept: "
        f"{linear_model.intercept_:.4f}"
    )

    # ------------------------------------------------------------
    # 7. Decision Tree feature importance
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("DECISION TREE FEATURE IMPORTANCE")
    print("===================================")

    for feature, importance in zip(
        feature_columns,
        tree_model.feature_importances_
    ):

        print(
            f"{feature}: "
            f"{importance:.4f}"
        )

    # ------------------------------------------------------------
    # 8. Save detailed predictions
    # ------------------------------------------------------------

    prediction_df = pd.DataFrame({
        "timestamp":
            test_df[
                "timestamp"
            ].values,

        "actual_temperature":
            y_test.values,

        "linear_prediction":
            linear_result[
                "prediction"
            ],

        "tree_prediction":
            tree_result[
                "prediction"
            ]
    })

    prediction_df.to_csv(
        PREDICTIONS_FILE,
        index=False
    )

    # ------------------------------------------------------------
    # 9. Plot first 300 final-test examples
    # ------------------------------------------------------------

    plot_data = (
        prediction_df.head(300)
    )

    plt.figure(
        figsize=(12, 5)
    )

    plt.plot(
        plot_data["timestamp"],
        plot_data[
            "actual_temperature"
        ],
        label="Actual Temperature",
        linewidth=1.6
    )

    plt.plot(
        plot_data["timestamp"],
        plot_data[
            "linear_prediction"
        ],
        label="Linear Regression",
        linewidth=1.2
    )

    plt.plot(
        plot_data["timestamp"],
        plot_data[
            "tree_prediction"
        ],
        label=(
            "Decision Tree "
            f"(depth={BEST_TREE_DEPTH})"
        ),
        linewidth=1.2
    )

    plt.xlabel("Time")

    plt.ylabel(
        "Temperature (°C)"
    )

    plt.title(
        "Final Test: Actual vs "
        "Reconstructed Temperature"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "final_model_comparison.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("Stage 7 finished!")


if __name__ == "__main__":
    main()