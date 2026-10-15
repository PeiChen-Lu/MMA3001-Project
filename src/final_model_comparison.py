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

# Validation results written by decision_tree_optimisation.py.
DEPTH_RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "csv"
    / "decision_tree_depth_results.csv"
)

FEATURE_COLUMNS = [
    "temp_minus_20",
    "temp_minus_10",
    "temp_plus_10",
    "temp_plus_20"
]

TARGET_COLUMN = "target_temperature"


def load_ml_dataset(path=INPUT_FILE):
    """
    Load the machine-learning dataset in chronological order.

    Parameters
    ----------
    path : pathlib.Path, optional
        CSV file created by ``create_ml_dataset.py``.

    Returns
    -------
    pandas.DataFrame
        Examples sorted by ``timestamp`` (Australia/Melbourne time).
    """

    df = pd.read_csv(path)

    df["timestamp"] = (
        pd.to_datetime(df["timestamp"], utc=True)
        .dt.tz_convert("Australia/Melbourne")
    )

    return df.sort_values("timestamp").reset_index(drop=True)


def load_best_depth(results_file=DEPTH_RESULTS_FILE):
    """
    Read the Decision Tree depth with the lowest validation RMSE.

    Parameters
    ----------
    results_file : pathlib.Path, optional
        CSV written by ``decision_tree_optimisation.py`` with columns
        ``max_depth`` and ``validation_rmse``. An empty ``max_depth``
        means an unlimited tree.

    Returns
    -------
    int or None
        Best ``max_depth`` (``None`` means unlimited depth).

    Raises
    ------
    FileNotFoundError
        If the optimisation results have not been generated yet.
    """

    if not Path(results_file).exists():
        raise FileNotFoundError(
            f"{results_file} not found. "
            "Run decision_tree_optimisation.py first."
        )

    results = pd.read_csv(results_file)

    best_row = results.loc[
        results["validation_rmse"].idxmin()
    ]

    best_depth = best_row["max_depth"]

    if pd.isna(best_depth):
        return None

    return int(best_depth)


def split_development_test(
    df,
    boundary_fraction=0.80,
    purge_minutes=30
):
    """
    Split examples chronologically into development and test sets.

    Examples within ``purge_minutes`` of the boundary are discarded
    because each example uses neighbouring measurements up to 20
    minutes away; the gap prevents the two sets from sharing readings.

    Parameters
    ----------
    df : pandas.DataFrame
        Examples sorted by ``timestamp``.
    boundary_fraction : float, optional
        Fraction of examples (in time order) placed before the boundary.
    purge_minutes : float, optional
        Half-width of the discarded zone around the boundary. Must be
        at least 20 minutes.

    Returns
    -------
    development_df, test_df : pandas.DataFrame
        Earlier and later examples respectively.

    Raises
    ------
    ValueError
        If ``purge_minutes`` is below 20 or the fraction is not in (0, 1).
    """

    if purge_minutes < 20:
        raise ValueError("purge_minutes must be at least 20")

    if not 0 < boundary_fraction < 1:
        raise ValueError("boundary_fraction must be between 0 and 1")

    boundary = df.iloc[
        int(len(df) * boundary_fraction)
    ]["timestamp"]

    purge_time = pd.Timedelta(minutes=purge_minutes)

    development_df = df[
        df["timestamp"] <= boundary - purge_time
    ].copy()

    test_df = df[
        df["timestamp"] >= boundary + purge_time
    ].copy()

    return development_df, test_df


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
    # 1. Load dataset and chronological development/test split
    # ------------------------------------------------------------

    df = load_ml_dataset()

    development_df, test_df = split_development_test(df)

    best_tree_depth = load_best_depth()

    feature_columns = FEATURE_COLUMNS

    target_column = TARGET_COLUMN

    X_dev = development_df[feature_columns]

    y_dev = development_df[target_column]

    X_test = test_df[feature_columns]

    y_test = test_df[target_column]

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
            max_depth=best_tree_depth,
            random_state=42
        )
    )

    tree_result = evaluate_model(
        (
            "Decision Tree "
            f"(depth={best_tree_depth})"
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
            ].reset_index(drop=True),

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
            f"(depth={best_tree_depth})"
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