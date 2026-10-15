"""
Optimise Decision Tree maximum depth using chronological training,
validation and testing datasets.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error


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
    / "decision_tree_depth_results.csv"
)
FIGURE_DIR = (
    PROJECT_ROOT
    / "results"
    / "figures"
)


def main():
    """Optimise the Decision Tree maximum depth."""

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------
    # 1. Load dataset
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

    df = (
        df.sort_values("timestamp")
        .reset_index(drop=True)
    )

    print("ML dataset loaded.")
    print(
        f"Total examples: {len(df)}"
    )

    # ------------------------------------------------------------
    # 2. Define features and target
    # ------------------------------------------------------------

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
    # 3. Chronological 60 / 20 / 20 split
    # ------------------------------------------------------------

    n = len(df)

    train_end_index = int(
        n * 0.60
    )

    validation_end_index = int(
        n * 0.80
    )

    train_cutoff = (
        df.iloc[
            train_end_index
        ]["timestamp"]
    )

    validation_cutoff = (
        df.iloc[
            validation_end_index
        ]["timestamp"]
    )

    # Each ML example uses measurements up to
    # 20 minutes before and after the target.
    # A 30-minute purge reduces overlap
    # between adjacent datasets.

    purge_time = pd.Timedelta(
        minutes=30
    )

    train_df = df[
        df["timestamp"]
        <= train_cutoff - purge_time
    ].copy()

    validation_df = df[
        (
            df["timestamp"]
            >= train_cutoff + purge_time
        )
        &
        (
            df["timestamp"]
            <= validation_cutoff - purge_time
        )
    ].copy()

    test_df = df[
        df["timestamp"]
        >= validation_cutoff + purge_time
    ].copy()

    print()
    print("===================================")
    print("TRAIN / VALIDATION / TEST SPLIT")
    print("===================================")

    print(
        f"Training examples:   "
        f"{len(train_df)}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_df)}"
    )

    print(
        f"Testing examples:    "
        f"{len(test_df)}"
    )

    print()
    print("Training period:")

    print(
        train_df["timestamp"].min(),
        "to",
        train_df["timestamp"].max()
    )

    print()
    print("Validation period:")

    print(
        validation_df["timestamp"].min(),
        "to",
        validation_df["timestamp"].max()
    )

    print()
    print("Testing period:")

    print(
        test_df["timestamp"].min(),
        "to",
        test_df["timestamp"].max()
    )

    # ------------------------------------------------------------
    # 4. Create X and y
    # ------------------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        target_column
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        target_column
    ]

    # The final test dataset is intentionally
    # not used during hyperparameter selection.

    # ------------------------------------------------------------
    # 5. Depths to investigate
    # ------------------------------------------------------------

    depths = [
        2,
        3,
        4,
        5,
        6,
        8,
        10,
        15,
        None
    ]

    results = []

    # ------------------------------------------------------------
    # 6. Train one model for each maximum depth
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("DECISION TREE DEPTH TEST")
    print("===================================")

    for depth in depths:

        model = DecisionTreeRegressor(
            max_depth=depth,
            random_state=42
        )

        start_time = (
            time.perf_counter()
        )

        model.fit(
            X_train,
            y_train
        )

        training_time = (
            time.perf_counter()
            - start_time
        )

        train_prediction = (
            model.predict(X_train)
        )

        validation_prediction = (
            model.predict(
                X_validation
            )
        )

        train_rmse = np.sqrt(
            mean_squared_error(
                y_train,
                train_prediction
            )
        )

        validation_rmse = np.sqrt(
            mean_squared_error(
                y_validation,
                validation_prediction
            )
        )

        results.append({
            "max_depth": depth,
            "train_rmse": train_rmse,
            "validation_rmse":
                validation_rmse,
            "training_time":
                training_time
        })

        print(
            f"Depth = "
            f"{str(depth):>4} | "
            f"Train RMSE = "
            f"{train_rmse:.4f} °C | "
            f"Validation RMSE = "
            f"{validation_rmse:.4f} °C | "
            f"Time = "
            f"{training_time:.6f} s"
        )

    # ------------------------------------------------------------
    # 7. Convert results to DataFrame
    # ------------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # ------------------------------------------------------------
    # 8. Find best depth
    # ------------------------------------------------------------

    best_row = results_df.loc[
        results_df[
            "validation_rmse"
        ].idxmin()
    ]

    best_depth_value = (
        best_row["max_depth"]
    )

    if pd.isna(best_depth_value):
        best_depth = None
    else:
        best_depth = int(
            best_depth_value
        )

    best_validation_rmse = (
        best_row[
            "validation_rmse"
        ]
    )

    print()
    print("===================================")
    print("BEST MODEL")
    print("===================================")

    print(
        f"Best max_depth: "
        f"{best_depth}"
    )

    print(
        f"Lowest Validation RMSE: "
        f"{best_validation_rmse:.4f} °C"
    )

    # ------------------------------------------------------------
    # 9. Show full results table
    # ------------------------------------------------------------

    print()
    print("===================================")
    print("FULL RESULTS")
    print("===================================")

    print(results_df)

    # ------------------------------------------------------------
    # 10. Save results
    # ------------------------------------------------------------

    results_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print(
        f"Saved as: "
        f"{OUTPUT_FILE}"
    )

    # ------------------------------------------------------------
    # 11. Plot training and validation RMSE
    # ------------------------------------------------------------

    depth_labels = [
        str(depth)
        for depth in depths
    ]

    x = np.arange(
        len(depth_labels)
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.plot(
        x,
        results_df["train_rmse"],
        marker="o",
        label="Training RMSE"
    )

    plt.plot(
        x,
        results_df[
            "validation_rmse"
        ],
        marker="o",
        label="Validation RMSE"
    )

    plt.xticks(
        x,
        depth_labels
    )

    plt.xlabel(
        "Decision Tree max_depth"
    )

    plt.ylabel(
        "RMSE (°C)"
    )

    plt.title(
        "Decision Tree Depth Optimisation"
    )

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "decision_tree_depth_optimisation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()

    print()
    print("Stage 6 finished.")

    print(
        "The final test set has "
        "NOT been used for "
        "hyperparameter selection."
    )


if __name__ == "__main__":
    main()