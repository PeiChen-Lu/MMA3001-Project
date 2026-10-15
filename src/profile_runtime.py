"""
Profile the runtime of the temperature-reconstruction workflow.

Two analyses are produced:

1. A ``cProfile`` report of the data-cleaning stage, which reads the
   353 MB raw CSV in chunks and parses one JSON string per row. This
   is the slowest stage of the project. It runs only if the raw file
   is available in ``data/raw/``.
2. Fit and predict times (median of repeated runs) of Linear
   Regression and the Decision Tree for increasing numbers of
   training examples, to show how each model scales.

Outputs are written to ``results/profile/`` and ``results/csv/``.
"""

import contextlib
import cProfile
import io
import pstats
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor

try:
    from src import clean_temperature
    from src.final_model_comparison import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        load_best_depth,
        load_ml_dataset,
        split_development_test,
    )
except ModuleNotFoundError:
    # Allows ``python src/profile_runtime.py`` as well as
    # ``python -m src.profile_runtime``.
    import clean_temperature
    from final_model_comparison import (
        FEATURE_COLUMNS,
        TARGET_COLUMN,
        load_best_depth,
        load_ml_dataset,
        split_development_test,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROFILE_DIR = PROJECT_ROOT / "results" / "profile"

TIMING_FILE = (
    PROJECT_ROOT / "results" / "csv" / "model_timing_scaling.csv"
)


def profile_function(func, top_n=15):
    """
    Run ``func`` under ``cProfile`` and summarise the slowest calls.

    Parameters
    ----------
    func : callable
        Function with no arguments. Its printed output is discarded.
    top_n : int, optional
        Number of rows of the profile (sorted by cumulative time).

    Returns
    -------
    str
        Text table produced by ``pstats``.
    """

    profiler = cProfile.Profile()

    with contextlib.redirect_stdout(io.StringIO()):
        profiler.enable()
        func()
        profiler.disable()

    stream = io.StringIO()

    pstats.Stats(
        profiler,
        stream=stream
    ).sort_stats("cumulative").print_stats(top_n)

    return stream.getvalue()


def median_time(func, repeats=20):
    """
    Return the median wall-clock time of ``repeats`` calls to ``func``.

    Parameters
    ----------
    func : callable
        Function with no arguments.
    repeats : int, optional
        Number of repetitions.

    Returns
    -------
    float
        Median time in seconds.
    """

    times = []

    for _ in range(repeats):
        start = time.perf_counter()
        func()
        times.append(time.perf_counter() - start)

    return float(np.median(times))


def time_models(
    development_df,
    test_df,
    tree_depth,
    fractions=(0.25, 0.50, 0.75, 1.0),
    repeats=20
):
    """
    Measure fit and predict time versus number of training examples.

    Parameters
    ----------
    development_df : pandas.DataFrame
        Training examples (chronological order).
    test_df : pandas.DataFrame
        Examples used for the prediction timing.
    tree_depth : int or None
        ``max_depth`` of the Decision Tree.
    fractions : sequence of float, optional
        Fractions of the development set used for training.
    repeats : int, optional
        Repetitions for each median time.

    Returns
    -------
    pandas.DataFrame
        One row per model and training fraction, with times in
        milliseconds.
    """

    models = {
        "Linear Regression": LinearRegression(),
        f"Decision Tree (depth={tree_depth})": DecisionTreeRegressor(
            max_depth=tree_depth,
            random_state=42
        )
    }

    X_test = test_df[FEATURE_COLUMNS]

    rows = []

    for fraction in fractions:

        subset = development_df.iloc[
            : int(len(development_df) * fraction)
        ]

        X_train = subset[FEATURE_COLUMNS]
        y_train = subset[TARGET_COLUMN]

        for name, model in models.items():

            fit_time = median_time(
                lambda m=model: clone(m).fit(X_train, y_train),
                repeats
            )

            fitted = clone(model).fit(X_train, y_train)

            predict_time = median_time(
                lambda f=fitted: f.predict(X_test),
                repeats
            )

            rows.append({
                "Model": name,
                "Training_Fraction": fraction,
                "Training_Examples": len(subset),
                "Fit_Time_ms": fit_time * 1000,
                "Predict_Time_ms": predict_time * 1000
            })

    return pd.DataFrame(rows)


def main():
    """Profile data cleaning and time the regression models."""

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    TIMING_FILE.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 1. Profile the data-cleaning stage (needs the raw CSV)
    # ------------------------------------------------------------

    if clean_temperature.INPUT_FILE.exists():

        report = profile_function(clean_temperature.main)

        report_file = PROFILE_DIR / "clean_temperature_profile.txt"

        report_file.write_text(report, encoding="utf-8")

        print(report)
        print(f"Profile saved as: {report_file}")

    else:
        print(
            "Raw data not found "
            f"({clean_temperature.INPUT_FILE}); "
            "skipping the cleaning profile."
        )

    # ------------------------------------------------------------
    # 2. Time the models for increasing training-set size
    # ------------------------------------------------------------

    df = load_ml_dataset()

    development_df, test_df = split_development_test(df)

    timing = time_models(
        development_df,
        test_df,
        load_best_depth()
    )

    timing.to_csv(TIMING_FILE, index=False)

    print()
    print(timing.to_string(index=False))
    print()
    print(f"Timing table saved as: {TIMING_FILE}")


if __name__ == "__main__":
    main()