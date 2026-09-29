from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor


# ============================================================
# MMA3001 Project Tests
#
# These tests check:
# 1. Cleaned temperature data
# 2. Machine-learning dataset structure
# 3. Linear Regression output
# 4. Decision Tree output
# 5. Numerical validity of predictions
# ============================================================


# ------------------------------------------------------------
# Locate the project root folder
#
# test_project.py is inside:
# project/tests/test_project.py
#
# parents[1] takes us back to:
# project/
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]


TEMPERATURE_FILE = (
    PROJECT_ROOT / "sensor_0869_temperature.csv"
)

ML_FILE = (
    PROJECT_ROOT / "ml_temperature_dataset.csv"
)


# ------------------------------------------------------------
# Helper function: load ML dataset
# ------------------------------------------------------------

def load_ml_data():

    df = pd.read_csv(ML_FILE)

    features = [
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20"
    ]

    X = df[features]

    y = df["target_temperature"]

    return df, X, y


# ============================================================
# TEST 1
# Check that cleaned temperature data exists
# ============================================================

def test_temperature_file_exists():

    assert TEMPERATURE_FILE.exists()


# ============================================================
# TEST 2
# Check selected sensor data contains valid temperatures
# ============================================================

def test_temperature_data_is_valid():

    df = pd.read_csv(TEMPERATURE_FILE)

    assert len(df) > 0

    assert "timestamp" in df.columns

    assert "temperature" in df.columns

    assert df["temperature"].isna().sum() == 0

    assert np.isfinite(
        df["temperature"]
    ).all()


# ============================================================
# TEST 3
# Check ML dataset has the required columns
# ============================================================

def test_ml_dataset_columns():

    df = pd.read_csv(ML_FILE)

    required_columns = [
        "timestamp",
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20",
        "target_temperature"
    ]

    for column in required_columns:
        assert column in df.columns


# ============================================================
# TEST 4
# Check ML dataset contains no missing values
# ============================================================

def test_ml_dataset_has_no_missing_values():

    df = pd.read_csv(ML_FILE)

    required_columns = [
        "temp_minus_20",
        "temp_minus_10",
        "temp_plus_10",
        "temp_plus_20",
        "target_temperature"
    ]

    assert (
        df[required_columns]
        .isna()
        .sum()
        .sum()
        == 0
    )


# ============================================================
# TEST 5
# Check X and y have compatible dimensions
# ============================================================

def test_feature_target_dimensions():

    df, X, y = load_ml_data()

    assert len(X) == len(y)

    assert X.shape[1] == 4

    assert len(df) == 8677


# ============================================================
# TEST 6
# Check Linear Regression can train and predict
# ============================================================

def test_linear_regression_prediction():

    df, X, y = load_ml_data()

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]

    model = LinearRegression()

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)

    assert np.isfinite(
        predictions
    ).all()


# ============================================================
# TEST 7
# Check optimised Decision Tree configuration
# ============================================================

def test_decision_tree_depth():

    model = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )

    assert model.max_depth == 8


# ============================================================
# TEST 8
# Check Decision Tree can train and predict
# ============================================================

def test_decision_tree_prediction():

    df, X, y = load_ml_data()

    split_index = int(len(df) * 0.8)

    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]

    X_test = X.iloc[split_index:]

    model = DecisionTreeRegressor(
        max_depth=8,
        random_state=42
    )

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(X_test)

    assert len(predictions) == len(X_test)

    assert np.isfinite(
        predictions
    ).all()