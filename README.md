# Reconstruction of Mi# Reconstruction of Missing Temperature Sensor Data Using Machine Learning Regression

MMA3001 Numerical Methods and Machine Learning - Individual Project.

This project reconstructs an isolated missing temperature measurement from a building environmental sensor using measurements recorded immediately before and after the missing point.

Two regression approaches are compared:

- **Linear Regression** - baseline model
- **Decision Tree Regression** - alternative model

The original measured temperature is used as the **ground-truth reference** for evaluating reconstruction accuracy.

---

## 1. Engineering Problem

Building monitoring systems collect environmental measurements such as temperature, humidity and CO2 for applications including HVAC monitoring, thermal-comfort analysis and building energy assessment.

Sensor measurements may occasionally be unavailable because of communication, logging or recording problems. Missing measurements reduce the completeness and reliability of historical datasets used for later engineering analysis.

The selected sensor normally records approximately every 10 minutes. Approximately **89.9% of adjacent sampling intervals are around 10 minutes**, while approximately **10.1% are around 20 minutes**. These longer intervals are consistent with occasional missing scheduled measurements.

### Task

The task is to reconstruct one isolated missing temperature measurement at time `t` using four surrounding measurements:

- `T(t-20)`
- `T(t-10)`
- `T(t+10)`
- `T(t+20)`

The model output is the reconstructed temperature `T(t)`.

This is an **offline reconstruction problem rather than a forecasting problem**, because measurements after the missing point are already available.

### Intended Use

The computational workflow is intended to reconstruct isolated missing measurements in historical temperature datasets before further engineering analysis.

### Scope

The project is limited to:

- one environmental sensor;
- one variable, temperature;
- isolated single missing measurements;
- approximately 10-minute sampling intervals;
- cases where two measurements before and two measurements after the missing point are available.

Longer missing sequences are not reconstructed.

---

## 2. Inputs and Outputs

| Item | Description |
| --- | --- |
| **Top-level input** | Raw environmental sensor CSV `5EnvSensor_MayToDec2024_180kRows.csv` |
| **Model inputs** | `temp_minus_20`, `temp_minus_10`, `temp_plus_10`, `temp_plus_20` |
| **Input units** | Degrees Celsius |
| **Model target** | `target_temperature`, representing `T(t)` |
| **Model output** | Reconstructed temperature at the missing time |
| **Valid reconstruction window** | All four adjacent time intervals must be between 9 and 11 minutes |
| **Invalid data handling** | Missing timestamps or values, malformed JSON records and non-numeric temperatures are skipped or removed during cleaning |
| **Duplicate handling** | Duplicate `(sensorid, timestamp)` records are removed |
| **Main result files** | `results/csv/final_model_comparison.csv` and `results/csv/final_test_predictions.csv` |
| **Figures** | Stored in `results/figures/` |

Timestamps are converted to the `Australia/Melbourne` timezone.

The dataset crosses the end of daylight saving time, so timestamps include both `+11:00` and `+10:00` offsets. Timestamp strings are therefore parsed using UTC before conversion to Melbourne local time.

---

## 3. Data and Sensor Selection

The environmental sensor dataset was supplied by the MMA3001 teaching team.

The original raw CSV is approximately **353 MB**. After extracting temperature measurements from the raw JSON records, the cleaning stage produced:

- Raw extracted temperature records: **179,447**
- Duplicate measurements removed: **136,604**
- Final cleaned measurements: **42,843**

The raw course dataset is not redistributed in this repository.

### Sensor Selection

The two sensors containing substantial temperature histories were:

| Sensor ID | Number of measurements |
| --- | ---: |
| `6012002000326` | 29,039 |
| `6012002000869` | 13,800 |

Although sensor `6012002000326` contains more total measurements, its records span a much wider period from 2023 to 2026.

Sensor `6012002000869` was selected because it provides a sufficiently large and relatively continuous record over a clearly defined period from February to June 2024, with a strong approximately 10-minute sampling pattern.

This makes sensor `6012002000869` more suitable for constructing continuous neighbouring-measurement windows for the missing-temperature reconstruction problem.

For the selected sensor:

- Number of measurements: **13,800**
- First measurement: **29 February 2024**
- Last measurement: **14 June 2024**
- Mean temperature: approximately **23.09 °C**
- Minimum temperature: **19.0 °C**
- Maximum temperature: **28.5 °C**

### Sampling Intervals

There are 13,799 adjacent intervals for the selected sensor.

| Approximate interval | Count |
| --- | ---: |
| 10 minutes | 12,409 |
| 20 minutes | 1,389 |
| 30 minutes | 1 |

Approximately **10.1% of adjacent intervals are around 20 minutes**, which is consistent with occasional missing scheduled measurements.

This percentage is not treated as a directly measured missing-data rate because the reason for each longer interval is not independently known.

### Machine-Learning Dataset

A reconstruction example is created only when the following five measurements form a continuous sequence:

```text
T(t-20), T(t-10), T(t), T(t+10), T(t+20)
```

Every adjacent interval within the sequence must be between 9 and 11 minutes.

This produces:

```text
8,677 usable machine-learning examples
```

Each example contains four input temperatures and one target temperature.

---

## 4. Computational Method

The complete workflow is implemented in Python.

### Stage 1 - Data Cleaning

`src/clean_temperature.py`

The raw CSV is read in chunks because of its large file size.

The cleaning process:

1. reads the raw CSV in chunks;
2. parses the JSON data;
3. extracts Temperature measurements;
4. converts timestamps to Melbourne local time;
5. converts temperature values to numeric form;
6. removes duplicate `(sensorid, timestamp)` measurements;
7. sorts the cleaned measurements;
8. saves the cleaned dataset.

The main cleaned dataset is:

```text
data/processed/clean_temperature.csv
```

### Stage 2 - Sensor Analysis

`src/analyse_temperature.py`

The selected sensor is analysed for:

- number of measurements;
- measurement period;
- temperature statistics;
- sampling intervals;
- frequency of approximately 10-, 20- and 30-minute gaps.

The selected sensor dataset is saved as:

```text
data/processed/sensor_0869_temperature.csv
```

### Stage 3 - Reconstruction Window Generation

`src/create_ml_dataset.py`

The function:

```python
build_windows()
```

creates machine-learning reconstruction examples.

The inputs are:

```text
T(t-20)
T(t-10)
T(t+10)
T(t+20)
```

and the prediction target is:

```text
T(t)
```

Any candidate window containing an adjacent interval outside the accepted 9-11 minute range is excluded.

The resulting dataset is:

```text
data/processed/ml_temperature_dataset.csv
```

### Stage 4 - Linear Regression

`src/linear_regression_model.py`

Linear Regression is used as the **baseline regression model**.

It was selected because it is:

- simple;
- computationally inexpensive;
- easy to interpret;
- suitable for identifying whether surrounding temperatures have an approximately linear relationship with the missing value.

### Stage 5 - Decision Tree Regression

`src/decision_tree_model.py`

Decision Tree Regression is used as the **alternative model**.

Unlike Linear Regression, the Decision Tree can represent non-linear relationships between the surrounding measurements and the target temperature.

### Stage 6 - Decision Tree Optimisation

`src/decision_tree_optimisation.py`

The Decision Tree `max_depth` parameter is selected using chronological training and validation data.

The data are divided approximately into:

```text
60% training
20% validation
20% final test
```

A 30-minute purge region is applied around dataset boundaries to reduce overlap between neighbouring reconstruction windows.

The following values of `max_depth` were tested:

```text
2, 3, 4, 5, 6, 8, 10, 15, None
```

The final test dataset is not used for hyperparameter selection.

### Stage 7 - Final Model Comparison

`src/final_model_comparison.py`

The final comparison automatically reads the Decision Tree optimisation results and selects the depth with the lowest validation RMSE.

The final development/test split contains:

```text
Development examples: 6,939
Final test examples: 1,732
```

Development period:

```text
5 March 2024 to 27 May 2024
```

Final test period:

```text
27 May 2024 to 14 June 2024
```

The original measured temperatures in the final test set are treated as the **ground-truth reference**.

---

## 5. Validation and Optimisation

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Coefficient of determination (R²)

Chronological splitting is used rather than random splitting because the measurements form a time series.

A purge zone is used around split boundaries so that neighbouring reconstruction windows from different datasets are separated.

### Decision Tree Depth Optimisation

The validation results were:

| max_depth | Training RMSE (°C) | Validation RMSE (°C) |
| ---: | ---: | ---: |
| 2 | 0.3664 | 0.3575 |
| 3 | 0.2165 | 0.2204 |
| 4 | 0.1475 | 0.1536 |
| 5 | 0.1091 | 0.1131 |
| 6 | 0.0906 | 0.0950 |
| **8** | **0.0778** | **0.0872** |
| 10 | 0.0743 | 0.0900 |
| 15 | 0.0734 | 0.0894 |
| None | 0.0734 | 0.0894 |

The lowest validation RMSE occurred at:

```text
max_depth = 8
```

The difference between depth 8 and the deeper trees is relatively small, so the selected depth should not be interpreted as a universally optimal value.

### Software Verification

The project includes a pytest test suite using both project data and small synthetic examples with known expected results.

The final test suite contains:

```text
25 tests
```

Latest result:

```text
25 passed
```

The tests check:

- processed data-file existence;
- valid temperature data;
- chronological timestamps;
- daylight-saving offset handling;
- ML dataset columns and missing values;
- reproducibility of the committed ML dataset;
- correct reconstruction-window values;
- rejection of windows crossing a longer sampling gap;
- handling of unsorted input;
- missing-column handling;
- invalid JSON handling;
- missing measurement handling;
- chronological split purge behaviour;
- MAE, MSE, RMSE and R² calculations against known answers;
- Linear Regression behaviour on a known synthetic relationship;
- automatic Decision Tree depth selection;
- confirmation that the committed optimisation selects depth 8;
- Melbourne timezone handling;
- plain neighbour-average interpolation against hand-calculated values;
- counting of repeated measurements that have different temperatures.

The HTML test report is available at:

```text
reports/test_report.html
```

---

## 6. Results

### Final Test Performance

The final untouched test dataset contains **1,732 examples**.

| Model | MAE (°C) | RMSE (°C) | R² |
| --- | ---: | ---: | ---: |
| Linear Regression | 0.0521 | 0.0865 | 0.9931 |
| Decision Tree, depth 8 | 0.0478 | 0.0868 | 0.9931 |
| Neighbour average (plain linear interpolation) | 0.0506 | 0.0865 | 0.9931 |
| Decision Tree, depth 5 (before tuning) | 0.0728 | 0.1175 | 0.9873 |

The last two rows are reference models. The neighbour average predicts the missing value as the mean of `T(t-10)` and `T(t+10)` and learns nothing. The depth-5 tree is the tree used before the depth search.

Both models achieved very similar reconstruction accuracy.

The Decision Tree produced a slightly lower MAE.

Linear Regression produced a slightly lower RMSE and slightly higher R².

Therefore, the Decision Tree did not provide a clear improvement across all evaluation metrics despite being the more complex model.

The neighbour average has the same RMSE as Linear Regression (0.0865 °C), so Linear Regression did not improve on simple interpolation. Tuning `max_depth` was useful: the depth-5 tree has an RMSE of 0.1175 °C, about 26 % higher than the tuned depth-8 tree.

### Linear Regression Coefficients

The final Linear Regression coefficients were:

| Feature | Coefficient |
| --- | ---: |
| `temp_minus_20` | 0.0051 |
| `temp_minus_10` | 0.4971 |
| `temp_plus_10` | 0.4926 |
| `temp_plus_20` | 0.0082 |

Intercept:

```text
-0.0688
```

The two nearest measurements, `T(t-10)` and `T(t+10)`, each receive approximately half of the total weighting.

This indicates that the trained Linear Regression model learned a relationship close to interpolation between the nearest neighbouring measurements.

The measurements at `T(t-20)` and `T(t+20)` contribute relatively little.

### Decision Tree Feature Importance

The final Decision Tree feature importances were:

| Feature | Importance |
| --- | ---: |
| `temp_minus_20` | 0.0015 |
| `temp_minus_10` | 0.8243 |
| `temp_plus_10` | 0.1737 |
| `temp_plus_20` | 0.0005 |

The tree relies mainly on the closest neighbouring measurements.

These importance values should be interpreted cautiously because the input temperature features are strongly correlated and a Decision Tree may preferentially select one correlated feature over another.

---

## 7. Performance Analysis

`src/profile_runtime.py`

Two forms of computational performance analysis were completed.

### Data-Cleaning Profile

The raw-data cleaning stage was profiled using Python `cProfile`.

The latest run recorded:

```text
Total profiled cleaning runtime: 5.93 seconds
```

Important cumulative timings included approximately:

```text
extract_temperature_records(): 3.37 s
json.loads():                  2.04 s
CSV chunk reading:             1.71 s
```

These values are cumulative profiler times and therefore should not be added together directly.

The results show that parsing the large number of JSON records is a major part of the computational cost of the cleaning stage.

The profiling report is stored in:

```text
results/profile/clean_temperature_profile.txt
```

### Model Runtime Scaling

Training and prediction times were measured repeatedly and the median runtime was recorded.

| Training examples | Linear Regression fit (ms) | Decision Tree fit (ms) |
| ---: | ---: | ---: |
| 1,734 | 0.591 | 1.147 |
| 3,469 | 0.621 | 1.511 |
| 5,204 | 0.658 | 1.963 |
| 6,939 | 0.690 | 2.336 |

Linear Regression training time increased only slightly as the training dataset increased.

Decision Tree training time increased more noticeably with training-set size.

Prediction time remained below approximately 0.4 ms for both models in these tests.

The complete timing table is stored in:

```text
results/csv/model_timing_scaling.csv
```

---

## 8. Limitations

The project has several limitations:

- Only one sensor is analysed in detail.
- Only temperature is reconstructed.
- Only isolated single missing measurements are considered.
- Measurements before and after the missing point must already be available.
- The method is intended for offline reconstruction rather than real-time forecasting.
- Longer missing sequences are not reconstructed.
- The selected sensor covers approximately 3.5 months of data.
- The final test set covers one continuous period of approximately 18 days.
- Seasonal generalisation has not been established.
- Results from one sensor cannot automatically be generalised to other sensors.
- Decision Trees cannot reliably extrapolate outside patterns represented in their training data.
- The four temperature input variables are strongly correlated.
- Actual sampling intervals vary slightly around the nominal 10-minute interval.
- Decision Tree depth optimisation uses one chronological validation period, so a different validation period could produce a slightly different selected depth.

---

## 9. Reproduction and Repository Structure

### Repository Structure

```text
MMA3001-Project/
│
├── data/
│   ├── raw/
│   │   └── raw course dataset - not committed to GitHub
│   │
│   └── processed/
│       ├── clean_temperature.csv
│       ├── sensor_0869_temperature.csv
│       └── ml_temperature_dataset.csv
│
├── docs/
│   └── HTML source-code documentation generated with pdoc
│
├── reports/
│   └── test_report.html
│
├── results/
│   ├── csv/
│   │   ├── decision_tree_depth_results.csv
│   │   ├── final_model_comparison.csv
│   │   ├── final_test_predictions.csv
│   │   └── model_timing_scaling.csv
│   │
│   ├── figures/
│   │   ├── temperature_time_series.png
│   │   ├── decision_tree_depth_optimisation.png
│   │   ├── final_model_comparison.png
│   │   └── final_test_absolute_error.png
│   │
│   └── profile/
│       └── clean_temperature_profile.txt
│
├── src/
│   ├── __init__.py
│   ├── clean_temperature.py
│   ├── analyse_temperature.py
│   ├── create_ml_dataset.py
│   ├── linear_regression_model.py
│   ├── decision_tree_model.py
│   ├── decision_tree_optimisation.py
│   ├── final_model_comparison.py
│   └── profile_runtime.py
│
├── tests/
│   └── test_project.py
│
├── .gitignore
├── .gitattributes
├── pytest.ini
├── LICENSE
├── README.md
└── requirements.txt
```

### Installation

Python 3.10 or newer is recommended.

Install the required packages:

```bash
pip install -r requirements.txt
```

The raw course dataset is not included in the repository.

To reproduce the entire workflow from the raw dataset, create:

```text
data/raw/
```

and place the following file inside:

```text
5EnvSensor_MayToDec2024_180kRows.csv
```

### Run the Workflow

From the project root:

```bash
python -m src.clean_temperature
python -m src.analyse_temperature
python -m src.create_ml_dataset
python -m src.linear_regression_model
python -m src.decision_tree_model
python -m src.decision_tree_optimisation
python -m src.final_model_comparison
python -m src.profile_runtime
```

The cleaning stage requires the original raw dataset.

The runtime profiler automatically skips the cleaning profile if the raw CSV is unavailable.

### Run Tests

```bash
python -m pytest -v
```

Generate the HTML test report:

```bash
python -m pytest -v --html=reports/test_report.html --self-contained-html
```

### Generate HTML Documentation

```bash
pdoc src -o docs
```

A warning about suppressed execution of `lscpu` may appear during pdoc generation on some systems. This warning does not prevent the HTML documentation from being generated.

---

## 10. AI Use

Two AI tools were used: ChatGPT and Claude.

- **ChatGPT** was used mainly at the start of the project to analyse the data and
  to write the first version of the code.
- **Claude** was used afterwards to correct and restructure the code, write the
  tests, explain each script and draft the report from the author's results and
  Chinese draft.

The AI contribution was very high for the code, high for the wording of the
report, and lower for the choice of problem, the questions asked, the literature
and the interpretation of results. AI was used to build a complete and tested
workflow in the available time, while the author's effort was focused on
understanding the method and checking the results.

AI output was not accepted automatically. It was checked by running the code and
the tests (including a fresh clone of this repository run with the raw data),
by comparing the numbers in the report with the output files, and by reading the
three cited papers. The author found, read and selected the papers.

AI was wrong in several places, for example it stated a sensor resolution of
0.1 °C without checking the data (the data show an observed value step of about
0.25 °C, and the cause is not confirmed), misunderstood the meaning of
"baseline", and made a wrong hand-calculated value in a test. These were found
and corrected.

The author remains responsible for all code, results, claims and references.
See Section 7 of the project report for the full AI-use reflection.