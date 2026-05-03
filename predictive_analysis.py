"""
Empirical comparison of machine learning models for predictive analysis.

The script evaluates four supervised learning models on a structured IoT-style
sensor dataset. It can use the included CSV file or create a reproducible sample
dataset when the file is missing.

Example:
    python predictive_analysis.py --data data/iot_sensor_energy.csv --output outputs
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import time
from pathlib import Path


DEFAULT_DATA_PATH = Path("data/iot_sensor_energy.csv")
DEFAULT_OUTPUT_DIR = Path("outputs")


def generate_sample_data(path: Path, rows: int = 480, seed: int = 42) -> None:
    """Create a reproducible IoT-style regression dataset for testing."""
    random.seed(seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    devices = {
        "pump_A": 0.35,
        "pump_B": 0.45,
        "compressor_A": 0.80,
        "hvac_A": 0.65,
        "fan_A": 0.20,
        "conveyor_A": 0.55,
    }

    header = [
        "record_id",
        "device_id",
        "hour",
        "temperature_c",
        "humidity_pct",
        "vibration_mm_s",
        "voltage_v",
        "current_a",
        "occupancy_count",
        "maintenance_flag",
        "energy_kwh",
    ]

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for record_id in range(1, rows + 1):
            device_id = random.choice(list(devices.keys()))
            hour = (record_id - 1) % 24
            day = (record_id - 1) // 24

            daily_cycle = math.sin((2 * math.pi * hour) / 24)
            workday_load = 1 if 7 <= hour <= 18 else 0
            weekend_adjustment = -0.15 if day % 7 in (5, 6) else 0.0

            temperature = 21.5 + 6.5 * daily_cycle + random.gauss(0, 1.2)
            humidity = 48 + 12 * math.cos((2 * math.pi * hour) / 24) + random.gauss(0, 3.0)
            vibration = 1.1 + devices[device_id] + random.uniform(0, 1.2)
            voltage = 229 + random.gauss(0, 2.6)
            current = 3.4 + devices[device_id] * 2.2 + workday_load * 1.4 + random.gauss(0, 0.45)
            occupancy = max(0, int(random.gauss(18 if workday_load else 4, 5)))
            maintenance = 1 if random.random() < 0.08 else 0

            energy = (
                1.85
                + devices[device_id]
                + 0.055 * temperature
                + 0.010 * humidity
                + 0.42 * current
                + 0.18 * vibration
                + 0.018 * occupancy
                + 0.38 * maintenance
                + weekend_adjustment
                + random.gauss(0, 0.22)
            )

            # A few blank values are included so preprocessing can demonstrate imputation.
            temp_value = "" if record_id % 97 == 0 else round(temperature, 2)
            humidity_value = "" if record_id % 113 == 0 else round(humidity, 2)
            vibration_value = "" if record_id % 131 == 0 else round(vibration, 2)

            writer.writerow(
                [
                    record_id,
                    device_id,
                    hour,
                    temp_value,
                    humidity_value,
                    vibration_value,
                    round(voltage, 2),
                    round(current, 2),
                    occupancy,
                    maintenance,
                    round(energy, 3),
                ]
            )


def load_ml_dependencies():
    """Import data science dependencies only when model evaluation is requested."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        from sklearn.model_selection import cross_val_score, train_test_split
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.svm import SVR
        from sklearn.tree import DecisionTreeRegressor
    except ImportError as exc:
        missing_package = exc.name or "a required package"
        raise SystemExit(
            f"Missing dependency: {missing_package}\n"
            "Install the required libraries, then run the script again:\n"
            "    pip install -r requirements.txt"
        ) from exc

    return {
        "plt": plt,
        "np": np,
        "pd": pd,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestRegressor": RandomForestRegressor,
        "SimpleImputer": SimpleImputer,
        "LinearRegression": LinearRegression,
        "mean_absolute_error": mean_absolute_error,
        "mean_squared_error": mean_squared_error,
        "r2_score": r2_score,
        "cross_val_score": cross_val_score,
        "train_test_split": train_test_split,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "StandardScaler": StandardScaler,
        "SVR": SVR,
        "DecisionTreeRegressor": DecisionTreeRegressor,
    }


def build_preprocessor(dataframe, target_column: str, deps):
    """Create preprocessing steps for numeric and categorical features."""
    pd = deps["pd"]
    ColumnTransformer = deps["ColumnTransformer"]
    Pipeline = deps["Pipeline"]
    SimpleImputer = deps["SimpleImputer"]
    OneHotEncoder = deps["OneHotEncoder"]
    StandardScaler = deps["StandardScaler"]

    feature_frame = dataframe.drop(columns=[target_column])
    numeric_features = feature_frame.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = feature_frame.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_transformer, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_transformer, categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns were found in the dataset.")

    return ColumnTransformer(transformers=transformers)


def evaluate_models(data_path: Path, output_dir: Path, seed: int = 42) -> None:
    """Train, evaluate, and compare machine learning models."""
    deps = load_ml_dependencies()
    np = deps["np"]
    pd = deps["pd"]
    plt = deps["plt"]
    Pipeline = deps["Pipeline"]
    LinearRegression = deps["LinearRegression"]
    DecisionTreeRegressor = deps["DecisionTreeRegressor"]
    RandomForestRegressor = deps["RandomForestRegressor"]
    SVR = deps["SVR"]
    train_test_split = deps["train_test_split"]
    cross_val_score = deps["cross_val_score"]
    mean_squared_error = deps["mean_squared_error"]
    mean_absolute_error = deps["mean_absolute_error"]
    r2_score = deps["r2_score"]

    if not data_path.exists():
        print(f"Data file not found at {data_path}. Creating a reproducible sample dataset.")
        generate_sample_data(data_path, seed=seed)

    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.read_csv(data_path)

    target_column = "energy_kwh"
    if target_column not in dataframe.columns:
        raise ValueError(f"Expected target column '{target_column}' was not found.")

    x = dataframe.drop(columns=[target_column, "record_id"], errors="ignore")
    y = dataframe[target_column]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.20,
        random_state=seed,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(max_depth=7, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=160,
            max_depth=12,
            random_state=seed,
            n_jobs=-1,
        ),
        "Support Vector Regression": SVR(kernel="rbf", C=12.0, epsilon=0.08),
    }

    rows = []
    predictions_by_model = {}

    for model_name, model in models.items():
        preprocessor = build_preprocessor(
            pd.concat([x_train, y_train], axis=1),
            target_column=target_column,
            deps=deps,
        )
        pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])

        train_start = time.perf_counter()
        pipeline.fit(x_train, y_train)
        training_time = time.perf_counter() - train_start

        predict_start = time.perf_counter()
        predictions = pipeline.predict(x_test)
        prediction_time = time.perf_counter() - predict_start
        predictions_by_model[model_name] = predictions

        mse = mean_squared_error(y_test, predictions)
        rmse = float(np.sqrt(mse))
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        cv_scores = cross_val_score(
            pipeline,
            x_train,
            y_train,
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        cv_rmse = float(-cv_scores.mean())

        rows.append(
            {
                "model": model_name,
                "mse": round(float(mse), 5),
                "rmse": round(rmse, 5),
                "mae": round(float(mae), 5),
                "r2_score": round(float(r2), 5),
                "cv_rmse": round(cv_rmse, 5),
                "training_time_seconds": round(training_time, 5),
                "prediction_time_seconds": round(prediction_time, 5),
            }
        )

    results = pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)
    results_path = output_dir / "model_results.csv"
    results.to_csv(results_path, index=False)

    best_model_name = results.loc[0, "model"]
    best_predictions = predictions_by_model[best_model_name]

    plt.figure(figsize=(9, 5))
    plt.bar(results["model"], results["rmse"], color=["#2f6f8f", "#7a8f2f", "#9a4f48", "#6c5b7b"])
    plt.title("Model Comparison by RMSE")
    plt.ylabel("Root Mean Squared Error")
    plt.xlabel("Model")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    rmse_plot_path = output_dir / "model_rmse_comparison.png"
    plt.savefig(rmse_plot_path, dpi=160)
    plt.close()

    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, best_predictions, alpha=0.75, color="#2f6f8f", edgecolor="white", linewidth=0.5)
    min_value = min(y_test.min(), best_predictions.min())
    max_value = max(y_test.max(), best_predictions.max())
    plt.plot([min_value, max_value], [min_value, max_value], color="#9a4f48", linewidth=2)
    plt.title(f"Actual vs Predicted Energy ({best_model_name})")
    plt.xlabel("Actual energy_kwh")
    plt.ylabel("Predicted energy_kwh")
    plt.tight_layout()
    prediction_plot_path = output_dir / "actual_vs_predicted_best_model.png"
    plt.savefig(prediction_plot_path, dpi=160)
    plt.close()

    print("\nModel evaluation completed.")
    print(f"Dataset: {data_path}")
    print(f"Rows evaluated: {len(dataframe)}")
    print(f"Best model by RMSE: {best_model_name}")
    print("\nResults:")
    print(results.to_string(index=False))
    print(f"\nSaved metrics: {results_path}")
    print(f"Saved plots: {rmse_plot_path}, {prediction_plot_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare machine learning models for IoT-style predictive analysis."
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Path to the input CSV dataset.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where result files and plots will be saved.",
    )
    parser.add_argument(
        "--generate-data",
        action="store_true",
        help="Create the sample dataset and exit unless --run-after-generate is also used.",
    )
    parser.add_argument(
        "--run-after-generate",
        action="store_true",
        help="Run the model comparison immediately after generating data.",
    )
    parser.add_argument(
        "--rows",
        type=int,
        default=480,
        help="Number of records to create when --generate-data is used.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible data generation and train/test split.",
    )
    args, unknown_args = parser.parse_known_args()
    if unknown_args:
        print(f"Ignoring notebook/runtime arguments: {' '.join(unknown_args)}")
    return args


def main() -> None:
    args = parse_args()

    if args.generate_data:
        generate_sample_data(args.data, rows=args.rows, seed=args.seed)
        print(f"Generated sample dataset: {args.data}")
        if not args.run_after_generate:
            return

    evaluate_models(data_path=args.data, output_dir=args.output, seed=args.seed)


if __name__ == "__main__":
    main()
