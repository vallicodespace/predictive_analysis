# Predictive Analysis Using Machine Learning Models

This project compares multiple supervised machine learning models for a structured predictive analysis task. The experiment uses an IoT-style sensor dataset and predicts `energy_kwh` from device, environment, and operating measurements.

## Project Files

- `predictive_analysis.py` - main Python script for data generation, preprocessing, training, evaluation, and plotting.
- `colab_runner.ipynb` - optional Google Colab notebook that installs dependencies and runs the script.
- `data/iot_sensor_energy.csv` - testable sample dataset used by the experiment.
- `requirements.txt` - Python package dependencies.
- `outputs/` - generated model metrics and plots after running the script.

## Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression

## Evaluation Metrics

The script reports:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared score
- 5-fold cross-validation RMSE
- Training time
- Prediction time

## How to Run in Google Colab

Google Colab is the intended execution environment for this project.

### Option 1: Run with the Included Colab Notebook

1. Upload the full project folder to Google Drive.
2. Open `colab_runner.ipynb` in Google Colab.
3. Run each notebook cell from top to bottom.
4. Check `outputs/model_results.csv` and the two generated plot images after execution.

### Option 2: Run Commands in a New Colab Notebook

1. Upload the full project folder to Google Drive.
2. Open a new Google Colab notebook.
3. Mount Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Change into the project directory. Update the path if your folder name is different:

```python
%cd /content/drive/MyDrive/New project
```

5. Install the required libraries:

```python
!pip install -r requirements.txt
```

6. Run the experiment:

```python
!python predictive_analysis.py --data data/iot_sensor_energy.csv --output outputs
```

7. Review the generated files:

```python
import pandas as pd
pd.read_csv("outputs/model_results.csv")
```

The plots will be saved as:

- `outputs/model_rmse_comparison.png`
- `outputs/actual_vs_predicted_best_model.png`

## Regenerating the Dataset

The included dataset is reproducible. To recreate it:

```bash
python predictive_analysis.py --generate-data --data data/iot_sensor_energy.csv --rows 480
```

To regenerate the data and immediately run the experiment:

```bash
python predictive_analysis.py --generate-data --run-after-generate --data data/iot_sensor_energy.csv --output outputs
```

## Dataset Description

The dataset contains structured IoT-style sensor readings:

- `device_id` - device category identifier.
- `hour` - hour of the day.
- `temperature_c` - measured temperature.
- `humidity_pct` - measured humidity.
- `vibration_mm_s` - vibration level.
- `voltage_v` - voltage measurement.
- `current_a` - current measurement.
- `occupancy_count` - estimated occupancy near the device.
- `maintenance_flag` - whether a maintenance condition was observed.
- `energy_kwh` - target value predicted by the models.

The script handles missing numeric values using mean imputation, scales numeric features, encodes categorical features, splits the dataset into 80% training and 20% testing, and applies 5-fold cross-validation for reliability.
