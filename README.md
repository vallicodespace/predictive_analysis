# predictive_analysis

This project looks at how different machine learning models perform on a predictive analysis problem. It uses an IoT-style sensor dataset and predicts `energy_kwh` based on device activity, environmental readings, and operating measurements.

## Project Files

- `predictive_analysis.py` - main Python script used to prepare the data, train the models, evaluate results, and create plots.
- `colab_runner.ipynb` - optional Google Colab notebook with the run steps already organized.
- `data/iot_sensor_energy.csv` - sample dataset used for testing the project.
- `requirements.txt` - list of Python libraries needed to run the project.
- `outputs/` - folder where the results and plots are saved after the script runs.

## Models Evaluated

- Linear Regression
- Decision Tree Regressor
- Random Forest Regressor
- Support Vector Regression

## Evaluation Metrics

The results are compared using:

- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)
- R-squared score
- 5-fold cross-validation RMSE
- Training time
- Prediction time

## How to Run in Google Colab

This project is meant to be run in Google Colab. The easiest way is to upload the project zip file directly into a Colab session.

1. Open a new notebook at https://colab.research.google.com.
2. Upload `predictive_analysis.zip` manually through the Colab upload prompt.
3. Unzip the project folder.
4. Install the required libraries.
5. Run the Python script and review the saved results.

Use these cells in Colab:

```python
from google.colab import files
uploaded = files.upload()
```

```python
!unzip predictive_analysis.zip
```

```python
%cd predictive_analysis
```

```python
!pip install -r requirements.txt
```

```python
!python predictive_analysis.py --data data/iot_sensor_energy.csv --output outputs
```

After the script finishes, use this cell to view the model results:

```python
import pandas as pd
pd.read_csv("outputs/model_results.csv")
```

Use this cell to display the two saved plots:

```python
from IPython.display import Image, display

display(Image(filename="outputs/model_rmse_comparison.png"))
display(Image(filename="outputs/actual_vs_predicted_best_model.png"))
```

The plots will be saved as:

- `outputs/model_rmse_comparison.png`
- `outputs/actual_vs_predicted_best_model.png`

## Regenerating the Dataset

The included dataset can be recreated from the script. To generate it again:

```bash
python predictive_analysis.py --generate-data --data data/iot_sensor_energy.csv --rows 480
```

To generate the data and immediately run the experiment:

```bash
python predictive_analysis.py --generate-data --run-after-generate --data data/iot_sensor_energy.csv --output outputs
```

## Dataset Description

The dataset contains structured IoT-style sensor readings. Each row represents one sensor record:

- `device_id` - device category identifier.
- `hour` - hour of the day.
- `temperature_c` - measured temperature.
- `humidity_pct` - measured humidity.
- `vibration_mm_s` - vibration level.
- `voltage_v` - voltage measurement.
- `current_a` - current measurement.
- `occupancy_count` - estimated occupancy near the device.
- `maintenance_flag` - whether a maintenance condition was observed.
- `energy_kwh` - target value that the models predict.

Before training, the script fills missing numeric values, scales numeric features, encodes categorical fields, and splits the data into 80% training and 20% testing. It also uses 5-fold cross-validation to make the comparison more reliable.
