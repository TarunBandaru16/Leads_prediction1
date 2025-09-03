from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import pandas as pd
import joblib

# Load reference and current data
reference = pd.read_csv("data/reference.csv")
current = pd.read_csv("data/current.csv")

report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=reference, current_data=current)

# Extract drift score
drift_score = report.as_dict()['metrics'][0]['result']['dataset_drift_score']
print(drift_score)  # Airflow expects this on stdout
