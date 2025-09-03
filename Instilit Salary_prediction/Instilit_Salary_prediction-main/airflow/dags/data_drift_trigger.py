from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.utils.dates import days_ago
import subprocess
import json

# DAG setup
default_args = {
    'owner': 'airflow',
    'start_date': days_ago(1),
}

dag = DAG(
    dag_id='data_drift_retrain_trigger',
    default_args=default_args,
    schedule_interval='@daily',  # Adjust as needed
    catchup=False
)

DRIFT_THRESHOLD = 0.5

def check_data_drift():
    result = subprocess.run(['python', '/opt/airflow/scripts/check_drift.py'], capture_output=True, text=True)
    drift_score = float(result.stdout.strip())
    print(f"Drift score: {drift_score}")
    return 'retrain_model' if drift_score > DRIFT_THRESHOLD else 'no_action'

def retrain_model():
    subprocess.run(['python', '/opt/airflow/scripts/retrain_model.py'])

def no_action():
    print("Drift is below threshold. Skipping retraining.")

# Tasks
branching = BranchPythonOperator(
    task_id='check_drift_threshold',
    python_callable=check_data_drift,
    dag=dag,
)

retrain_task = PythonOperator(
    task_id='retrain_model',
    python_callable=retrain_model,
    dag=dag,
)

no_action_task = PythonOperator(
    task_id='no_action',
    python_callable=no_action,
    dag=dag,
)

# DAG flow
branching >> [retrain_task, no_action_task]
