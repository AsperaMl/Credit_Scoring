from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

with DAG(
    'prediction_dag',
    default_args=default_args,
    description='Получение предсказаний каждые 3 минуты',
    schedule_interval='*/3 * * * *',
    catchup=False
) as dag:
    run_prediction = BashOperator(
        task_id='run_make_pred',
        bash_command='cd /opt/airflow/dags && python make_pred.py'
    )
