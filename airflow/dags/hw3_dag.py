from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
import os

DATA_DIR = "/data"
ML_DIR_SMALL = os.path.join(DATA_DIR, "ml-latest-small")
ML_DIR_LATEST = os.path.join(DATA_DIR, "ml-latest")
OUTPUT_DIR = os.path.join(DATA_DIR, "output")

default_args = {
    "owner": "hw3",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}

with DAG(
    dag_id="hw3_model_tracking",
    default_args=default_args,
    start_date=datetime(2025,1,1),
    schedule_interval=None,
    catchup=False,
    max_active_runs=1,
) as dag:

    ensure_dirs = PythonOperator(
        task_id="ensure_dirs",
        python_callable=lambda: (os.makedirs(ML_DIR_SMALL, exist_ok=True), os.makedirs(OUTPUT_DIR, exist_ok=True))
    )

    get_dataset = BashOperator(
        task_id="get_dataset",
        bash_command="python /opt/airflow/spark-jobs/get_dataset.py /data --version latest"
    )

    put_dataset = BashOperator(
        task_id="put_dataset",
        bash_command="python /opt/airflow/spark-jobs/put_dataset.py /data/ml-latest"
    )

    split_dataset = BashOperator(
        task_id="split_dataset",
        bash_command="python /opt/airflow/spark-jobs/split_dataset.py /data/ml-latest /data/output"
    )

    features_engineering_train = BashOperator(
        task_id="features_engineering_train",
        bash_command="python /opt/airflow/spark-jobs/features_engineering_train.py /data/output/train.csv /data/output/train_features.parquet /data/ml-latest"
    )

    features_engineering_test = BashOperator(
        task_id="features_engineering_test",
        bash_command="python /opt/airflow/spark-jobs/features_engineering_test.py /data/output/test.csv /data/output/test_features.parquet /data/ml-latest"
    )

    load_features = BashOperator(
        task_id="load_features",
        bash_command="python /opt/airflow/spark-jobs/load_to_redis_ratings.py /data/output/test_features.parquet redis 6379"
    )

    #train_and_save_model = BashOperator(
    #    task_id="train_and_save_model",
    #    bash_command="sudo docker exec -i hw3-mlflow python /app/train_model.py /data/output/train_features.parquet",
    #)

    train_and_save_model = BashOperator(
        task_id="train_and_save_model",
        bash_command=(
            "sudo docker exec -i hw3-mlflow "
            "python /app/train_model.py /data/output/train_features.parquet "
            "--sample-frac 0.1 --max-iter 100 --no-redis-validation"
        ),
    )


    
    ensure_dirs >> get_dataset >> put_dataset >> split_dataset
    split_dataset >> [features_engineering_train, features_engineering_test]
    features_engineering_test >> load_features >> train_and_save_model
    features_engineering_train >> train_and_save_model
