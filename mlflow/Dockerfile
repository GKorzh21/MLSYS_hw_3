# mlflow/Dockerfile
FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    procps \
    psmisc \
 && rm -rf /var/lib/apt/lists/*


WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY train_model.py /app/train_model.py
# create folder for artifacts /mlflow
RUN mkdir -p /mlflow/artifacts
VOLUME ["/mlflow"]
#CMD ["bash", "-c", "mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db --default-artifact-root /mlflow/artifacts --host 0.0.0.0 --port 5000"]

CMD bash -c '\
    mlflow server --backend-store-uri sqlite:////mlflow/mlflow.db \
        --default-artifact-root /mlflow/artifacts/0 --host 0.0.0.0 --port 5000 & \
    echo "[WAIT] Checking for run_id.txt..."; \
    for i in {1..30}; do \
        if [ -f /mlflow/run_id.txt ]; then \
            run_id=$(cat /mlflow/run_id.txt); \
            echo "[INFO] Found run_id=$run_id"; \
            model_path=$(find /mlflow/artifacts/0 -type d -path "*/$run_id/artifacts/model" | head -n 1); \
            echo "[INFO] Model path: $model_path"; \
            if [ -n "$model_path" ]; then \
                mlflow models serve -m "$model_path" -p 6000 --host 0.0.0.0; \
                exit 0; \
            fi; \
        fi; \
        echo "[WAIT] Model not yet saved... ($i/30)"; \
        sleep 5; \
    done; \
    echo "[ERROR] No model found after waiting."; \
    exit 1'