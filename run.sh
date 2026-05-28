#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Building and starting services..."
docker compose build #--no-cache
docker compose up -d

echo "[INFO] Waiting for all services to become healthy..."

# ждём, пока все контейнеры перейдут в статус healthy
MAX_WAIT=180  # секунд
SECONDS=0
while [ $SECONDS -lt $MAX_WAIT ]; do
    unhealthy=$(docker ps --filter "health=unhealthy" --format "{{.Names}}")
    starting=$(docker ps --filter "health=starting" --format "{{.Names}}")
    if [ -z "$unhealthy" ] && [ -z "$starting" ]; then
        echo "[OK] All services are healthy!"
        break
    fi
    echo "[WAIT] Services not ready yet..."
    sleep 5
done

if [ $SECONDS -ge $MAX_WAIT ]; then
    echo "[ERROR] Timeout waiting for healthy containers!"
    docker ps
    exit 1
fi

# Активируем и запускаем DAG
echo "[INFO] Triggering Airflow DAG..."
docker exec hw3-airflow-scheduler airflow dags list
docker compose exec -T airflow-scheduler airflow dags unpause hw3_model_tracking || true
docker exec hw3-airflow-scheduler airflow dags trigger hw3_model_tracking || true

# Ожидаем загрузку данных в Redis
echo "[INFO] Waiting for Redis to receive data..."
for i in {1..20}; do
    count=$(docker exec hw3-airflow-scheduler python -c "import redis; r=redis.Redis(host='redis', port=6379); print(len(r.keys('*'))) " || echo 0)
    if [ "$count" -gt 0 ]; then
        echo "[OK] Redis has $count keys — data ready."
        break
    fi
    echo "[WAIT] Redis still empty... ($i/20)"
    sleep 5
done

echo "[INFO] Waiting for run_id from training (mlflow)..."
MAX_WAIT=2000
SECONDS=0
while [ $SECONDS -lt $MAX_WAIT ]; do
  if docker exec hw3-mlflow test -f /mlflow/run_id.txt; then
    run_id=$(docker exec hw3-mlflow cat /mlflow/run_id.txt | tr -d '\r\n')
    if [ -n "$run_id" ]; then
      echo "[OK] Got run_id: $run_id"
      break
    fi
  fi
  echo "[WAIT] run_id not ready..."
  sleep 5
done


if [ -z "${run_id:-}" ]; then
  echo "[ERROR] run_id not found!"
  exit 1
fi

# Новый блок ожидания модели (универсальный поиск внутри контейнера)
echo "[INFO] Waiting for model artifacts for run_id=$run_id ..."

# попробуем несколько вариантов путей, которые встречаются в твоем tree
CAND1="/mlflow/artifacts/$run_id/artifacts/model"
CAND2="/mlflow/artifacts/0/$run_id/artifacts/model"

MODEL_DIR=""
for i in {1..15}; do
  echo "[DEBUG] check attempt $i: testing $CAND1 and $CAND2"
  if docker exec hw3-mlflow test -d "$CAND1" >/dev/null 2>&1; then
    MODEL_DIR="$CAND1"
    echo "[OK] Found model at $MODEL_DIR"
    break
  fi
  if docker exec hw3-mlflow test -d "$CAND2" >/dev/null 2>&1; then
    MODEL_DIR="$CAND2"
    echo "[OK] Found model at $MODEL_DIR"
    break
  fi
  echo "[WAIT] Model not yet saved... ($i/15)"
  sleep 20
done

# если не нашли в привычных местах — попробуем найти вручную в /mlflow/artifacts
if [ -z "$MODEL_DIR" ]; then
  echo "[INFO] Trying to discover model path with find inside container..."
  MODEL_DIR=$(docker exec hw3-mlflow bash -lc "find /mlflow/artifacts -type d -name model -print 2>/dev/null | grep '/$run_id/' | head -n1 || true")
  if [ -n "$MODEL_DIR" ]; then
    echo "[OK] Discovered model dir: $MODEL_DIR"
  fi
fi

if [ -z "$MODEL_DIR" ]; then
  echo "[ERROR] Model directory not found for run_id=$run_id. Check that train_model logged model into MLflow artifacts."
  echo "Checked: $CAND1 and $CAND2 and attempted find inside container."
  exit 1
fi

echo "[INFO] Starting mlflow model serve inside mlflow container on port 6000 (serve path: $MODEL_DIR)..."

# final check
docker compose exec -T mlflow ss -ltnp | grep -E ':6000\\b' || echo "[OK] port 6000 free inside container"

# запуск serve с найденным MODEL_DIR (MODEL_DIR уже расширен на хосте)
docker exec -d hw3-mlflow bash -lc "\
  set -e; \
  export GUNICORN_CMD_ARGS='--bind=0.0.0.0:6000 --timeout=60 -w 1'; \
  echo GUNICORN_CMD_ARGS=\$GUNICORN_CMD_ARGS; \
  mlflow models serve --host 0.0.0.0 -m \"$MODEL_DIR\" -p 6000 --no-conda > /mlflow/serve.log 2>&1 & echo \$! > /mlflow/serve.pid || true; \
"

echo "[OK] mlflow model serve command started (container hw3-mlflow:6000), logs -> /mlflow/serve.log"


echo "[DONE] All systems up and DAG triggered successfully."
echo "Airflow UI:  http://localhost:8080 (airflow / airflow)"
echo "Minio:        http://localhost:9001 (minioadmin / minioadmin)"
echo "Redis:        localhost:6379"
