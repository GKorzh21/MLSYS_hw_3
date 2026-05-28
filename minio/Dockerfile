# minio/Dockerfile — пробуем добавить wget (подходит если базовый образ поддерживает apt)
FROM minio/minio:latest

# Попытка установить через apt; если образ основан на Debian/Ubuntu — сработает.
USER root
RUN apt-get update && apt-get install -y --no-install-recommends wget && \
    rm -rf /var/lib/apt/lists/*

USER 1000
