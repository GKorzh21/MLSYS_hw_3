FROM openjdk:11-jre-slim
ENV SPARK_VERSION=3.4.0
# скачиваем spark (просто пример — в реале указываем рабочую версию)
RUN apt-get update && apt-get install -y curl python3 python3-pip unzip
RUN pip3 install pyspark boto3 redis
# ставим spark бинарники
RUN mkdir -p /opt/spark
# (здесь можно скачать и распаковать архив spark, но для простоты можно использовать pyspark без standalone binaries)
WORKDIR /opt/app
COPY spark-jobs /opt/app/spark-jobs
COPY spark/entrypoint.sh /opt/app/entrypoint.sh
RUN chmod +x /opt/app/entrypoint.sh
ENTRYPOINT ["/opt/app/entrypoint.sh"]

