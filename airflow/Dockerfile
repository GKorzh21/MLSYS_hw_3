FROM apache/airflow:2.6.2

USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    docker.io \
    sudo \
    openjdk-11-jre-headless \
    gcc build-essential \
    && rm -rf /var/lib/apt/lists/*

# allow 'airflow' to run docker without password
RUN echo "airflow ALL=(ALL) NOPASSWD: /usr/bin/docker" >> /etc/sudoers

RUN mkdir -p /data && chown -R airflow:root /data

USER airflow
ENV PATH="/home/airflow/.local/bin:${PATH}"
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV _JAVA_OPTIONS="-Dfile.encoding=UTF8"

COPY --chown=airflow:airflow requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --user -r /tmp/requirements.txt