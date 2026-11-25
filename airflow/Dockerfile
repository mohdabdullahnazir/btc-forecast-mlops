FROM apache/airflow:2.10.5-python3.12 AS builder
USER airflow
COPY requirements-train.txt /tmp/requirements-train.txt
RUN python -m venv /opt/airflow/etl-venv \
    && /opt/airflow/etl-venv/bin/pip install --no-cache-dir -r /tmp/requirements-train.txt

FROM apache/airflow:2.10.5-python3.12
ENV PYTHONPATH="/opt/airflow/project" ETL_PYTHON="/opt/airflow/etl-venv/bin/python" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
COPY --from=builder --chown=airflow:0 /opt/airflow/etl-venv /opt/airflow/etl-venv
COPY --chown=airflow:0 src/ /opt/airflow/project/src/
COPY --chown=airflow:0 dags/ /opt/airflow/dags/
USER airflow
