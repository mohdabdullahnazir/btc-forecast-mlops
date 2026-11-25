FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
COPY requirements-train.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements-train.txt

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:${PATH}" \
    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
WORKDIR /app
RUN groupadd --gid 10001 forecast \
    && useradd --uid 10001 --gid 10001 --no-create-home --home-dir /tmp forecast \
    && mkdir -p /minio /s3 /gcs \
    && chown 10001:10001 /minio /s3 /gcs
COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
USER 10001:10001
