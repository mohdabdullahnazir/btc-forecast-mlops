FROM python:3.12-slim AS builder
WORKDIR /build
RUN python -m venv /opt/venv
COPY requirements-api.txt .
RUN /opt/venv/bin/pip install --no-cache-dir -r requirements-api.txt

FROM python:3.12-slim AS runtime
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PATH="/opt/venv/bin:${PATH}"
WORKDIR /app
RUN groupadd --gid 10001 forecast && useradd --uid 10001 --gid 10001 --no-create-home forecast \
    && mkdir -p /app/models && chown 10001:10001 /app/models
COPY --from=builder /opt/venv /opt/venv
COPY app/__init__.py app/app.py ./app/
COPY src/__init__.py src/model.py ./src/
USER 10001:10001
EXPOSE 5000
CMD ["sh", "-c", "exec gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 --threads 2 --access-logfile - --log-level ${LOG_LEVEL:-info} app.app:app"]
