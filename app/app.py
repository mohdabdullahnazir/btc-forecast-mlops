"""Gunicorn entry point for the BTC-USD inference service."""

import csv
import os
from datetime import timedelta
from hmac import compare_digest
from pathlib import Path
from time import monotonic

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException

from src.model import (
    HORIZON,
    atomic_write,
    forecast_prices,
    load_artifact,
    parse_date,
    publish_lock,
    save_artifact,
    validate_artifact,
    validate_closes,
)


def create_app(config=None):
    app = Flask(__name__)
    app.config.update(
        MODEL_ROOT=os.environ.get("MODEL_ROOT", "/app/models"),
        MODEL_STAGE=os.environ.get("MODEL_STAGE", "local"),
        RELOAD_TOKEN=os.environ.get("RELOAD_TOKEN", ""),
        ALLOW_MODEL_UPLOAD=os.environ.get("ALLOW_MODEL_UPLOAD") == "true",
        MAX_CONTENT_LENGTH=2 * 1024 * 1024,
    )
    if config:
        app.config.update(config)
    app.logger.setLevel(os.environ.get("LOG_LEVEL", "INFO").upper())
    root = Path(app.config["MODEL_ROOT"])
    model_path = root / "model.json"
    started = monotonic()

    def body():
        data = request.get_json(silent=True)
        if not isinstance(data, dict):
            raise ValueError("Send a JSON object with Content-Type application/json.")
        return data

    @app.errorhandler(ValueError)
    def invalid_request(error):
        return jsonify(error=str(error)), 400

    @app.errorhandler(HTTPException)
    def http_error(error):
        return jsonify(error=error.description), error.code

    @app.errorhandler(Exception)
    def internal_error(error):
        app.logger.exception("Unexpected service error")
        return jsonify(error="Internal server error."), 500

    def active_model():
        try:
            return load_artifact(model_path)
        except (OSError, ValueError) as error:
            if isinstance(error, FileNotFoundError):
                raise
            raise RuntimeError("Cannot read active model artifact.") from error

    @app.errorhandler(FileNotFoundError)
    def missing_model(error):
        return jsonify(error="No model has been registered."), 503

    @app.get("/health")
    def health():
        model = active_model()
        return jsonify(
            status="ok",
            uptime_seconds=round(monotonic() - started, 3),
            model_version=model["model_version"],
            model_stage=app.config["MODEL_STAGE"],
        )

    @app.post("/predict")
    def predict():
        data = body()
        if data.keys() - {"forecast_days", "historical_closes", "last_date"}:
            raise ValueError("Unexpected prediction field.")
        days = data.get("forecast_days", HORIZON)
        if type(days) is not int or not 1 <= days <= HORIZON:
            raise ValueError("forecast_days must be an integer between 1 and 7.")
        model = active_model()
        if "historical_closes" in data:
            closes = validate_closes(data["historical_closes"])
            last_date = parse_date(data.get("last_date"))
        elif "last_date" in data:
            raise ValueError("last_date is only used with historical_closes.")
        else:
            closes = model["historical_closes"]
            last_date = parse_date(model["last_date"])
        prices = forecast_prices(
            closes, model["coefficients"], model["intercept"], days
        )
        return jsonify(
            symbol=model["symbol"],
            model_version=model["model_version"],
            last_date=last_date.isoformat(),
            predictions=[
                {
                    "date": (last_date + timedelta(days=index)).isoformat(),
                    "predicted_close": round(price, 2),
                }
                for index, price in enumerate(prices, start=1)
            ],
        )

    @app.get("/metrics")
    def metrics():
        model = active_model()
        return jsonify(model_version=model["model_version"], **model["metrics"])

    @app.post("/internal/reload")
    def reload_model():
        configured = app.config["RELOAD_TOKEN"].encode()
        supplied = request.headers.get("X-Reload-Token", "").encode()
        if not configured or not compare_digest(configured, supplied):
            return jsonify(error="Unauthorized."), 401
        data = body()
        if "model" in data:
            if not app.config["ALLOW_MODEL_UPLOAD"]:
                return jsonify(error="Artifact upload is disabled."), 403
            incoming = validate_artifact(data["model"])
            raw = data.get("raw_csv")
            if not isinstance(raw, str) or raw.partition("\n")[0].strip() != (
                "Date,Open,High,Low,Close,Volume"
            ):
                raise ValueError("raw_csv must contain the BTC-USD OHLCV snapshot.")
            version = incoming["model_version"]
            with publish_lock(root / ".publish.lock"):
                try:
                    current = active_model()
                except FileNotFoundError:
                    current = None
                if current:
                    if version == current["model_version"]:
                        return jsonify(status="ok", model_version=version)
                    old_key = (current["last_date"], current["model_version"])
                    if (incoming["last_date"], version) <= old_key:
                        return jsonify(error="An equal or newer model is active."), 409
                # Read the header as CSV and require at least one market observation.
                if not next(csv.DictReader(raw.splitlines()), None):
                    raise ValueError("raw_csv requires at least one observation.")
                atomic_write(root / "history" / f"{version}.csv", raw)
                save_artifact(root / "versions" / f"{version}.json", incoming)
                save_artifact(model_path, incoming)
        else:
            version = data.get("model_version")
            if not isinstance(version, str):
                raise ValueError("model_version is required.")
            if active_model()["model_version"] != version:
                return jsonify(error="Model version is unavailable."), 409
        app.logger.info("Serving model %s", version)
        return jsonify(status="ok", model_version=version)

    return app


app = create_app()
