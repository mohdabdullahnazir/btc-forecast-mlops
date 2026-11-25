"""Shared feature and forecast code. Inference needs only the standard library."""

import json
import math
import os
import re
import statistics
import tempfile
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

HORIZON = 7
LOOKBACK = 30
FEATURES = (
    *(f"return_{lag}" for lag in range(1, 8)),
    "ma_gap_7",
    "ma_gap_30",
    "volatility_7",
)
VERSION_PATTERN = re.compile(r"[0-9]{8}T[0-9]{12}Z-[0-9a-f]{8}")


def features(closes):
    prices = closes[-LOOKBACK:]
    returns = [math.log(b / a) for a, b in zip(prices, prices[1:])]
    return [
        *reversed(returns[-7:]),
        math.log(prices[-1] / statistics.fmean(prices[-7:])),
        math.log(prices[-1] / statistics.fmean(prices)),
        statistics.pstdev(returns[-7:]),
    ]


def forecast_prices(closes, coefficients, intercept, days=HORIZON):
    history = list(closes[-LOOKBACK:])
    result = []
    for _ in range(days):
        inputs = features(history)
        next_return = intercept + sum(
            value * coefficient for value, coefficient in zip(inputs, coefficients)
        )
        # Bound extrapolation when inputs lie far outside the training distribution.
        close = history[-1] * math.exp(max(-0.5, min(0.5, next_return)))
        result.append(close)
        history.append(close)
    return result


def finite_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def validate_closes(closes):
    if not isinstance(closes, list) or not LOOKBACK <= len(closes) <= 3650:
        raise ValueError("historical_closes must contain 30 to 3650 daily closes.")
    if any(
        not finite_number(value) or not 0.000001 <= value <= 1_000_000_000
        for value in closes
    ):
        raise ValueError("Each close must be a finite positive BTC-USD price.")
    return closes


def parse_date(value):
    if not isinstance(value, str):
        raise ValueError("last_date must be in YYYY-MM-DD format.")
    try:
        parsed = date.fromisoformat(value)
        if parsed.isoformat() == value and parsed <= date.max - timedelta(days=HORIZON):
            return parsed
    except ValueError:
        pass
    raise ValueError("last_date must be in YYYY-MM-DD format.")


def validate_artifact(artifact):
    if not isinstance(artifact, dict) or artifact.get("schema_version") != 1:
        raise ValueError("Unrecognized model artifact schema.")
    if artifact.get("symbol") != "BTC-USD" or artifact.get("features") != list(
        FEATURES
    ):
        raise ValueError("Model symbol or feature schema is incompatible.")
    version = artifact.get("model_version")
    if not isinstance(version, str) or not VERSION_PATTERN.fullmatch(version):
        raise ValueError("Invalid model version.")
    parse_date(artifact.get("last_date"))
    validate_closes(artifact.get("historical_closes"))
    coefficients = artifact.get("coefficients")
    if (
        not isinstance(coefficients, list)
        or len(coefficients) != len(FEATURES)
        or not all(finite_number(item) and abs(item) <= 1e6 for item in coefficients)
        or not finite_number(artifact.get("intercept"))
        or abs(artifact["intercept"]) > 1e6
    ):
        raise ValueError("Invalid model coefficients.")
    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict) or any(
        not finite_number(metrics.get(name)) or metrics[name] < 0
        for name in ("mae", "rmse", "mape_pct", "baseline_rmse")
    ):
        raise ValueError("Invalid model validation metrics.")
    return artifact


def atomic_write(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        # Airflow and the Flask process share group access in Compose.
        os.chmod(temporary, 0o640)
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def save_artifact(path, artifact):
    validate_artifact(artifact)
    atomic_write(path, json.dumps(artifact, indent=2, allow_nan=False) + "\n")


def load_artifact(path):
    return validate_artifact(json.loads(Path(path).read_text(encoding="utf-8")))


@contextmanager
def publish_lock(path):
    """Serialize concurrent writers in separate API workers on the same volume."""
    with Path(path).open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield
