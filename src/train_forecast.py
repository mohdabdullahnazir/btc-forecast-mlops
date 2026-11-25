"""Fit a ridge autoregression and evaluate a recursive seven-day holdout."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from src.model import FEATURES, HORIZON, LOOKBACK, forecast_prices, save_artifact


def fit(rows):
    rows = rows.dropna(subset=["target_return"])
    values = rows[list(FEATURES)].to_numpy()
    scaler = StandardScaler().fit(values)
    model = Ridge(alpha=10.0).fit(
        scaler.transform(values), rows["target_return"].to_numpy()
    )
    coefficients = model.coef_ / scaler.scale_
    intercept = model.intercept_ - coefficients @ scaler.mean_
    return coefficients.tolist(), float(intercept)


def train(features_path, output, registry=None):
    data = pd.read_csv(features_path)
    if len(data) < 120 - LOOKBACK + 1:
        raise ValueError("Insufficient prepared training rows.")
    train_rows = data.iloc[:-HORIZON]
    test_rows = data.iloc[-HORIZON:]
    # The last pre-holdout feature row targets holdout day 1, so exclude it.
    coefficients, intercept = fit(train_rows.iloc[:-1])
    prior = train_rows["Close"].tail(LOOKBACK).tolist()
    observed = test_rows["Close"].to_numpy()
    predicted = np.array(forecast_prices(prior, coefficients, intercept))
    error = predicted - observed
    baseline_rmse = float(np.sqrt(np.mean((prior[-1] - observed) ** 2)))
    metrics = {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape_pct": float(100 * np.mean(np.abs(error / observed))),
        "baseline_rmse": baseline_rmse,
        "holdout_start": test_rows["Date"].iloc[0],
        "holdout_end": test_rows["Date"].iloc[-1],
    }
    rmse_limit = max(1.5 * baseline_rmse, 0.005 * prior[-1])
    print(json.dumps({**metrics, "rmse_limit": rmse_limit}, indent=2))
    if not np.isfinite(list(metrics.values())[:4]).all():
        raise RuntimeError("Model produced non-finite evaluation metrics.")
    if metrics["mape_pct"] > 25 or metrics["rmse"] > rmse_limit:
        raise RuntimeError("Model failed the validation gate. Active model preserved.")

    # Refit only after evaluating the chronologically isolated holdout.
    coefficients, intercept = fit(data)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    version = f"{stamp}-{uuid4().hex[:8]}"
    artifact = {
        "schema_version": 1,
        "model_version": version,
        "symbol": "BTC-USD",
        "last_date": data["Date"].iloc[-1],
        "historical_closes": data["Close"].tail(LOOKBACK).tolist(),
        "features": list(FEATURES),
        "coefficients": coefficients,
        "intercept": intercept,
        "metrics": metrics,
    }
    if registry is not None:
        save_artifact(Path(registry) / f"{version}.json", artifact)
    save_artifact(output, artifact)
    print(f"Registered {version} from data through {artifact['last_date']}.")
    return version


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--registry")
    args = parser.parse_args()
    train(args.input, args.output, args.registry)
