"""Clean OHLCV data and build features using only past and current closes."""

import argparse

import numpy as np
import pandas as pd

from src.model import FEATURES, LOOKBACK, atomic_write, features

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def preprocess(raw_path, output):
    data = pd.read_csv(raw_path, usecols=["Date", *OHLCV])
    data["Date"] = pd.to_datetime(data["Date"], utc=True, errors="coerce")
    for column in OHLCV:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna().sort_values("Date").drop_duplicates("Date", keep="last")
    valid = (
        np.isfinite(data[OHLCV]).all(axis=1)
        & data[OHLCV[:-1]].gt(0).all(axis=1)
        & data["Volume"].ge(0)
    )
    data = data.loc[valid].reset_index(drop=True)
    if data.empty:
        raise ValueError("No valid OHLCV rows remain after cleaning.")
    breaks = data.index[data["Date"].diff().dt.days.gt(1)]
    if len(breaks):
        data = data.iloc[breaks[-1] :].reset_index(drop=True)
    if len(data) < 120:
        raise ValueError("At least 120 consecutive daily OHLCV rows are required.")
    closes = data["Close"].to_numpy()
    rows = [
        features(closes[index - LOOKBACK + 1 : index + 1])
        for index in range(LOOKBACK - 1, len(data))
    ]
    prepared = data.iloc[LOOKBACK - 1 :].reset_index(drop=True)
    prepared["Date"] = prepared["Date"].dt.strftime("%Y-%m-%d")
    prepared[list(FEATURES)] = pd.DataFrame(rows, columns=FEATURES)
    prepared["target_return"] = np.log(prepared["Close"].shift(-1) / prepared["Close"])
    atomic_write(output, prepared.to_csv(index=False, lineterminator="\n"))
    print(f"Prepared {len(prepared)} consecutive feature rows.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    preprocess(args.input, args.output)
