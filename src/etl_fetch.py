"""Fetch completed UTC daily Bitcoin candles from Yahoo Finance."""

import argparse
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf

from src.model import atomic_write

OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def fetch(output, start="2015-01-01"):
    today_utc = datetime.now(timezone.utc).date().isoformat()
    data = yf.download(
        "BTC-USD",
        start=start,
        end=today_utc,
        interval="1d",
        auto_adjust=False,
        multi_level_index=False,
        progress=False,
        threads=False,
        timeout=30,
    )
    if data.empty or not set(OHLCV).issubset(data.columns):
        raise RuntimeError("Yahoo Finance returned no usable daily BTC-USD data.")
    data = data[OHLCV].copy()
    data.index = pd.to_datetime(data.index, utc=True).strftime("%Y-%m-%d")
    data.index.name = "Date"
    atomic_write(output, data.to_csv(lineterminator="\n"))
    print(f"Fetched {len(data)} daily rows through {data.index.max()} UTC.")
    return len(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--start", default="2015-01-01")
    args = parser.parse_args()
    fetch(args.output, args.start)
