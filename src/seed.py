"""Initialize a single storage volume before starting the Kubernetes API."""

import os
from pathlib import Path

from src.etl_fetch import fetch
from src.preprocess import preprocess
from src.train_forecast import train


def main():
    root = Path(os.environ.get("MODEL_ROOT", "/app/models"))
    model_path = root / "model.json"
    if model_path.is_file():
        print("Active model already exists.")
        return
    raw = root / "raw.csv"
    prepared = root / "features.csv"
    fetch(raw)
    preprocess(raw, prepared)
    version = train(prepared, model_path, root / "versions")
    print(f"Seed complete. Model version {version}.")


if __name__ == "__main__":
    main()
