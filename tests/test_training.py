"""Offline end-to-end training and artifact promotion checks."""

import math
import tempfile
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from app.app import create_app
from src.model import HORIZON, load_artifact
from src.preprocess import preprocess
from src.train_forecast import train


class TestTraining(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.raw = self.root / "raw.csv"
        self.prepared = self.root / "features.csv"
        self.model = self.root / "model.json"

    def make_history(self, last_seven_multiplier=1):
        origin = date(2024, 1, 1)
        rows = []
        for index in range(170):
            close = 50_000 + 15 * index + 12 * math.sin(index / 7)
            if index >= 170 - HORIZON:
                close *= last_seven_multiplier
            rows.append(
                {
                    "Date": (origin + timedelta(days=index)).isoformat(),
                    "Open": close - 2,
                    "High": close + 6,
                    "Low": close - 8,
                    "Close": close,
                    "Volume": 1_000_000,
                }
            )
        pd.DataFrame(rows).to_csv(self.raw, index=False)

    def test_training_registers_valid_model_and_forecasts(self):
        self.make_history()
        preprocess(self.raw, self.prepared)
        version = train(self.prepared, self.model, self.root / "versions")
        artifact = load_artifact(self.model)
        self.assertEqual(artifact["model_version"], version)
        self.assertTrue((self.root / "versions" / f"{version}.json").exists())
        self.assertEqual(artifact["metrics"]["holdout_start"], "2024-06-12")
        self.assertEqual(artifact["metrics"]["holdout_end"], "2024-06-18")
        self.assertLess(
            artifact["metrics"]["rmse"], artifact["metrics"]["baseline_rmse"]
        )
        client = create_app({"MODEL_ROOT": str(self.root)}).test_client()
        forecast = client.post("/predict", json={}).json
        self.assertEqual(forecast["predictions"][0]["date"], "2024-06-19")
        self.assertEqual(len(forecast["predictions"]), HORIZON)

    def test_rejected_candidate_leaves_active_artifact_untouched(self):
        self.make_history()
        preprocess(self.raw, self.prepared)
        train(self.prepared, self.model)
        before = self.model.read_bytes()
        self.make_history(last_seven_multiplier=3)
        preprocess(self.raw, self.prepared)
        with self.assertRaisesRegex(RuntimeError, "failed the validation gate"):
            train(self.prepared, self.model)
        self.assertEqual(self.model.read_bytes(), before)
