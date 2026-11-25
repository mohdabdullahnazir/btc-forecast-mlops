"""Endpoint, validation and shared-artifact tests discovered by pytest."""

import copy
import tempfile
import unittest
from pathlib import Path

from app.app import create_app
from src.model import FEATURES, save_artifact


class TestAPI(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.model = {
            "schema_version": 1,
            "model_version": "20260101T013000000001Z-1234abcd",
            "symbol": "BTC-USD",
            "last_date": "2026-01-01",
            "historical_closes": [90_000.0] * 30,
            "features": list(FEATURES),
            "coefficients": [0.0] * len(FEATURES),
            "intercept": 0.001,
            "metrics": {
                "mae": 5.0,
                "rmse": 6.0,
                "mape_pct": 0.01,
                "baseline_rmse": 7.0,
            },
        }
        save_artifact(self.root / "model.json", self.model)
        self.config = {
            "TESTING": True,
            "MODEL_ROOT": str(self.root),
            "RELOAD_TOKEN": "test-token",
            "ALLOW_MODEL_UPLOAD": True,
        }
        self.client = create_app(self.config).test_client()
        self.token = {"X-Reload-Token": "test-token"}

    def test_health_metrics_and_default_forecast(self):
        health = self.client.get("/health")
        self.assertEqual(health.status_code, 200)
        self.assertEqual(health.json["status"], "ok")
        self.assertEqual(health.json["model_version"], self.model["model_version"])
        self.assertGreaterEqual(health.json["uptime_seconds"], 0)
        metrics = self.client.get("/metrics")
        self.assertEqual(metrics.status_code, 200)
        self.assertEqual(metrics.json["rmse"], 6.0)
        forecast = self.client.post("/predict", json={})
        self.assertEqual(forecast.status_code, 200)
        points = forecast.json["predictions"]
        self.assertEqual(len(points), 7)
        self.assertEqual(points[0]["date"], "2026-01-02")
        self.assertEqual(points[6]["date"], "2026-01-08")
        self.assertGreater(points[-1]["predicted_close"], points[0]["predicted_close"])

    def test_custom_history(self):
        response = self.client.post(
            "/predict",
            json={
                "forecast_days": 2,
                "historical_closes": [70_000] * 30,
                "last_date": "2026-02-28",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            [p["date"] for p in response.json["predictions"]],
            ["2026-03-01", "2026-03-02"],
        )
        self.assertEqual(response.json["predictions"][0]["predicted_close"], 70070.04)

    def test_invalid_request_returns_400(self):
        for value in (0, 8, True, None, "7"):
            with self.subTest(value=value):
                response = self.client.post("/predict", json={"forecast_days": value})
                self.assertEqual(response.status_code, 400)
        for data in (
            ["not an object"],
            {"historical_closes": [1] * 29, "last_date": "2026-01-01"},
            {"historical_closes": [1] * 29 + [float("nan")], "last_date": "2026-01-01"},
            {"historical_closes": [1] * 30, "last_date": "2026-02-30"},
            {"last_date": "2026-01-01"},
            {"typo": 1},
        ):
            with self.subTest(data=data):
                self.assertEqual(
                    self.client.post("/predict", json=data).status_code, 400
                )
        self.assertEqual(self.client.post("/predict", data="{").status_code, 400)

    def test_missing_and_unreadable_model_return_json_errors(self):
        model_path = self.root / "model.json"
        model_path.unlink()
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 503)
        self.assertIn("error", response.json)
        model_path.write_text("broken", encoding="utf-8")
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json["error"], "Internal server error.")

    def test_reload_authentication_and_local_read(self):
        payload = {"model_version": self.model["model_version"]}
        self.assertEqual(
            self.client.post("/internal/reload", json=payload).status_code, 401
        )
        response = self.client.post(
            "/internal/reload", json=payload, headers=self.token
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json["model_version"], payload["model_version"])

    def test_cluster_promotion_visible_to_second_instance_and_rejects_rollback(self):
        second = create_app(self.config).test_client()
        candidate = copy.deepcopy(self.model)
        candidate["model_version"] = "20260102T013000000001Z-1234abcd"
        candidate["last_date"] = "2026-01-02"
        raw = "Date,Open,High,Low,Close,Volume\n2026-01-02,90000,90000,90000,90000,10\n"
        payload = {"model": candidate, "raw_csv": raw}
        for _ in range(2):
            response = self.client.post(
                "/internal/reload", json=payload, headers=self.token
            )
            self.assertEqual(response.status_code, 200)
        version = second.get("/health").json["model_version"]
        self.assertEqual(version, candidate["model_version"])
        self.assertEqual((self.root / "history" / f"{version}.csv").read_text(), raw)
        self.assertTrue((self.root / "versions" / f"{version}.json").is_file())
        payload["model"] = self.model
        response = self.client.post(
            "/internal/reload", json=payload, headers=self.token
        )
        self.assertEqual(response.status_code, 409)
