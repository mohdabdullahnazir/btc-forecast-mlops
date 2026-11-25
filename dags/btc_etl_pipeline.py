"""Daily Bitcoin training DAG. Task processes use an isolated ETL environment."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import timedelta
from pathlib import Path
from urllib.request import Request, urlopen

import pendulum
from airflow.decorators import dag, task
from airflow.operators.python import get_current_context

from src.model import load_artifact, save_artifact

ROOT = Path(os.environ.get("MODEL_ROOT", "/app/models"))
ETL_PYTHON = os.environ.get("ETL_PYTHON", sys.executable)


def run_etl(module, *arguments):
    subprocess.run([ETL_PYTHON, "-m", module, *map(str, arguments)], check=True)


def notify(url, payload):
    token = os.environ["RELOAD_TOKEN"]
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "X-Reload-Token": token},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        result = json.load(response)
    expected_version = (payload.get("model") or payload)["model_version"]
    if result.get("status") != "ok" or result.get("model_version") != expected_version:
        raise RuntimeError("API did not confirm the model reload.")
    print(f"{url} now serves {result['model_version']}.")


@dag(
    dag_id="btc_etl_pipeline",
    schedule="0 1 * * *",
    start_date=pendulum.datetime(2025, 1, 1, tz="UTC"),
    catchup=False,
    max_active_runs=1,
    default_args={"retries": 2, "retry_delay": timedelta(minutes=1)},
    tags=["mlops", "bitcoin"],
)
def btc_etl_pipeline():
    @task
    def fetch_data():
        run_id = get_current_context()["run_id"]
        key = hashlib.sha256(run_id.encode()).hexdigest()[:20]
        raw = ROOT / "runs" / key / "raw.csv"
        run_etl("src.etl_fetch", "--output", raw)
        return str(raw)

    @task
    def preprocess_features(raw):
        prepared = Path(raw).with_name("features.csv")
        run_etl("src.preprocess", "--input", raw, "--output", prepared)
        return str(prepared)

    @task
    def train_evaluate_register(prepared):
        candidate = Path(prepared).with_name("candidate.json")
        run_etl(
            "src.train_forecast",
            "--input",
            prepared,
            "--output",
            candidate,
            "--registry",
            ROOT / "versions",
        )
        return str(candidate)

    @task
    def reload_api(candidate, raw):
        artifact = load_artifact(candidate)
        save_artifact(ROOT / "model.json", artifact)
        notify(
            os.environ["MODEL_RELOAD_URL"], {"model_version": artifact["model_version"]}
        )
        cluster_url = os.environ.get("K8S_MODEL_RELOAD_URL")
        if cluster_url:
            notify(
                cluster_url,
                {"model": artifact, "raw_csv": Path(raw).read_text(encoding="utf-8")},
            )

    raw = fetch_data()
    prepared = preprocess_features(raw)
    candidate = train_evaluate_register(prepared)
    reload_api(candidate, raw)


btc_etl_pipeline()
