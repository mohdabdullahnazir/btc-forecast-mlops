"""Compile the three containerized training steps into Kubeflow Pipeline IR."""

import argparse
import os

from kfp import compiler, dsl
from kfp.dsl import Dataset, Input, Model, Output

TRAINER_IMAGE = os.environ.get("BTC_TRAINER_IMAGE", "btc-trainer:dev")


@dsl.container_component
def fetch_btc(raw: Output[Dataset]):
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "src.etl_fetch"],
        args=["--output", raw.path],
    )


@dsl.container_component
def engineer_features(raw: Input[Dataset], prepared: Output[Dataset]):
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "src.preprocess"],
        args=["--input", raw.path, "--output", prepared.path],
    )


@dsl.container_component
def train_7day(prepared: Input[Dataset], model: Output[Model]):
    return dsl.ContainerSpec(
        image=TRAINER_IMAGE,
        command=["python", "-m", "src.train_forecast"],
        args=["--input", prepared.path, "--output", model.path],
    )


@dsl.pipeline(
    name="btc-7day-forecast",
    description="Fetch, preprocess and evaluate a seven-day BTC-USD forecast.",
)
def btc_pipeline():
    raw = fetch_btc().set_caching_options(False)
    prepared = engineer_features(raw=raw.outputs["raw"]).set_caching_options(False)
    train_7day(prepared=prepared.outputs["prepared"]).set_caching_options(False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="kubeflow/btc_pipeline.yaml")
    args = parser.parse_args()
    compiler.Compiler().compile(btc_pipeline, args.output)
