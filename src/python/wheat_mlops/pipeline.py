"""MLflow Pipeline runner for Pipeline A.

Coordinates the end-to-end MLOps pipeline on pipeline-a:
1. prepare_data (data prep, validation split, YOLO label formatting)
2. build_model (YOLO initialization, smoke test inference, base checkpoint saving)

Logs all parameters, metrics, and artifacts into an MLflow run structure.

Usage:
    python -m wheat_mlops.pipeline
    python -m wheat_mlops.pipeline --prepare-only
    python -m wheat_mlops.pipeline --build-only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow

from wheat_mlops.build_model import build
from wheat_mlops.build_model import log_to_mlflow as log_model_to_mlflow
from wheat_mlops.config import load_params
from wheat_mlops.prepare_data import log_to_mlflow as log_data_to_mlflow
from wheat_mlops.prepare_data import prepare


def run_pipeline(
    params_path: str | Path = "params.yaml",
    run_name: str | None = None,
    prepare_only: bool = False,
    build_only: bool = False,
) -> str:
    params = load_params(params_path)
    mlflow_cfg = params.get("mlflow", {})
    tracking_uri = mlflow_cfg.get("tracking_uri", "sqlite:///mlflow.db")
    experiment_name = mlflow_cfg.get("experiment_name", "wheat-mlops-pipeline-a")

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    actual_run_name = run_name or "pipeline-a-run"

    with mlflow.start_run(run_name=actual_run_name) as parent_run:
        run_id = parent_run.info.run_id
        print(f"=== MLflow Pipeline Run Started [ID: {run_id}] ===")
        mlflow.set_tag("pipeline", "pipeline-a")
        mlflow.set_tag("branch", "pipeline-a")

        # Stage 1: Data Preparation
        if not build_only:
            print("\n--> Running Stage: prepare_data")
            p_data = params["data"]
            out_dir = Path(p_data["processed_dir"])
            with mlflow.start_run(run_name="prepare_data", nested=True):
                stats = prepare(
                    raw_dir=p_data["raw_dir"],
                    out_dir=out_dir,
                    val_fraction=p_data["val_fraction"],
                    seed=p_data["seed"],
                    count_bucket=p_data["count_bucket"],
                    max_count_bucket=p_data["max_count_bucket"],
                    min_box_area=p_data["min_box_area"],
                    max_box_area=p_data["max_box_area"],
                    link_images=p_data["link_images"],
                )
                log_data_to_mlflow(stats, p_data, out_dir)
                print(f"    Completed prepare_data: {stats['n_images']} images processed.")

        # Stage 2: Model Build & Base Checkpoint
        if not prepare_only:
            print("\n--> Running Stage: build_model")
            p_model = params["model"]
            with mlflow.start_run(run_name="build_model", nested=True):
                info = build(
                    pretrained=p_model["pretrained"],
                    base_checkpoint=p_model["base_checkpoint"],
                    imgsz=p_model["imgsz"],
                )
                log_model_to_mlflow(info, p_model)
                print(f"    Completed build_model: {info['n_parameters']} parameters.")

        print("\n=== MLflow Pipeline Run Finished Successfully ===")
        print(f"Experiment: {experiment_name}")
        print(f"Run ID:     {run_id}")
        print(f"Tracked in: {tracking_uri}")
        print("To view results in MLflow UI:")
        print(f"  uv run mlflow ui --backend-store-uri {tracking_uri}")

        return run_id


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--params", default="params.yaml", help="Path to params.yaml")
    parser.add_argument("--run-name", default=None, help="Custom name for the MLflow run")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only execute the prepare_data stage",
    )
    parser.add_argument(
        "--build-only",
        action="store_true",
        help="Only execute the build_model stage",
    )
    args = parser.parse_args()

    run_pipeline(
        params_path=args.params,
        run_name=args.run_name,
        prepare_only=args.prepare_only,
        build_only=args.build_only,
    )


if __name__ == "__main__":
    main()
