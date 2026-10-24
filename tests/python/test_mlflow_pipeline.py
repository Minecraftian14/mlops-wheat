"""Unit tests for MLflow logging and pipeline orchestration on pipeline-a."""

import mlflow
import pytest
from mlflow.tracking import MlflowClient
from wheat_mlops.build_model import log_to_mlflow as log_model_to_mlflow
from wheat_mlops.prepare_data import log_to_mlflow as log_data_to_mlflow


@pytest.fixture()
def mlflow_env(tmp_path):
    db_file = tmp_path / "test_mlflow.db"
    tracking_uri = f"sqlite:///{db_file}"
    mlflow.set_tracking_uri(tracking_uri)
    experiment_id = mlflow.create_experiment("test-experiment")
    mlflow.set_experiment(experiment_id=experiment_id)
    return MlflowClient(tracking_uri=tracking_uri), experiment_id, tmp_path


def test_prepare_data_mlflow_logging(mlflow_env):
    client, exp_id, tmp_path = mlflow_env
    out_dir = tmp_path / "processed"
    out_dir.mkdir(parents=True)
    (out_dir / "dataset.yaml").write_text("train: images/train\nnc: 1\n", encoding="utf-8")
    (out_dir / "prepare_stats.json").write_text("{}", encoding="utf-8")

    stats = {
        "n_images": 100,
        "n_train": 80,
        "n_val": 20,
        "n_background_images": 5,
        "n_boxes_raw": 500,
        "n_boxes_kept": 490,
        "n_boxes_dropped": 10,
        "boxes_per_image_mean": 4.9,
    }
    data_params = {
        "val_fraction": 0.2,
        "seed": 42,
        "min_box_area": 100,
    }

    with mlflow.start_run(run_name="test_prepare_run") as run:
        log_data_to_mlflow(stats, data_params, out_dir)
        run_id = run.info.run_id

    run_data = client.get_run(run_id).data
    assert run_data.params["data.seed"] == "42"
    assert run_data.params["data.val_fraction"] == "0.2"
    assert run_data.metrics["data.n_images"] == 100
    assert run_data.metrics["data.n_train"] == 80
    assert run_data.metrics["data.boxes_per_image_mean"] == pytest.approx(4.9)

    artifacts = [a.path for a in client.list_artifacts(run_id, path="data")]
    assert "data/dataset.yaml" in artifacts
    assert "data/prepare_stats.json" in artifacts


def test_build_model_mlflow_logging(mlflow_env):
    client, exp_id, tmp_path = mlflow_env
    ckpt_dir = tmp_path / "checkpoints" / "base"
    ckpt_dir.mkdir(parents=True)
    fake_ckpt = ckpt_dir / "yolov8n_base.pt"
    fake_ckpt.write_text("dummy-weights", encoding="utf-8")
    (ckpt_dir / "model_info.json").write_text('{"imgsz": 640}', encoding="utf-8")

    info = {
        "base_checkpoint": str(fake_ckpt),
        "n_parameters": 3000000,
        "checkpoint_size_mb": 6.5,
    }
    model_params = {
        "imgsz": 640,
        "pretrained": "dummy.pt",
    }

    with mlflow.start_run(run_name="test_model_run") as run:
        log_model_to_mlflow(info, model_params)
        run_id = run.info.run_id

    run_data = client.get_run(run_id).data
    assert run_data.params["model.imgsz"] == "640"
    assert run_data.metrics["model.n_parameters"] == 3000000
    assert run_data.metrics["model.checkpoint_size_mb"] == pytest.approx(6.5)

    artifacts = [a.path for a in client.list_artifacts(run_id, path="checkpoints")]
    assert "checkpoints/yolov8n_base.pt" in artifacts
    assert "checkpoints/model_info.json" in artifacts
