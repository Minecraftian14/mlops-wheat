# wheat-mlops

MLOps mini-project on the Kaggle *Global Wheat Detection* competition
(small Ultralytics YOLO model; the point is the pipeline, not the leaderboard).

## Part 1 status
Data download, YOLO-format conversion, model creation and base checkpoint. No training yet.

```
params.yaml                         all paths and parameters (single source of truth)
dvc.yaml                            data pipeline stages
src/python/wheat_mlops/             installable Python package
 - download_data.py                 Kaggle download -> data/raw
 - prepare_data.py                  data/raw -> data/processed (YOLO format, stratified split)
 - build_model.py                   pretrained YOLO -> checkpoints/base/yolov8n_base.pt
src/scripts/                        development test and lint scripts
src/old_scripts/                    setup scripts, not for running, for record.
tests/python/                       tests that need neither real data nor a checkpoint
```

## Commands
```
uv run dvc repro                    # run/update the DVC data pipeline
uv run python -m wheat_mlops.build_model
uv run pytest
uv run ruff format --check .
```
