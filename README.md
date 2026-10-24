# wheat-mlops

MLOps mini-project on the Kaggle *Global Wheat Detection* competition
(small Ultralytics YOLO model; the point is the pipeline, not the leaderboard).

## Part 1 status
Data download, YOLO-format conversion, model creation and base checkpoint. No training yet.

```
params.yaml            all paths and parameters (single source of truth)
src/download_data.py   Kaggle download -> data/raw
src/prepare_data.py    data/raw -> data/processed (YOLO format, stratified split)
src/build_model.py     pretrained YOLO -> checkpoints/base/yolov8n_base.pt
tests/                 unit tests that need neither the real data nor any checkpoint
```

## Commands
```
python -m src.download_data
python -m src.prepare_data
python -m src.build_model
pytest
ruff check .
```
