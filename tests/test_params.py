from src.config import load_params


def test_params_have_required_keys():
    p = load_params()
    for key in (
        "competition",
        "raw_dir",
        "processed_dir",
        "val_fraction",
        "seed",
        "min_box_area",
        "max_box_area",
    ):
        assert key in p["data"]
    for key in ("pretrained", "base_checkpoint", "imgsz"):
        assert key in p["model"]


def test_params_are_sane():
    p = load_params()
    assert 0 < p["data"]["val_fraction"] < 1
    assert p["data"]["min_box_area"] < p["data"]["max_box_area"]
    assert p["model"]["imgsz"] % 32 == 0  # YOLO stride requirement
