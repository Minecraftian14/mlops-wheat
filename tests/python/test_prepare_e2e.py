"""End-to-end test of prepare_data on a tiny synthetic dataset (no Kaggle data needed)."""

import json

import pandas as pd
import pytest
import yaml
from PIL import Image
from wheat_mlops import prepare

SIZE = 64


@pytest.fixture()
def raw_dir(tmp_path):
    raw = tmp_path / "raw"
    (raw / "train").mkdir(parents=True)
    rows = []
    for i in range(20):
        Image.new("RGB", (SIZE, SIZE), (i * 10, 50, 50)).save(raw / "train" / f"img{i:02d}.jpg")
        if i < 17:  # the last 3 images are background images (no boxes in the csv)
            source = "a" if i % 2 == 0 else "b"
            rows.append((f"img{i:02d}", SIZE, SIZE, "[10.0, 10.0, 20.0, 20.0]", source))
            # second box overflows the image edge and must be clipped
            rows.append((f"img{i:02d}", SIZE, SIZE, "[50.0, 50.0, 30.0, 30.0]", source))
            rows.append((f"img{i:02d}", SIZE, SIZE, "[1.0, 1.0, 2.0, 2.0]", source))  # tiny
    pd.DataFrame(rows, columns=["image_id", "width", "height", "bbox", "source"]).to_csv(
        raw / "train.csv", index=False
    )
    return raw


def test_prepare_end_to_end(raw_dir, tmp_path):
    out = tmp_path / "processed"
    stats = prepare(raw_dir, out, val_fraction=0.25, seed=0, min_box_area=50, max_box_area=1e6)

    assert stats["n_images"] == 20
    assert stats["n_train"] + stats["n_val"] == 20
    assert stats["n_background_images"] == 3
    assert stats["n_boxes_raw"] == 51
    assert stats["n_boxes_dropped"] == 17  # the tiny box of every annotated image
    assert stats["n_boxes_kept"] == 34

    for split in ("train", "val"):
        imgs = sorted(p.stem for p in (out / "images" / split).glob("*.jpg"))
        lbls = sorted(p.stem for p in (out / "labels" / split).glob("*.txt"))
        assert imgs == lbls  # every image has a label file (possibly empty)

    all_lines = []
    for p in (out / "labels").rglob("*.txt"):
        all_lines += [ln for ln in p.read_text().splitlines() if ln]
    assert len(all_lines) == 34
    for ln in all_lines:
        cls, *vals = ln.split()
        assert cls == "0" and len(vals) == 4
        assert all(0.0 <= float(v) <= 1.0 for v in vals)

    cfg = yaml.safe_load((out / "dataset.yaml").read_text())
    assert cfg["train"] == "images/train" and cfg["val"] == "images/val"
    assert cfg["nc"] == 1 and cfg["names"] == {0: "wheat"}
    assert "path" not in cfg

    assert json.loads((out / "prepare_stats.json").read_text()) == stats
    split_csv = pd.read_csv(out / "split.csv")
    assert len(split_csv) == 20


def test_prepare_is_deterministic_and_reruns_cleanly(raw_dir, tmp_path):
    out = tmp_path / "processed"
    prepare(raw_dir, out, val_fraction=0.25, seed=0, min_box_area=50)
    first = (out / "split.csv").read_text()
    prepare(raw_dir, out, val_fraction=0.25, seed=0, min_box_area=50)
    assert (out / "split.csv").read_text() == first


def test_refuses_to_wipe_raw_dir(raw_dir):
    with pytest.raises(ValueError):
        prepare(raw_dir, raw_dir)


def test_missing_data_gives_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="download_data"):
        prepare(tmp_path / "nope", tmp_path / "out")
