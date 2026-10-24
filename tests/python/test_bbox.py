import pandas as pd
import pytest
from wheat_mlops import add_yolo_columns, clip_and_filter, parse_bboxes


def _boxes(rows):
    df = pd.DataFrame(rows, columns=["image_id", "width", "height", "x", "y", "w", "h"])
    return df


def test_parse_bboxes():
    df = pd.DataFrame(
        {
            "image_id": ["a"],
            "width": [1024],
            "height": [1024],
            "bbox": ["[834.0, 222.0, 56.0, 36.0]"],
            "source": ["usask_1"],
        }
    )
    out = parse_bboxes(df)
    assert list(out[["x", "y", "w", "h"]].iloc[0]) == [834.0, 222.0, 56.0, 36.0]
    assert "bbox" not in out.columns


def test_yolo_conversion_values():
    kept, dropped = clip_and_filter(_boxes([("a", 1000, 1000, 100, 200, 50, 100)]), 1, 1e9)
    assert dropped == 0
    out = add_yolo_columns(kept).iloc[0]
    assert out["cx"] == pytest.approx(0.125)
    assert out["cy"] == pytest.approx(0.25)
    assert out["bw"] == pytest.approx(0.05)
    assert out["bh"] == pytest.approx(0.10)


def test_box_is_clipped_to_image():
    kept, dropped = clip_and_filter(_boxes([("a", 1000, 1000, 990, 100, 50, 100)]), 1, 1e9)
    assert dropped == 0
    assert kept.iloc[0]["x2"] == 1000  # 990 + 50 would overflow the image


def test_filters_drop_bad_boxes():
    rows = [
        ("a", 1000, 1000, 100, 100, 50, 50),  # fine (area 2500)
        ("a", 1000, 1000, 100, 100, 5, 5),  # too small (area 25)
        ("a", 1000, 1000, 0, 0, 900, 900),  # too large (area 810000)
        ("a", 1000, 1000, 2000, 2000, 50, 50),  # completely outside the image
        ("a", 1000, 1000, 100, 100, 0, 40),  # zero width
    ]
    kept, dropped = clip_and_filter(_boxes(rows), min_area=100, max_area=100_000)
    assert len(kept) == 1
    assert dropped == 4


def test_yolo_values_always_in_unit_range():
    rows = [("a", 1024, 1024, -20, 1000, 100, 100), ("a", 1024, 1024, 500, 500, 60, 60)]
    kept, _ = clip_and_filter(_boxes(rows), 1, 1e9)
    out = add_yolo_columns(kept)
    for col in ("cx", "cy", "bw", "bh"):
        assert out[col].between(0, 1).all()
