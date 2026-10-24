import pandas as pd

from wheat_mlops import make_strata, split_ids


def _images(n_per_source=40, sources=("s1", "s2", "s3")):
    rows = []
    for s_idx, s in enumerate(sources):
        for i in range(n_per_source):
            rows.append((f"{s}_{i}", s, (i * (s_idx + 1)) % 50))
    return pd.DataFrame(rows, columns=["image_id", "source", "n_boxes"])


def test_split_is_disjoint_and_complete():
    images = _images()
    train, val = split_ids(images, 0.2, seed=42)
    assert train.isdisjoint(val)
    assert train | val == set(images["image_id"])
    assert abs(len(val) - 0.2 * len(images)) <= 1


def test_split_is_deterministic():
    images = _images()
    assert split_ids(images, 0.2, seed=1) == split_ids(images, 0.2, seed=1)


def test_every_source_in_both_splits():
    images = _images()
    _, val = split_ids(images, 0.2, seed=42)
    val_sources = set(images[images["image_id"].isin(val)]["source"])
    assert val_sources == {"s1", "s2", "s3"}


def test_rare_strata_do_not_crash():
    images = _images(n_per_source=30)
    extra = pd.DataFrame(
        [("odd_1", "weird", 7), ("odd_2", "weird2", 99), ("odd_3", "weird3", 0)],
        columns=images.columns,
    )
    images = pd.concat([images, extra], ignore_index=True)
    key = make_strata(images)
    assert key.map(key.value_counts()).min() >= 2
    train, val = split_ids(images, 0.2, seed=42)
    assert train | val == set(images["image_id"])


def test_tiny_dataset_falls_back_to_plain_split():
    images = pd.DataFrame(
        {"image_id": list("abcd"), "source": list("wxyz"), "n_boxes": [1, 2, 3, 4]}
    )
    train, val = split_ids(images, 0.25, seed=0)
    assert len(train) == 3 and len(val) == 1
