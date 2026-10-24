"""wheat_mlops package."""

from __future__ import annotations

__all__ = [
    "add_yolo_columns",
    "build",
    "clip_and_filter",
    "download",
    "load_params",
    "make_strata",
    "parse_bboxes",
    "prepare",
    "split_ids",
]


def __getattr__(name: str):
    if name in {
        "add_yolo_columns",
        "clip_and_filter",
        "make_strata",
        "parse_bboxes",
        "prepare",
        "split_ids",
    }:
        import wheat_mlops.prepare_data as m

        return getattr(m, name)
    if name == "build":
        import wheat_mlops.build_model as m

        return getattr(m, name)
    if name == "download":
        import wheat_mlops.download_data as m

        return getattr(m, name)
    if name == "load_params":
        import wheat_mlops.config as m

        return getattr(m, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
