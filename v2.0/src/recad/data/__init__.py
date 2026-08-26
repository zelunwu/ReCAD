"""Data pipeline: grid, source metadata, ingestion, features, split, tensors, tracks."""

from recad.data.features import FEATURE_NAMES, PreparedData
from recad.data.grid import DomainGrid, build_target_grid
from recad.data.ingest import (
    apply_qc,
    calc_clim_anom,
    flatten_time,
    regrid_to_target,
    remove_outliers_3sigma,
)
from recad.data.specs import describe_sources, required_variables
from recad.data.split import SplitMasks, make_split_masks
from recad.data.tensorize import CoastalPatchDataset, TensorShapes, collate_items
from recad.data.tracks import (
    TrackData,
    load_tracks,
    prepare_point_table,
    qc_tracks,
    sample_predictors,
)

__all__ = [
    "FEATURE_NAMES",
    "CoastalPatchDataset",
    "DomainGrid",
    "PreparedData",
    "SplitMasks",
    "TensorShapes",
    "TrackData",
    "apply_qc",
    "build_target_grid",
    "calc_clim_anom",
    "collate_items",
    "describe_sources",
    "flatten_time",
    "load_tracks",
    "make_split_masks",
    "prepare_point_table",
    "qc_tracks",
    "regrid_to_target",
    "remove_outliers_3sigma",
    "required_variables",
    "sample_predictors",
]
