"""Data pipeline: grid, source metadata, ingestion, features, split, tensors."""

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

__all__ = [
    "FEATURE_NAMES",
    "PreparedData",
    "DomainGrid",
    "build_target_grid",
    "apply_qc",
    "calc_clim_anom",
    "flatten_time",
    "regrid_to_target",
    "remove_outliers_3sigma",
    "describe_sources",
    "required_variables",
    "SplitMasks",
    "make_split_masks",
    "CoastalPatchDataset",
    "TensorShapes",
    "collate_items",
]