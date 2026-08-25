"""Registry of supported raw data sources.

Each entry documents one physical variable needed by the reconstruction. The
v1.1 provenance column links to the exact v1.1 notebook/live-script that used
the source, so v2.0 stays auditable against the published product (Wu et al.,
2024, ESSD). The full recommended *global* stack (with versions, access, and
caveats) is documented in ``docs/data_sources.md``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataSourceSpec:
    """Static metadata of one raw data source."""

    variable: str  # logical variable name used in the pipeline
    kind: str  # obs_target | satellite | reanalysis | atmosphere | mask
    units: str
    native_resolution_deg: float | tuple[float, float]
    temporal_sampling: str
    years: str
    v11_provenance: str  # where in v1.1 this source was used
    notes: str = ""


_SOURCES: tuple[DataSourceSpec, ...] = (
    DataSourceSpec(
        variable="fco2",
        kind="obs_target",
        units="µatm",
        native_resolution_deg=0.25,
        temporal_sampling="monthly gridded, coastal-weighted",
        years="1993-2021",
        v11_provenance="Data_readalldata.ipynb cell 3 (SOCAT v2023 quarter-degree "
        "gridded coastal monthly)",
        notes="Target observations for training. Global extent in SOCAT.",
    ),
    DataSourceSpec(
        variable="sst",
        kind="satellite",
        units="degC",
        native_resolution_deg=0.25,
        temporal_sampling="daily, averaged to monthly",
        years="1981-present",
        v11_provenance="Data_readalldata.ipynb cell 6 (OISST v2.1 AVHRR)",
        notes="Regridded to the target mesh with conservative averaging.",
    ),
    DataSourceSpec(
        variable="sss",
        kind="reanalysis",
        units="PSU",
        native_resolution_deg=(0.5, 0.5),
        temporal_sampling="monthly",
        years="1980-2018 (SODA 3.15.2)",
        v11_provenance="Data_readalldata.ipynb cell 5 (SODA 3.15.2 reg)",
        notes="v2.0 recommends GLORYS12v1 (1/12 deg) for global 1/8 deg runs.",
    ),
    DataSourceSpec(
        variable="adt",
        kind="satellite",
        units="m",
        native_resolution_deg=0.25,
        temporal_sampling="daily/monthly maps",
        years="1993-present",
        v11_provenance="Data_readalldata.ipynb cell 4 (CMEMS SEALEVEL_GLO_PHY_L4_MY)",
        notes="Absolute dynamic topography (adt) and SLA both available.",
    ),
    DataSourceSpec(
        variable="sla",
        kind="satellite",
        units="m",
        native_resolution_deg=0.25,
        temporal_sampling="monthly maps",
        years="1993-present",
        v11_provenance="Data_readalldata.ipynb cell 9 (CMEMS msla)",
        notes="Sea level anomaly; optional predictor.",
    ),
    DataSourceSpec(
        variable="wspd",
        kind="satellite/analysis",
        units="m/s",
        native_resolution_deg=0.25,
        temporal_sampling="6-hourly -> monthly mean",
        years="1987-present",
        v11_provenance="Data_readalldata.ipynb cell 7 (CCMP v2.0/v3.1)",
        notes="Optional predictor (used in the v1.1 leave-one-year-out tests).",
    ),
    DataSourceSpec(
        variable="chla",
        kind="satellite",
        units="mg m-3",
        native_resolution_deg=0.04,
        temporal_sampling="monthly composites",
        years="2002-present (MODIS Aqua)",
        v11_provenance="Data_readalldata.ipynb cell 10 (MODIS mapped monthly)",
        notes="Optional predictor; sparse early in the record.",
    ),
    DataSourceSpec(
        variable="xco2air",
        kind="atmosphere",
        units="µmol mol-1",
        native_resolution_deg=(4.0, 5.0),
        temporal_sampling="monthly",
        years="1992-present",
        v11_provenance="Data_readalldata.ipynb cell 8 (NOAA GML Surface CO2, "
        "90S-90N latitudes)",
        notes="Converted to pCO2air at in-situ SST/SSS with PyCO2SYS.",
    ),
    DataSourceSpec(
        variable="mask",
        kind="mask",
        units="1=coastal ocean",
        native_resolution_deg=(0.125, 0.125),
        temporal_sampling="static",
        years="-",
        v11_provenance="v1.1 hand-crafted regional masks (RFR_models.mlx cell 12)",
        notes="v2.0 uses a reproducible coastline-distance mask (GSHHG), "
        "see docs/data_sources.md.",
    ),
)


def describe_sources() -> tuple[DataSourceSpec, ...]:
    """Return the registered source metadata (stable ordering)."""
    return _SOURCES


def required_variables() -> tuple[str, ...]:
    """Ordered list of predictor/target logical variables used by the model."""
    return ("fco2", "sst", "sss", "adt", "pco2air", "wspd")