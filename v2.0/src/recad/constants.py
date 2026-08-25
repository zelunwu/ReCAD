"""Physical constants and default QC thresholds for ReCAD v2.0.

Values are carried over and *hardened* from the v1.1 pipeline
(``v1.1/u_inputs_Monte_Carlo.mlx`` and ``v1.1/RFR_models.mlx``) so that v2.0
stays quantitatively comparable with the published v1.1 product
(Wu et al., 2024, ESSD). Each constant is annotated with its provenance.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Target years of the reconstruction (v1.1: 1993-2021, monthly)
# ---------------------------------------------------------------------------
YEAR_MIN_DEFAULT = 1993
YEAR_MAX_DEFAULT = 2021

# ---------------------------------------------------------------------------
# Quality-control thresholds applied to each predictor / target.
# Source: v1.1/RFR_models.mlx lines "fco2(fco2<1)=nan; sla(sla<-10)=nan; ..."
# ---------------------------------------------------------------------------
QC_RANGES: dict[str, tuple[float | None, float | None]] = {
    "fco2": (1.0, 1000.0),  # µatm; values below 1 or above 1000 removed
    "pco2": (1.0, 1000.0),  # µatm
    "sst": (-4.0, 100.0),  # °C
    "sss": (0.0, 100.0),  # PSU
    "sla": (-10.0, None),  # m; only the lower bound is applied in v1.1
    "adt": (None, None),  # m; no explicit QC in v1.1
    "wspd": (0.0, None),  # m/s
    "chla": (0.0, None),  # mg m-3
    "pco2air": (200.0, None),  # µatm; values below 200 removed
}

# Outlier rule used for SOCAT fCO2 in v1.1 (Data_readalldata.ipynb):
# values lying more than 3 population std devs from the domain mean are NaN.
N_STD_OUTLIER = 3.0

# ---------------------------------------------------------------------------
# Monthly climatology / anomaly decomposition (detrend-then-average), mirroring
# v1.1's calc_clim_anom in Data_readalldata.ipynb.
# ---------------------------------------------------------------------------
MONTHS_PER_YEAR = 12

# ---------------------------------------------------------------------------
# Resolutions supported for the reconstruction grid.
# ---------------------------------------------------------------------------
RESOLUTIONS_SUPPORTED = (1 / 12, 1 / 8, 0.25)

# ---------------------------------------------------------------------------
# Input measurement uncertainties (1 sigma) used in the v2.0 Monte-Carlo
# input-error propagation. Source: v1.1/u_inputs_Monte_Carlo.mlx:
#   u_sst = 0.23; u_sss = 0.6; u_ssh = 0.018; u_u10 = 0.901; u_pco2air = 0.22;
# NOTE: v1.1 treated these as homogeneous; v2.0 additionally supports
# per-pixel SST/SSH error fields (cfg.uncertainty.per_pixel_sst_err,
# per_pixel_ssh_err) when available.
# ---------------------------------------------------------------------------
DEFAULT_INPUT_UNCERTAINTIES: dict[str, float] = {
    "u_sst": 0.23,  # °C
    "u_sss": 0.6,  # PSU
    "u_ssh": 0.018,  # m
    "u_u10": 0.901,  # m/s
    "u_pco2air": 0.22,  # %  (relative, as in v1.1: normrnd(0, 0.22))
}

# ---------------------------------------------------------------------------
# fCO2 -> pCO2 conversion (Wanninkhof 1992 fugacity correction, as used in
# v1.1 Data_readalldata.ipynb): pCO2 = fCO2 * (1.00436 - 4.669e-5 * SST).
# ---------------------------------------------------------------------------
FUGACITY_COEFF_A = 1.00436
FUGACITY_COEFF_B = 4.669e-5

# ---------------------------------------------------------------------------
# Default Monte-Carlo draw count (v1.1 used N_test = 100).
# ---------------------------------------------------------------------------
DEFAULT_MC_DRAWS = 100

# ---------------------------------------------------------------------------
# Ensemble size default.
# ---------------------------------------------------------------------------
DEFAULT_ENSEMBLE_MEMBERS = 5
