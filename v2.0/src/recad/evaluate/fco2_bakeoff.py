"""Leakage-safe models and diagnostics for the global coastal fCO2 bake-off."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SeasonalTrendBackground:
    """Fold-fitted trend/season background with shrunk LME-month corrections."""

    center_year: float = 2009.0
    shrinkage_groups: float = 20.0

    def fit(self, frame: pd.DataFrame) -> SeasonalTrendBackground:
        year = frame.yr.to_numpy(float) - self.center_year
        month = frame.mon.to_numpy(int)
        phase = 2.0 * np.pi * (month - 1.0) / 12.0
        design = np.column_stack([np.ones(len(frame)), year, np.sin(phase), np.cos(phase)])
        truth = frame.truth.to_numpy(float)
        self.coefficients = np.linalg.lstsq(design, truth, rcond=None)[0]
        residual = truth - design @ self.coefficients
        work = frame[["lme_id", "mon", "group_key"]].copy()
        work["residual"] = residual
        self.global_residual = float(np.mean(residual))
        lme = work.groupby("lme_id", observed=True).agg(
            correction=("residual", "mean"), groups=("group_key", "nunique")
        )
        lme["correction"] *= lme.groups / (lme.groups + self.shrinkage_groups)
        self.lme_correction = lme.correction.to_dict()
        cell = work.groupby(["lme_id", "mon"], observed=True).agg(
            correction=("residual", "mean"), groups=("group_key", "nunique")
        )
        parent = np.asarray(
            [self.lme_correction.get(key[0], self.global_residual) for key in cell.index]
        )
        weight = cell.groups.to_numpy(float) / (cell.groups.to_numpy(float) + self.shrinkage_groups)
        cell["correction"] = weight * cell.correction.to_numpy(float) + (1 - weight) * parent
        self.cell_correction = cell.correction.to_dict()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        year = frame.yr.to_numpy(float) - self.center_year
        phase = 2.0 * np.pi * (frame.mon.to_numpy(int) - 1.0) / 12.0
        design = np.column_stack([np.ones(len(frame)), year, np.sin(phase), np.cos(phase)])
        base = design @ self.coefficients
        correction = np.asarray(
            [
                self.cell_correction.get(
                    (row.lme_id, row.mon),
                    self.lme_correction.get(row.lme_id, self.global_residual),
                )
                for row in frame.itertuples()
            ]
        )
        return base + correction


@dataclass
class LMEResidualAdapter:
    """Shrunk LME/month correction with an exact unseen-LME zero fallback."""

    shrinkage_groups: float = 30.0

    def fit(self, frame: pd.DataFrame, base_prediction: np.ndarray) -> LMEResidualAdapter:
        work = frame[["lme_id", "mon", "group_key"]].copy()
        work["error"] = frame.truth.to_numpy(float) - np.asarray(base_prediction, dtype=float)
        lme = work.groupby("lme_id", observed=True).agg(
            correction=("error", "mean"), groups=("group_key", "nunique")
        )
        lme["correction"] *= lme.groups / (lme.groups + self.shrinkage_groups)
        self.lme_correction = lme.correction.to_dict()
        cell = work.groupby(["lme_id", "mon"], observed=True).agg(
            correction=("error", "mean"), groups=("group_key", "nunique")
        )
        parent = np.asarray([self.lme_correction[key[0]] for key in cell.index])
        weight = cell.groups.to_numpy(float) / (cell.groups.to_numpy(float) + self.shrinkage_groups)
        cell["correction"] = weight * cell.correction.to_numpy(float) + (1 - weight) * parent
        self.cell_correction = cell.correction.to_dict()
        return self

    def predict_correction(self, frame: pd.DataFrame) -> np.ndarray:
        return np.asarray(
            [
                self.cell_correction.get(
                    (row.lme_id, row.mon), self.lme_correction.get(row.lme_id, 0.0)
                )
                for row in frame.itertuples()
            ]
        )


def fco2_band(values: pd.Series | np.ndarray) -> pd.Categorical:
    """Return the frozen fCO2 diagnostic bands."""

    return pd.cut(
        np.asarray(values, dtype=float),
        [-np.inf, 250, 350, 450, 550, np.inf],
        labels=["lt250", "250_350", "350_450", "450_550", "ge550"],
        right=False,
    )


def interval_scores(
    truth: np.ndarray,
    prediction: np.ndarray,
    width50: np.ndarray,
    width90: np.ndarray,
) -> dict[str, float]:
    """Return empirical coverage, interval widths, and a two-level WIS."""

    absolute = np.abs(np.asarray(truth) - np.asarray(prediction))
    width50 = np.asarray(width50)
    width90 = np.asarray(width90)
    score50 = 2 * width50 + 4 * np.maximum(absolute - width50, 0)
    score90 = 2 * width90 + 20 * np.maximum(absolute - width90, 0)
    return {
        "coverage50": float(np.mean(absolute <= width50)),
        "coverage90": float(np.mean(absolute <= width90)),
        "mean_width50": float(np.mean(2 * width50)),
        "mean_width90": float(np.mean(2 * width90)),
        "wis": float(np.mean((absolute + 0.25 * score50 + 0.05 * score90) / 1.3)),
    }
