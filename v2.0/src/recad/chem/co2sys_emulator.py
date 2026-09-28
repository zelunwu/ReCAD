"""Differentiable, frozen emulator for the PyCO2SYS TA+DIC -> fCO2 map.

The emulator is only used to propagate the soft chemistry loss during model
training.  Reported closure is always recomputed with PyCO2SYS itself.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn


INPUT_NAMES = ("temperature", "salinity", "TA", "DIC")


class CO2SYSEmulator(nn.Module):
    def __init__(self, mean: torch.Tensor, std: torch.Tensor, width: int = 192) -> None:
        super().__init__()
        self.register_buffer("input_mean", mean.float())
        self.register_buffer("input_std", std.float())
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self, temperature: torch.Tensor, salinity: torch.Tensor,
        ta: torch.Tensor, dic: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack((temperature, salinity, ta, dic), dim=-1)
        z = (x - self.input_mean) / self.input_std
        # log(fCO2) is substantially smoother over the broad training domain.
        return self.net(z)[..., 0].exp()

    @classmethod
    def load_frozen(cls, path: str | Path, device: str | torch.device = "cpu") -> "CO2SYSEmulator":
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(state["input_mean"], state["input_std"], int(state["width"]))
        model.load_state_dict(state["model"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.to(device)


class CO2SYSDICEmulator(nn.Module):
    """Differentiable inverse map: T, S, TA and fCO2 to DIC."""

    def __init__(
        self, mean: torch.Tensor, std: torch.Tensor,
        target_mean: torch.Tensor, target_std: torch.Tensor, width: int = 192,
    ) -> None:
        super().__init__()
        self.register_buffer("input_mean", mean.float())
        self.register_buffer("input_std", std.float())
        self.register_buffer("target_mean", target_mean.float())
        self.register_buffer("target_std", target_std.float())
        self.net = nn.Sequential(
            nn.Linear(4, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, width), nn.SiLU(),
            nn.Linear(width, 1),
        )

    def forward(
        self, temperature: torch.Tensor, salinity: torch.Tensor,
        ta: torch.Tensor, fco2: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.stack((temperature, salinity, ta, fco2.clamp_min(1).log()), dim=-1)
        z = (x - self.input_mean) / self.input_std
        return self.net(z)[..., 0] * self.target_std + self.target_mean

    @classmethod
    def load_frozen(cls, path: str | Path, device: str | torch.device = "cpu") -> "CO2SYSDICEmulator":
        state = torch.load(path, map_location=device, weights_only=False)
        model = cls(state["input_mean"], state["input_std"], state["target_mean"],
                    state["target_std"], int(state["width"]))
        model.load_state_dict(state["model"])
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model.to(device)


def pyco2sys_fco2(temperature: np.ndarray, salinity: np.ndarray,
                  ta: np.ndarray, dic: np.ndarray) -> np.ndarray:
    """Exact PyCO2SYS forward calculation with fixed documented options."""
    import PyCO2SYS as pyco2

    out = pyco2.sys(
        par1=np.asarray(ta, dtype=np.float64), par2=np.asarray(dic, dtype=np.float64),
        par1_type=1, par2_type=2, salinity=np.asarray(salinity, dtype=np.float64),
        temperature=np.asarray(temperature, dtype=np.float64),
        temperature_out=np.asarray(temperature, dtype=np.float64),
        pressure=0, pressure_out=0, opt_pH_scale=1, opt_k_carbonic=10,
        opt_k_bisulfate=1,
    )
    return np.asarray(out["fCO2_out"], dtype=np.float64)
