"""Synthetic exogenous signals: weather, internal gains, comfort band, CO2.

All profiles are generated on the fine (1-min) grid and can be averaged to
the controller grid (10 min) with `to_coarse`.
"""
from dataclasses import dataclass

import numpy as np

FINE_DT = 1 / 60    # h, plant simulation step
CTRL_DT = 1 / 6     # h, controller sampling time (10 min)
SUB = int(round(CTRL_DT / FINE_DT))  # fine steps per control step


@dataclass
class Profiles:
    t: np.ndarray        # fine time grid [h]
    T_out: np.ndarray    # outdoor temperature [degC]
    q_gain: np.ndarray   # unmeasured internal + solar gains [kW]
    lower: np.ndarray    # comfort band lower bound [degC]
    upper: np.ndarray    # comfort band upper bound [degC]
    co2: np.ndarray      # CO2 intensity of delivered heat [kg/kWh]


def to_coarse(x: np.ndarray, sub: int = SUB) -> np.ndarray:
    """Average a fine-grid signal over each control interval."""
    n = len(x) // sub
    return x[: n * sub].reshape(n, sub).mean(axis=1)


def comfort_band(t: np.ndarray):
    """Occupied hours 06:00-22:00: [20, 23] degC; night set-back: [17, 23] degC."""
    hour = np.mod(t, 24.0)
    day = (hour >= 6.0) & (hour < 22.0)
    lower = np.where(day, 20.0, 17.0)
    upper = np.full_like(t, 23.0)
    return lower, upper


def make_profiles(days: float, seed: int = 0, T_mean: float = 2.0) -> Profiles:
    """Winter week with daily temperature swing, weather drift and gains."""
    rng = np.random.default_rng(seed)
    n = int(round(days * 24 / FINE_DT))
    t = np.arange(n) * FINE_DT
    hour = np.mod(t, 24.0)

    # outdoor temperature: daily cycle (peak 15:00) + slow weather drift
    drift = np.cumsum(rng.normal(0, 1, n))
    kernel = np.ones(6 * 60) / (6 * 60)                      # 6 h smoothing
    drift = np.convolve(drift - drift.mean(), kernel, mode="same")
    drift = 3.0 * drift / (np.abs(drift).max() + 1e-9)
    T_out = T_mean + 4.0 * np.cos(2 * np.pi * (hour - 15.0) / 24.0) + drift

    # internal gains: base load + occupancy peaks + solar on random days
    base = 0.10
    occ = 0.30 * (((hour >= 6.5) & (hour < 8.5)) | ((hour >= 17.5) & (hour < 22.5)))
    sunny = rng.uniform(0.0, 1.0, int(np.ceil(days)) + 1)[(t // 24).astype(int)]
    solar = 0.9 * sunny * np.clip(np.sin(np.pi * (hour - 8.0) / 8.0), 0, None) * (
        (hour > 8.0) & (hour < 16.0))
    q_gain = base + occ + solar

    lower, upper = comfort_band(t)

    # CO2 intensity of heat: higher in the evening peak of the heat network
    co2 = 0.20 + 0.06 * np.cos(2 * np.pi * (hour - 19.0) / 24.0)

    return Profiles(t, T_out, q_gain, lower, upper, co2)
