"""Thermal models of an apartment.

Units used throughout the project:
    time        hours [h]
    temperature degrees Celsius [degC]
    power       kilowatts [kW]
    energy      kilowatt-hours [kWh]
    capacity    kWh/K
    resistance  K/kW

The "true" plant is a 2R2C grey-box model (indoor air + building mass).
The controller never sees it: it only gets a first-order model identified
from data (see sysid.py), so there is always a structural model mismatch,
as there would be in a real apartment.
"""
from dataclasses import dataclass, replace

import numpy as np
from scipy.linalg import expm


@dataclass(frozen=True)
class RoomParams:
    """2R2C parameters of a typical apartment.

    Ca  : capacity of air + furniture                 [kWh/K]
    Cm  : capacity of walls / floor (building mass)   [kWh/K]
    Ram : resistance air <-> mass                     [K/kW]
    Rao : resistance air <-> outside (windows, vent.) [K/kW]
    Rmo : resistance mass <-> outside (envelope)      [K/kW]
    """
    Ca: float = 0.3
    Cm: float = 3.0
    Ram: float = 1.0
    Rao: float = 20.0
    Rmo: float = 20.0

    def scaled(self, c_scale: float = 1.0, r_scale: float = 1.0) -> "RoomParams":
        """Return parameters with all capacities / resistances scaled."""
        return replace(self, Ca=self.Ca * c_scale, Cm=self.Cm * c_scale,
                       Ram=self.Ram * r_scale, Rao=self.Rao * r_scale,
                       Rmo=self.Rmo * r_scale)


def continuous_matrices(p: RoomParams):
    """State x = [T_air, T_mass]; inputs u = heating power, d = [T_out, q_gain].

    q_gain (internal + solar gains) acts on the air node.
    """
    A = np.array([
        [-(1 / p.Ram + 1 / p.Rao) / p.Ca, 1 / (p.Ram * p.Ca)],
        [1 / (p.Ram * p.Cm), -(1 / p.Ram + 1 / p.Rmo) / p.Cm],
    ])
    Bu = np.array([[1 / p.Ca], [0.0]])
    Bd = np.array([
        [1 / (p.Rao * p.Ca), 1 / p.Ca],
        [1 / (p.Rmo * p.Cm), 0.0],
    ])
    return A, Bu, Bd


def discretize(A, B, dt):
    """Exact zero-order-hold discretization of x' = A x + B w."""
    n, m = A.shape[0], B.shape[1]
    M = np.zeros((n + m, n + m))
    M[:n, :n] = A
    M[:n, n:] = B
    Md = expm(M * dt)
    return Md[:n, :n], Md[:n, n:]


class Plant:
    """2R2C apartment simulated with a fine time step (default 1 min).

    The controller runs every 10 min, so the plant is integrated in
    sub-steps: this also creates the discretization mismatch between the
    continuous system and the controller's discrete-time model.
    """

    def __init__(self, params: RoomParams = RoomParams(), dt: float = 1 / 60,
                 T0: float = 20.0):
        self.params = params
        self.dt = dt
        A, Bu, Bd = continuous_matrices(params)
        self.Ad, Bw = discretize(A, np.hstack([Bu, Bd]), dt)
        self.Bu, self.Bd = Bw[:, :1], Bw[:, 1:]
        self.x = np.array([T0, T0], dtype=float)

    @property
    def T_air(self) -> float:
        return float(self.x[0])

    def step(self, u: float, T_out: float, q: float) -> float:
        """Advance one fine step and return the new air temperature."""
        self.x = self.Ad @ self.x + self.Bu[:, 0] * u + self.Bd @ np.array([T_out, q])
        return self.T_air

    def steady_state_gain(self) -> float:
        """Static gain from heating power to air temperature [K/kW]."""
        A, Bu, _ = continuous_matrices(self.params)
        return float((-np.linalg.solve(A, Bu))[0, 0])
