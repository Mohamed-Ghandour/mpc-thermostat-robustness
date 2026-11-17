"""Identification of the controller model from (simulated) measurement data.

The controller uses an affine first-order ARX model on the 10-min grid:

    T[k+1] = a T[k] + b u[k] + c T_out[k] + e

fitted by least squares. This mimics the "identified linear dynamic model
for room temperature prediction" the MPC builds on: it is simple, and it is
wrong in a structured way (it ignores the building mass and the gains).
"""
from dataclasses import dataclass

import numpy as np

from .model import Plant, RoomParams
from .profiles import CTRL_DT, SUB, make_profiles, to_coarse


@dataclass(frozen=True)
class ARXModel:
    a: float
    b: float
    c: float
    e: float
    dt: float = CTRL_DT

    def step(self, T, u, T_out, d=0.0):
        return self.a * T + self.b * u + self.c * T_out + self.e + d

    @property
    def tau_hours(self) -> float:
        """Equivalent continuous time constant of the identified model."""
        return -self.dt / np.log(self.a)

    @property
    def gain(self) -> float:
        """Static gain heating -> temperature [K/kW]."""
        return self.b / (1 - self.a)


def generate_id_data(days: float = 14, seed: int = 1, u_max: float = 4.0,
                     noise_std: float = 0.05, params: RoomParams = RoomParams()):
    """Excite the plant with a random multi-level heating sequence."""
    rng = np.random.default_rng(seed)
    prof = make_profiles(days, seed=seed)
    n_ctrl = len(prof.t) // SUB
    u = np.empty(n_ctrl)
    k = 0
    while k < n_ctrl:                     # random levels held 1-6 h
        hold = int(rng.integers(6, 37))
        u[k:k + hold] = rng.uniform(0.0, u_max)
        k += hold
    plant = Plant(params, T0=19.0)
    T_meas = np.empty(n_ctrl + 1)
    T_meas[0] = plant.T_air + rng.normal(0, noise_std)
    for k in range(n_ctrl):
        for j in range(SUB):
            i = k * SUB + j
            plant.step(u[k], prof.T_out[i], prof.q_gain[i])
        T_meas[k + 1] = plant.T_air + rng.normal(0, noise_std)
    T_out = to_coarse(prof.T_out)[:n_ctrl]
    return T_meas, u, T_out


def fit_arx(T_meas, u, T_out) -> ARXModel:
    Phi = np.column_stack([T_meas[:-1], u, T_out, np.ones_like(u)])
    theta, *_ = np.linalg.lstsq(Phi, T_meas[1:], rcond=None)
    return ARXModel(*map(float, theta))


def refine_output_error(model: ARXModel, T_meas, u, T_out) -> ARXModel:
    """Refine the ARX fit by minimising the free-run (multi-step) error.

    One-step least squares favours the fast air dynamics; for MPC the
    multi-hour prediction matters, so we re-fit on simulation error.
    """
    from scipy.optimize import least_squares

    def resid(theta):
        sim = simulate_arx(ARXModel(*theta), T_meas[0], u, T_out)
        return sim - T_meas

    x0 = [model.a, model.b, model.c, model.e]
    sol = least_squares(resid, x0, bounds=([0.5, 0, -1, -10], [0.9999, 5, 1, 10]))
    return ARXModel(*map(float, sol.x))


def simulate_arx(model: ARXModel, T0, u, T_out):
    """Free-run (multi-step) simulation, the honest test of a control model."""
    T = np.empty(len(u) + 1)
    T[0] = T0
    for k in range(len(u)):
        T[k + 1] = model.step(T[k], u[k], T_out[k])
    return T


def identify(seed: int = 1):
    """Fit on 14 days, validate free-run on 7 independent days."""
    data = generate_id_data(14, seed=seed)
    model = refine_output_error(fit_arx(*data), *data)
    T_val, u_val, To_val = generate_id_data(7, seed=seed + 100)
    one_step = model.a * T_val[:-1] + model.b * u_val + model.c * To_val + model.e
    sim = simulate_arx(model, T_val[0], u_val, To_val)
    report = {
        "one_step_rmse": float(np.sqrt(np.mean((one_step - T_val[1:]) ** 2))),
        "free_run_rmse": float(np.sqrt(np.mean((sim - T_val) ** 2))),
        "free_run_max_err": float(np.max(np.abs(sim - T_val))),
    }
    return model, report, (T_val, sim)
