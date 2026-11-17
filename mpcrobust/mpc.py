"""Linear economic MPC for a smart-thermostat zone.

Decision: heating power over the horizon. Objective: CO2-weighted energy,
a small input-rate penalty, and (soft) comfort-band violations.

The optimisation problem is built once with cvxpy Parameters, so every
10-min solve only updates numbers (it is a QP, solved with OSQP).
"""
from dataclasses import dataclass
import time

import cvxpy as cp
import numpy as np

from .sysid import ARXModel


@dataclass
class MPCConfig:
    horizon: int = 36            # 36 x 10 min = 6 h
    u_max: float = 4.0           # kW
    soft: bool = True            # soft comfort constraints (slack variables)
    rho_lin: float = 5.0         # linear slack penalty  [per K per step]
    rho_quad: float = 50.0       # quadratic slack penalty [per K^2 per step]
    r_du: float = 0.2            # input-rate penalty [per kW^2]
    tightening: float = 0.0      # back-off from the comfort band [K]


class EconomicMPC:
    def __init__(self, model: ARXModel, cfg: MPCConfig = MPCConfig()):
        self.model, self.cfg = model, cfg
        N, m = cfg.horizon, model

        self.p_T0 = cp.Parameter(name="T0")
        self.p_d = cp.Parameter(name="d")                # estimated offset
        self.p_Tout = cp.Parameter(N, name="T_out")
        self.p_lo = cp.Parameter(N, name="lower")
        self.p_hi = cp.Parameter(N, name="upper")
        self.p_w = cp.Parameter(N, nonneg=True, name="co2")
        self.p_uprev = cp.Parameter(name="u_prev")

        self.u = cp.Variable(N)
        self.T = cp.Variable(N + 1)
        cons = [self.T[0] == self.p_T0,
                self.T[1:] == m.a * self.T[:-1] + m.b * self.u
                + m.c * self.p_Tout + m.e + self.p_d,
                self.u >= 0, self.u <= cfg.u_max]
        lo = self.p_lo + cfg.tightening
        hi = self.p_hi - cfg.tightening
        du = cp.hstack([self.u[0] - self.p_uprev, cp.diff(self.u)])
        cost = cp.sum(cp.multiply(self.p_w, self.u)) * m.dt + cfg.r_du * cp.sum_squares(du)
        if cfg.soft:
            self.s = cp.Variable(N, nonneg=True)
            cons += [self.T[1:] >= lo - self.s, self.T[1:] <= hi + self.s]
            cost += cfg.rho_lin * cp.sum(self.s) + cfg.rho_quad * cp.sum_squares(self.s)
        else:
            self.s = None
            cons += [self.T[1:] >= lo, self.T[1:] <= hi]
        self.prob = cp.Problem(cp.Minimize(cost), cons)
        assert self.prob.is_dpp()

    def solve(self, T0, d, T_out, lower, upper, co2, u_prev):
        """Return (plan, status, solve_time_s). plan is None if infeasible."""
        self.p_T0.value, self.p_d.value = float(T0), float(d)
        self.p_Tout.value, self.p_lo.value, self.p_hi.value = T_out, lower, upper
        self.p_w.value, self.p_uprev.value = co2, float(u_prev)
        t0 = time.perf_counter()
        try:
            self.prob.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-5,
                            eps_rel=1e-5, max_iter=20000)
        except cp.SolverError:
            return None, "solver_error", time.perf_counter() - t0
        dt = time.perf_counter() - t0
        if self.prob.status not in ("optimal", "optimal_inaccurate"):
            return None, self.prob.status, dt
        plan = np.clip(np.asarray(self.u.value), 0.0, self.cfg.u_max)
        return plan, self.prob.status, dt
