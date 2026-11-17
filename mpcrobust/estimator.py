"""Offset-free estimation: Kalman filter with an integrating disturbance.

Augmented model on the controller grid:

    T[k+1] = a T[k] + b u[k] + c T_out[k] + e + d[k]
    d[k+1] = d[k] + w_d
    y[k]   = T[k] + v

The disturbance d absorbs everything the identified model gets wrong
(building mass, unmeasured gains, parameter mismatch). Feeding the estimate
to the MPC prediction removes steady-state offset (Pannocchia & Rawlings,
2003; Maeder, Borrelli & Morari, 2009).
"""
import numpy as np

from .sysid import ARXModel


class OffsetFreeKF:
    """kind="input":  d enters the state equation every step (as above).
    kind="output": d is a constant offset on the measurement, y = T + d.

    Both remove steady-state offset. They behave differently in a D-step
    delay compensation: an input disturbance is accumulated D times
    (T + D*d), which amplifies estimation errors; an output disturbance is
    not. See README, finding 4.
    """

    def __init__(self, model: ARXModel, T0: float, q_T=0.01, q_d=0.002, r=0.05 ** 2,
                 kind: str = "output"):
        self.m, self.kind = model, kind
        self.x = np.array([T0, 0.0])
        self.P = np.diag([0.5, 0.5])
        if kind == "input":
            self.A = np.array([[model.a, 1.0], [0.0, 1.0]])
            self.H = np.array([1.0, 0.0])
        elif kind == "output":
            self.A = np.array([[model.a, 0.0], [0.0, 1.0]])
            self.H = np.array([1.0, 1.0])
        else:
            raise ValueError(kind)
        self.Q = np.diag([q_T, q_d])
        self.R = r

    @property
    def T(self):
        """Model state used as the MPC initial condition."""
        return float(self.x[0])

    @property
    def d(self):
        """Disturbance entering the prediction model (input kind only)."""
        return float(self.x[1]) if self.kind == "input" else 0.0

    @property
    def y_offset(self):
        """Offset between measured and model temperature (output kind only)."""
        return float(self.x[1]) if self.kind == "output" else 0.0

    def correct(self, y: float):
        S = self.H @ self.P @ self.H + self.R
        K = self.P @ self.H / S
        self.x = self.x + K * (y - self.H @ self.x)
        self.P = (np.eye(2) - np.outer(K, self.H)) @ self.P

    def predict(self, u: float, T_out: float):
        m = self.m
        self.x = self.A @ self.x + np.array([m.b * u + m.c * T_out + m.e, 0.0])
        self.P = self.A @ self.P @ self.A.T + self.Q


class Passthrough:
    """No estimator: the measurement is used as the state, d = 0."""

    def __init__(self, model: ARXModel, T0: float, **_):
        self._T = T0

    T = property(lambda self: self._T)
    d = property(lambda self: 0.0)
    y_offset = property(lambda self: 0.0)

    def correct(self, y):
        self._T = y

    def predict(self, u, T_out):
        pass
