"""Fast sanity tests: run with `pytest -q` (about 30 s)."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mpcrobust.estimator import OffsetFreeKF                      # noqa: E402
from mpcrobust.model import Plant, RoomParams                     # noqa: E402
from mpcrobust.mpc import EconomicMPC, MPCConfig                  # noqa: E402
from mpcrobust.simulate import BASELINE, ROBUST, Controller, Scenario, run  # noqa: E402
from mpcrobust.sysid import ARXModel, identify                    # noqa: E402


@pytest.fixture(scope="module")
def model():
    m, _, _ = identify()
    return m


def test_plant_steady_state():
    """Constant heating settles at T_out + gain * u (energy balance)."""
    p = Plant(RoomParams(), T0=0.0)
    for _ in range(60 * 24 * 20):          # 20 days of 1-min steps
        p.step(1.0, 0.0, 0.0)
    assert p.T_air == pytest.approx(p.steady_state_gain() * 1.0, abs=0.05)


def test_identified_model_is_stable_and_plausible(model):
    assert 0 < model.a < 1
    assert 10 < model.tau_hours < 60                 # slow thermal dynamics
    true_gain = Plant().steady_state_gain()
    assert abs(model.gain - true_gain) / true_gain < 0.3


def test_mpc_respects_input_bounds_and_heats_when_cold(model):
    cfg = MPCConfig()
    mpc = EconomicMPC(model, cfg)
    N = cfg.horizon
    plan, status, _ = mpc.solve(18.0, 0.0, np.zeros(N), np.full(N, 20.0),
                                np.full(N, 23.0), np.full(N, 0.2), 0.0)
    assert status.startswith("optimal")
    assert plan.min() >= 0 and plan.max() <= cfg.u_max + 1e-6
    assert plan[0] > 0.5                              # too cold -> heat


def test_hard_mpc_reports_infeasibility(model):
    cfg = MPCConfig(soft=False)
    mpc = EconomicMPC(model, cfg)
    N = cfg.horizon
    # 10 degC room cannot reach 20 degC in the first 10 minutes
    plan, status, _ = mpc.solve(10.0, 0.0, np.zeros(N), np.full(N, 20.0),
                                np.full(N, 23.0), np.full(N, 0.2), 0.0)
    assert plan is None and "infeasible" in status


def test_offset_free_kf_converges_to_constant_bias(model):
    for kind in ("input", "output"):
        kf = OffsetFreeKF(model, 20.0, kind=kind, q_d=1e-3)
        T = 20.0
        for _ in range(500):                         # true system has +0.05 K/step
            T = model.step(T, 1.0, 0.0) + 0.05
            kf.predict(1.0, 0.0)
            kf.correct(T)
        # one-step prediction of the *measured* temperature must be unbiased
        pred = model.step(kf.T, 1.0, 0.0, kf.d) + kf.y_offset
        true_next = model.step(T, 1.0, 0.0) + 0.05
        assert abs(true_next - pred) < 0.05, kind


def test_closed_loop_nominal_is_comfortable(model):
    for ctl in (BASELINE, ROBUST):
        met, _ = run(model, Scenario(days=1), ctl)
        assert met["violation_Kh"] < 0.5
        assert met["failed_solves"] == 0


def test_plan_buffer_helps_during_outage(model):
    """3 h server outage across the 06:00 comfort step."""
    sc = Scenario(days=2, outages=((27.5, 3.0),), delay_steps=1)
    hold = Controller("hold")
    buffered = Controller("buffer", plan_buffer=True)
    v_hold = run(model, sc, hold)[0]["violation_Kh"]
    v_buf = run(model, sc, buffered)[0]["violation_Kh"]
    assert v_buf < v_hold
