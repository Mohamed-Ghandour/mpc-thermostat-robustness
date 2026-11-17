"""Closed-loop simulation: server-side MPC -> network -> thermostat -> apartment.

One call to `run` simulates the whole chain for a given *scenario* (what
the world does to the controller) and *controller* (which countermeasures
are switched on), and returns trajectories plus metrics.
"""
from dataclasses import dataclass, field

import numpy as np

from .estimator import OffsetFreeKF, Passthrough
from .model import Plant, RoomParams
from .mpc import EconomicMPC, MPCConfig
from .profiles import CTRL_DT, FINE_DT, SUB, make_profiles, to_coarse
from .sysid import ARXModel


@dataclass
class Scenario:
    """Sources of uncertainty / failure the controller is exposed to."""
    days: float = 7
    seed: int = 0
    plant: RoomParams = field(default_factory=RoomParams)   # true apartment
    forecast_bias: float = 0.0     # K, systematic weather-forecast error
    forecast_sigma: float = 0.0    # K, random forecast error at end of horizon
    gain_scale: float = 1.0        # multiplies the (unmeasured) internal gains
    delay_steps: int = 0           # actuation latency in control steps (10 min)
    loss_prob: float = 0.0         # probability a command packet is lost
    outages: tuple = ()            # server outages: ((start_h, duration_h), ...)
    sensor_noise: float = 0.05     # K, std of thermostat measurement noise
    quantization: float = 0.0      # K, sensor resolution (0 = none)
    T0: float = 20.0


@dataclass
class Controller:
    """Which robustness countermeasures are active."""
    name: str = "baseline"
    offset_free: bool = False      # Kalman filter with disturbance state
    delay_comp: bool = False       # predict the state at actuation time
    plan_buffer: bool = False      # thermostat stores the whole MPC plan
    kf_q_d: float = 1e-3           # disturbance random-walk variance (per step)
    kf_kind: str = "output"        # "output" or "input" disturbance model
    mpc: MPCConfig = field(default_factory=MPCConfig)


BASELINE = Controller("baseline")
ROBUST = Controller("robust", offset_free=True, delay_comp=True, plan_buffer=True,
                    mpc=MPCConfig(tightening=0.2))


def _measure(T, sc: Scenario, rng):
    y = T + rng.normal(0.0, sc.sensor_noise)
    if sc.quantization > 0:
        y = np.round(y / sc.quantization) * sc.quantization
    return y


def run(model: ARXModel, sc: Scenario, ctl: Controller, mpc: EconomicMPC = None):
    rng = np.random.default_rng(1000 + sc.seed)
    N, D = ctl.mpc.horizon, sc.delay_steps
    K = int(round(sc.days * 24 / CTRL_DT))
    pad = N + D + 2
    prof = make_profiles(sc.days + pad * CTRL_DT / 24 + 0.1, seed=sc.seed)
    To_c, lo_c, hi_c, co2_c = (to_coarse(x) for x in (prof.T_out, prof.lower,
                                                       prof.upper, prof.co2))
    q_fine = prof.q_gain * sc.gain_scale
    mpc = mpc or EconomicMPC(model, ctl.mpc)
    plant = Plant(sc.plant, T0=sc.T0)
    est = (OffsetFreeKF if ctl.offset_free else Passthrough)(
        model, sc.T0, q_d=ctl.kf_q_d, kind=ctl.kf_kind, r=sc.sensor_noise ** 2 + sc.quantization ** 2 / 12 + 1e-6)

    expected = {}                  # step -> input the controller believes applies
    inbox = {}                     # arrival step -> (plan, sent step)
    held_plan, held_from = np.zeros(1), 0
    u_applied = np.zeros(K)
    T_fine = np.empty(K * SUB)
    log = {"d_hat": np.zeros(K), "status_fail": 0, "solve_time": []}

    for k in range(K):
        # 1) measurement + estimator correction
        est.correct(_measure(plant.T_air, sc, rng))
        T_hat, d_hat, y_off = est.T, est.d, est.y_offset
        log["d_hat"][k] = d_hat + y_off

        # 2) where will the system be when this command takes effect?
        start = k
        if ctl.delay_comp and D > 0:
            for j in range(k, k + D):
                T_hat = model.step(T_hat, expected.get(j, 0.0), To_c[j], d_hat)
            start = k + D
        u_prev = expected.get(start - 1, u_applied[k - 1] if k > 0 else 0.0)

        # 3) forecasts as seen by the server (true future + forecast error)
        lead = np.sqrt(np.arange(1, N + 1) / N)
        To_fc = (To_c[start:start + N] + sc.forecast_bias
                 + sc.forecast_sigma * lead * rng.normal(0, 1, N))

        # comfort band is on the measured temperature = model state + offset
        plan, status, t_solve = mpc.solve(T_hat, d_hat, To_fc, lo_c[start:start + N] - y_off,
                                          hi_c[start:start + N] - y_off,
                                          co2_c[start:start + N], u_prev)
        log["solve_time"].append(t_solve)
        if plan is None:                       # infeasible: hold previous input
            log["status_fail"] += 1
            plan = np.full(N, u_prev)

        # 4) send over the network (arrives after D steps, may be lost)
        arrival = k + D
        t_now = k * CTRL_DT
        down = any(a <= t_now < a + b for a, b in sc.outages)
        if rng.uniform() >= sc.loss_prob and not down:
            inbox[arrival] = (plan, arrival)
        for j in range(N):                     # controller's belief
            if ctl.plan_buffer:
                expected[arrival + j] = plan[j]
            elif j == 0:
                for jj in range(arrival, arrival + N):
                    expected[jj] = plan[0]

        # 5) thermostat decides what to apply now
        if k in inbox:
            held_plan, held_from = inbox.pop(k)
        idx = k - held_from
        if ctl.plan_buffer:
            u = held_plan[min(idx, len(held_plan) - 1)]
        else:
            u = held_plan[0]
        u_applied[k] = u

        # 6) apartment evolves for 10 min (fine steps); estimator predicts
        for j in range(SUB):
            i = k * SUB + j
            T_fine[i] = plant.step(u, prof.T_out[i], q_fine[i])
        est.predict(u, To_c[k])

    n = K * SUB
    lower, upper = prof.lower[:n], prof.upper[:n]
    under = np.clip(lower - T_fine, 0, None)
    over = np.clip(T_fine - upper, 0, None)
    co2_fine = np.repeat(co2_c[:K], SUB)
    u_fine = np.repeat(u_applied, SUB)
    metrics = {
        "violation_Kh": float(np.sum(under + over) * FINE_DT),
        "max_violation_K": float(np.max(under + over)),
        "time_outside_pct": float(100 * np.mean((under + over) > 0.05)),
        "energy_kWh": float(np.sum(u_applied) * CTRL_DT),
        "co2_kg": float(np.sum(u_fine * co2_fine) * FINE_DT),
        "failed_solves": int(log["status_fail"]),
        "mean_solve_ms": float(1000 * np.mean(log["solve_time"])),
    }
    traj = {"t": prof.t[:n], "T": T_fine, "lower": lower, "upper": upper,
            "T_out": prof.T_out[:n], "u": u_fine, "d_hat": np.repeat(log["d_hat"], SUB)}
    return metrics, traj
