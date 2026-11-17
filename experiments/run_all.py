"""Run every robustness experiment and write figures/ and results/.

Usage:  python experiments/run_all.py [--quick]
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from multiprocessing import Pool

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from mpcrobust.model import RoomParams                       # noqa: E402
from mpcrobust.mpc import MPCConfig                          # noqa: E402
from mpcrobust.simulate import BASELINE, ROBUST, Controller, Scenario, run  # noqa: E402
from mpcrobust.sysid import identify                         # noqa: E402

FIG = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results")
COL = {"baseline": "#c0392b", "robust": "#1f5fbf", "hard": "#7f8c8d"}
HARD = Controller("hard", mpc=MPCConfig(soft=False))
MODEL = None


def _init():
    global MODEL
    MODEL, _, _ = identify()


def _job(args):
    sc, ctl = args
    met, _ = run(MODEL, sc, ctl)
    return met


def batch(pool, jobs):
    return pool.map(_job, jobs)


def style(ax):
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


# --------------------------------------------------------------------------
def exp_identification(model, report, val):
    T_val, sim = val
    t = np.arange(len(T_val)) / 6
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.plot(t, T_val, color="k", lw=1, label="measured (2R2C plant + noise)")
    ax.plot(t, sim, color=COL["robust"], lw=1.4, label="identified 1st-order model, free run")
    ax.set_xlabel("time [h]"), ax.set_ylabel("air temperature [°C]")
    ax.set_title(f"Model validation on unseen data, free-run RMSE = {report['free_run_rmse']:.2f} K")
    ax.legend(frameon=False), style(ax)
    fig.tight_layout(), fig.savefig(f"{FIG}/01_identification.png", dpi=150), plt.close(fig)


def exp_nominal(model):
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.5), sharex=True)
    out = {}
    for ctl in (BASELINE, ROBUST):
        met, tr = run(model, Scenario(days=4, plant=RoomParams().scaled(1, 0.7)), ctl)
        out[ctl.name] = met
        h = tr["t"]
        axes[0].plot(h, tr["T"], color=COL[ctl.name], lw=1.3,
                     label=f"{ctl.name}: {met['violation_Kh']:.2f} K·h outside band")
        axes[1].plot(h, tr["u"], color=COL[ctl.name], lw=1)
    axes[0].fill_between(h, tr["lower"], tr["upper"], color="0.85", label="comfort band")
    axes[0].set_ylabel("room temp. [°C]"), axes[0].set_ylim(16.5, 24.5)
    axes[0].legend(frameon=False, loc="upper center", ncol=3)
    axes[0].set_title("Apartment 30% leakier than the identified model (R × 0.7)")
    axes[1].set_ylabel("heating [kW]"), axes[1].set_xlabel("time [h]")
    for a in axes:
        style(a)
    fig.tight_layout(), fig.savefig(f"{FIG}/02_trajectories_mismatch.png", dpi=150), plt.close(fig)
    return out


def exp_monte_carlo(pool, n, days):
    rng = np.random.default_rng(42)
    cs, rs = np.exp(rng.normal(0, 0.25, n)), np.exp(rng.normal(0, 0.25, n))
    scen = [Scenario(days=days, seed=i, plant=RoomParams().scaled(c, r))
            for i, (c, r) in enumerate(zip(cs, rs))]
    res = {c.name: batch(pool, [(s, c) for s in scen]) for c in (BASELINE, ROBUST)}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for c in res:
        v = [m["violation_Kh"] for m in res[c]]
        axes[0].scatter(rs, v, color=COL[c], s=22, label=c, alpha=0.8)
    axes[0].set_xscale("log"), axes[0].set_xlabel("plant resistance scale (×R)")
    axes[0].set_xticks([0.6, 0.8, 1.0, 1.25, 1.6], ["0.6", "0.8", "1.0", "1.25", "1.6"])
    axes[0].minorticks_off()
    axes[0].set_ylabel("comfort violation [K·h / week]"), axes[0].legend(frameon=False)
    axes[0].set_title(f"{n} random apartments (C, R ~ ±25 %)")
    e = [[m["energy_kWh"] / days for m in res[c]] for c in res]
    axes[1].boxplot(e, tick_labels=list(res)), axes[1].set_ylabel("energy [kWh/day]")
    axes[1].set_title("Energy cost of robustness")
    for a in axes:
        style(a)
    fig.tight_layout(), fig.savefig(f"{FIG}/03_monte_carlo_mismatch.png", dpi=150), plt.close(fig)
    return {"c_scale": cs.tolist(), "r_scale": rs.tolist(), **res}


def sweep(pool, name, values, make, xlabel, title, days, fname, controllers=(BASELINE, ROBUST)):
    res = {c.name: batch(pool, [(make(v, days), c) for v in values]) for c in controllers}
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for c in res:
        ax.plot(values, [m["violation_Kh"] for m in res[c]], "o-", color=COL[c], label=c)
    ax.set_xlabel(xlabel), ax.set_ylabel("comfort violation [K·h / week]")
    ax.set_title(title), ax.legend(frameon=False), style(ax)
    fig.tight_layout(), fig.savefig(f"{FIG}/{fname}", dpi=150), plt.close(fig)
    return {"values": list(values), **res}


def exp_hard_vs_soft(pool, days):
    scen = {"nominal": Scenario(days=days),
            "leaky (R×0.7)": Scenario(days=days, plant=RoomParams().scaled(1, 0.7)),
            "forecast +3 K": Scenario(days=days, forecast_bias=3.0),
            "server outages": Scenario(days=days, outages=OUTAGES)}
    soft = Controller("baseline")
    res = {}
    for name, sc in scen.items():
        mh, ms = batch(pool, [(sc, HARD), (sc, soft)])
        res[name] = {"hard": mh, "soft": ms}
    return res


def exp_estimator_delay_map(pool, days):
    """Which offset-free design survives delay compensation?"""
    rows = [("input dist., q_d=5e-5", dict(kf_kind="input", kf_q_d=5e-5), MPCConfig(tightening=0.2)),
            ("input dist., q_d=1e-3", dict(kf_kind="input", kf_q_d=1e-3), MPCConfig(tightening=0.2)),
            ("output dist., q_d=1e-3", dict(kf_kind="output", kf_q_d=1e-3), MPCConfig(tightening=0.2)),
            ("output dist., aggressive MPC", dict(kf_kind="output", kf_q_d=1e-3),
             MPCConfig(tightening=0.2, r_du=0.002))]
    delays = [0, 1, 3, 6]
    jobs = []
    for _, kw, mcfg in rows:
        ctl = Controller("x", offset_free=True, delay_comp=True, plan_buffer=True, mpc=mcfg, **kw)
        jobs += [(Scenario(days=days, delay_steps=D), ctl) for D in delays]
    mets = batch(pool, jobs)
    grid = np.array([m["violation_Kh"] for m in mets]).reshape(len(rows), len(delays))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    im = ax.imshow(np.log10(grid + 1e-2), cmap="RdYlGn_r", aspect="auto", vmin=-2, vmax=1.5)
    ax.set_xticks(range(len(delays)), [f"{10 * d} min" for d in delays])
    ax.set_yticks(range(len(rows)), [r[0] for r in rows])
    ax.set_xlabel("actuation delay (with delay compensation)")
    for i in range(len(rows)):
        for j in range(len(delays)):
            ax.text(j, i, f"{grid[i, j]:.2f}", ha="center", va="center", fontsize=9)
    ax.set_title("Comfort violation [K·h / week]")
    fig.colorbar(im, label="log10 violation")
    fig.tight_layout(), fig.savefig(f"{FIG}/08_estimator_vs_delay.png", dpi=150), plt.close(fig)
    return {"rows": [r[0] for r in rows], "delays": delays, "violation_Kh": grid.tolist()}


OUTAGES = ((27.5, 3.0), (75.5, 3.0), (123.5, 3.0))   # 03:30-06:30 on days 2, 4, 6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="short runs for testing")
    args = ap.parse_args()
    days, n_mc = (2, 6) if args.quick else (7, 30)
    os.makedirs(FIG, exist_ok=True), os.makedirs(RES, exist_ok=True)

    model, report, val = identify()
    summary = {"model": {"a": model.a, "b": model.b, "c": model.c, "e": model.e,
                         "tau_h": model.tau_hours, "gain_K_per_kW": model.gain, **report}}
    exp_identification(model, report, val)
    summary["trajectories_R0.7"] = exp_nominal(model)

    with Pool(min(8, os.cpu_count()), initializer=_init) as pool:
        summary["nominal"] = dict(zip(["baseline", "robust"], batch(
            pool, [(Scenario(days=days), BASELINE), (Scenario(days=days), ROBUST)])))
        summary["monte_carlo"] = exp_monte_carlo(pool, n_mc, days)
        summary["forecast_bias"] = sweep(
            pool, "bias", [-4, -2, 0, 2, 4],
            lambda v, d: Scenario(days=d, forecast_bias=v),
            "outdoor forecast bias [K]", "Weather-forecast error", days, "04_forecast_bias.png")
        summary["internal_gains"] = sweep(
            pool, "gains", [0.0, 0.5, 1.0, 2.0, 3.0],
            lambda v, d: Scenario(days=d, gain_scale=v),
            "unmeasured gains (× nominal)", "Occupancy / solar disturbances", days,
            "05_internal_gains.png")
        summary["delay"] = sweep(
            pool, "delay", [0, 1, 2, 3, 6],
            lambda v, d: Scenario(days=d, delay_steps=v),
            "actuation delay [×10 min]", "Communication delay", days, "06_delay.png")
        summary["outages"] = sweep(
            pool, "loss", [0.0, 0.2, 0.4, 0.6],
            lambda v, d: Scenario(days=d, loss_prob=v, outages=OUTAGES, delay_steps=1),
            "random packet-loss probability", "Packet loss + 3 h server outages", days,
            "07_packet_loss_outages.png")
        summary["sensor"] = sweep(
            pool, "quant", [0.0, 0.1, 0.25, 0.5],
            lambda v, d: Scenario(days=d, quantization=v, sensor_noise=0.1),
            "thermostat resolution [K]", "Sensor quantization + noise", days,
            "09_sensor_quantization.png")
        summary["hard_vs_soft"] = exp_hard_vs_soft(pool, days)
        summary["estimator_vs_delay"] = exp_estimator_delay_map(pool, days)

    with open(f"{RES}/summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: summary[k] for k in ("model", "nominal")}, indent=1))


if __name__ == "__main__":
    main()
