"""Matplotlib figures for the input data and the optimisation results.

Every function returns the ``Figure`` and optionally saves it, so the same code works in a
script (``python main.py``) and in a notebook (``plot_schedule(results, data);``).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .data_loader import InputData
from .model import Results


def _finish(fig: plt.Figure, save_to: Path | str | None) -> plt.Figure:
    fig.tight_layout()
    if save_to is not None:
        Path(save_to).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_to, dpi=150)
    return fig


def plot_inputs(data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly prices (with tariffs) and available PV / load preferences, side by side."""
    h = data.hours
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))

    ax1.step(h, data.energy_price, where="mid", label="energy price", color="k")
    ax1.step(h, data.energy_price + data.import_tariff, where="mid", ls="--", label="price + import tariff")
    ax1.step(h, data.energy_price - data.export_tariff, where="mid", ls=":", label="price - export tariff")
    ax1.set(xlabel="hour", ylabel="DKK/kWh", title="Electricity prices")
    ax1.legend(fontsize=8)

    ax2.fill_between(h, data.pv_available, step="mid", alpha=0.4, color="orange", label="PV available")
    ax2.axhline(data.load_max_kWh, color="C3", ls="--", label="max load")
    if data.load_min_kWh > 0:
        ax2.axhline(data.load_min_kWh, color="C3", ls=":", label="min load")
    if data.reference_load is not None:
        ax2.step(h, data.reference_load, where="mid", color="C0", label="reference load")
    ax2.set(xlabel="hour", ylabel="kWh/h", title="PV and load preferences")
    ax2.legend(fontsize=8)
    fig.suptitle(f"Input data - {data.question}", fontsize=11)
    return _finish(fig, save_to)


def plot_schedule(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Optimal schedule: load, PV used, import/export, with prices on a second axis."""
    hr = results.hourly
    h = hr.index.to_numpy()
    fig, ax = plt.subplots(figsize=(11, 4.2))

    width = 0.8
    if "load" in hr:
        ax.bar(h, hr["load"], width, color="C0", alpha=0.7, label="load")
    if "pv" in hr:
        ax.bar(h, -hr["pv"], width, color="orange", alpha=0.7, label="PV used (negative = generation)")
    if "pv_available" in hr and "pv" in hr:
        ax.step(h, -hr["pv_available"], where="mid", color="orange", ls="--", lw=1, label="PV available")
    if "import" in hr and "export" in hr:
        ax.plot(h, hr["import"] - hr["export"], "k.-", label="net import (+) / export (-)")
    if "reference_load" in hr:
        ax.step(h, hr["reference_load"], where="mid", color="C0", ls=":", label="reference load")
    ax.axhline(0, color="grey", lw=0.8)
    ax.set(xlabel="hour", ylabel="kWh/h", title=f"Optimal schedule - {results.question} (cost {results.objective:.1f} DKK)")

    ax2 = ax.twinx()
    ax2.step(h, hr["price"], where="mid", color="C3", lw=1.2, label="energy price")
    ax2.set_ylabel("DKK/kWh", color="C3")

    lines, labels = ax.get_legend_handles_labels()
    l2, lb2 = ax2.get_legend_handles_labels()
    ax.legend(lines + l2, labels + lb2, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    return _finish(fig, save_to)


def plot_duals(results: Results, data: InputData, save_to: Path | str | None = None) -> plt.Figure:
    """Hourly dual variables (all ``dual_*`` columns) against the price signals."""
    hr = results.hourly
    dual_cols = [c for c in hr.columns if c.startswith("dual_")]
    fig, ax = plt.subplots(figsize=(11, 4))
    h = hr.index.to_numpy()
    for c in dual_cols:
        ax.step(h, hr[c], where="mid", label=c.removeprefix("dual_"))
    ax.step(h, data.energy_price + data.import_tariff, where="mid", color="grey", ls="--", lw=1, label="price + import tariff")
    ax.step(h, data.energy_price - data.export_tariff, where="mid", color="grey", ls=":", lw=1, label="price - export tariff")
    ax.set(xlabel="hour", ylabel="DKK/kWh", title=f"Dual variables - {results.question}")
    ax.legend(fontsize=8, ncol=3)
    return _finish(fig, save_to)


def plot_min_energy_comparison(
    constrained: Results,
    unconstrained: Results,
    data: InputData,
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Question 3.(e): the E_min-constrained schedule against the reference profile and against the
    unconstrained consumer of Question 2.(c), solved on the same data.

    Three panels on a shared hour axis:
      1. hourly load (kWh/h): reference profile, unconstrained 2.(c) load, constrained Q3 load;
      2. load shift Q3 - 2.(c) (kWh/h): where the extra energy required by E_min is placed;
      3. marginal value of energy at the connection point, lambda_t = dual of the power balance
         (DKK/kWh), against the E_min multiplier mu (DKK/kWh). Hours with mu > lambda_t are shaded:
         there the Q3 load is expected above the reference (if not at a load bound).
    ``constrained`` must contain the ``dual_balance`` column and ``duals["daily_energy_min"]``.
    """
    ref_color, unc_color, con_color = "#8a8a85", "#2a78d6", "#eb6834"
    h = constrained.hourly.index.to_numpy()
    ref = data.reference_load
    l_unc = unconstrained.hourly["load"].to_numpy()
    l_con = constrained.hourly["load"].to_numpy()
    lam = constrained.hourly["dual_balance"].to_numpy()
    mu = constrained.duals.get("daily_energy_min", 0.0)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(11, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.6, 1.8]}
    )
    for ax in (ax1, ax2, ax3):
        ax.grid(axis="y", color="#e4e3dc", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    ax1.step(h, ref, where="mid", color=ref_color, ls="--", lw=1.5, label=f"reference ({ref.sum():.1f} kWh)")
    ax1.plot(h, l_unc, "o-", color=unc_color, lw=2, ms=4,
             label=f"without E_min ({l_unc.sum():.1f} kWh)")
    ax1.plot(h, l_con, "s-", color=con_color, lw=2, ms=4,
             label=f"with E_min = {data.min_daily_energy_kWh:g} kWh ({l_con.sum():.1f} kWh)")
    ax1.set(ylabel="load [kWh/h]", title="Effect of the minimum daily energy requirement")
    ax1.legend(fontsize=8, loc="upper left", frameon=False)

    shift = l_con - l_unc
    ax2.bar(h, shift, 0.7, color=con_color)
    ax2.axhline(0, color="#55544f", lw=0.8)
    ax2.set(ylabel="load shift\n[kWh/h]")
    ax2.text(0.99, 0.95, f"with - without E_min, total {shift.sum():+.1f} kWh", transform=ax2.transAxes,
             ha="right", va="top", fontsize=8, color="#55544f")

    above = mu > lam + 1e-9
    for t in h[above]:
        ax3.axvspan(t - 0.5, t + 0.5, color=con_color, alpha=0.12, lw=0)
    ax3.step(h, lam, where="mid", color="#1f1f1d", lw=1.5, label="lambda_t (dual of power balance)")
    ax3.axhline(mu, color=con_color, lw=2, label=f"mu (dual of E_min) = {mu:.3f}")
    ax3.set(xlabel="hour", ylabel="DKK/kWh")
    ax3.set_xticks(h[::2])
    ax3.legend(fontsize=8, loc="upper left", frameon=False, ncol=2,
               title="shaded: mu > lambda_t, load expected above reference", title_fontsize=8)
    return _finish(fig, save_to)


def plot_scenario_comparison(
    runs: dict[str, Results], metric: str = "objective", save_to: Path | str | None = None
) -> plt.Figure:
    """Bar chart of one metric across scenarios. ``metric`` is ``"objective"`` or the name of an
    hourly column whose daily sum is compared (e.g. ``"import"``, ``"export"``, ``"load"``)."""
    names = list(runs)
    if metric == "objective":
        values = [r.objective for r in runs.values()]
        ylabel = "daily cost [DKK]"
    else:
        values = [r.hourly[metric].sum() for r in runs.values()]
        ylabel = f"daily {metric} [kWh]"
    fig, ax = plt.subplots(figsize=(max(5, 1.2 * len(names)), 3.8))
    ax.bar(names, values, color="C0")
    ax.set(ylabel=ylabel, title=f"Scenario comparison - {metric}")
    ax.tick_params(axis="x", rotation=20)
    return _finish(fig, save_to)
