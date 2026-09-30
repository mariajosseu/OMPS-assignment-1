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


def plot_temporal_schedules(
    profiles: dict[str, np.ndarray],
    runs: dict[str, dict[str, Results]],
    save_to: Path | str | None = None,
) -> plt.Figure:
    """Plot matched price profiles and resulting schedules for all preference models."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex="col")
    for column, (profile_name, prices) in enumerate(profiles.items()):
        hours = np.arange(len(prices))
        axes[0, column].step(hours, prices, where="mid", color="black", label="energy price")
        axes[0, column].set(title=f"{profile_name.title()} prices", ylabel="DKK/kWh")
        axes[0, column].grid(alpha=0.2)

        for label, result in runs[profile_name].items():
            axes[1, column].plot(hours, result.hourly["load"], marker=".", label=label)
        reference = next(iter(runs[profile_name].values())).hourly["reference_load"]
        axes[1, column].step(hours, reference, where="mid", color="black", linestyle="--", label="reference")
        axes[1, column].set(xlabel="hour", ylabel="load (kWh/h)")
        axes[1, column].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    axes[1, 1].legend(fontsize=8)
    fig.suptitle("Price ordering and optimal load schedules")
    return _finish(fig, save_to)


def plot_temporal_metrics(summary, save_to: Path | str | None = None) -> plt.Figure:
    """Compare net objective-derived value and total absolute deviation by profile/model."""
    profiles = list(summary["price_profile"].drop_duplicates())
    models = list(summary["model"].drop_duplicates())
    x = np.arange(len(profiles))
    width = 0.8 / len(models)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    for axis, metric, ylabel, title in (
        (axes[0], "daily_net_surplus_DKK", "DKK", "Net surplus / objective-derived value"),
        (axes[1], "total_absolute_deviation_kWh", "kWh", "Total absolute deviation"),
    ):
        for index, model in enumerate(models):
            values = [
                summary.loc[
                    (summary["price_profile"] == profile) & (summary["model"] == model), metric
                ].iloc[0]
                for profile in profiles
            ]
            offset = (index - (len(models) - 1) / 2) * width
            axis.bar(x + offset, values, width, label=model)
        axis.set(xticks=x, xticklabels=[profile.title() for profile in profiles], ylabel=ylabel, title=title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(fontsize=8)
    return _finish(fig, save_to)
