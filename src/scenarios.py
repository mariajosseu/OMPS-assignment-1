"""Derive sensitivity scenarios from a base ``InputData`` without touching the JSON files.

Each helper returns a *new* ``InputData`` (the base data is never modified), so scenarios can
be chained::

    from src.data_loader import load_question
    from src.scenarios import scale_prices, set_tariffs, scale_pv

    base = load_question("Q1_caseA")
    high_spread = scale_prices(base, factor=2.0, keep_mean=True)      # same mean, doubled spread
    no_export_tariff = set_tariffs(high_spread, export_tariff=0.0)

Add your own helpers here (e.g. shifting the price peak towards the PV peak, changing the
minimum daily energy or the reference profile) and document them in the README.
"""
from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .data_loader import InputData, load_question
from .model import FlexibleConsumerModel
from .model import Results
from .plotting import plot_scenario_comparison


def scale_prices(data: InputData, factor: float, keep_mean: bool = False) -> InputData:
    """Multiply hourly prices by ``factor``. With ``keep_mean=True`` only the spread around the
    mean is scaled, which isolates the effect of price variability from the effect of price level."""
    p = data.energy_price
    new_p = p.mean() + factor * (p - p.mean()) if keep_mean else factor * p
    return replace(data, energy_price=np.clip(new_p, 0.0, None))   # prices stay non-negative


def shift_prices(data: InputData, delta: float) -> InputData:
    """Add ``delta`` DKK/kWh to every hourly price (changes the mean, keeps the spread)."""
    return replace(data, energy_price=np.clip(data.energy_price + delta, 0.0, None))


def roll_prices(data: InputData, hours: int) -> InputData:
    """Shift the price profile in time by ``hours`` (positive = later). Useful to study the
    temporal correlation between the price peak and the PV peak."""
    return replace(data, energy_price=np.roll(data.energy_price, hours))


def set_tariffs(data: InputData, import_tariff: float | None = None, export_tariff: float | None = None) -> InputData:
    """Override the import and/or export grid tariff (DKK/kWh)."""
    return replace(
        data,
        import_tariff=data.import_tariff if import_tariff is None else import_tariff,
        export_tariff=data.export_tariff if export_tariff is None else export_tariff,
    )


def scale_pv(data: InputData, factor: float) -> InputData:
    """Scale the available PV production (e.g. 0.5 for a cloudy day, 0.0 for no PV)."""
    profile = np.clip(data.pv_profile * factor, 0.0, 1.0)
    return replace(data, pv_profile=profile, pv_available=data.pv_max_kW * profile)


def set_load_preferences(
    data: InputData,
    load_max_kWh: float | None = None,
    load_min_kWh: float | None = None,
    min_daily_energy_kWh: float | None = None,
) -> InputData:
    """Override the hourly load bounds and/or the minimum daily energy requirement."""
    return replace(
        data,
        load_max_kWh=data.load_max_kWh if load_max_kWh is None else load_max_kWh,
        load_min_kWh=data.load_min_kWh if load_min_kWh is None else load_min_kWh,
        min_daily_energy_kWh=data.min_daily_energy_kWh if min_daily_energy_kWh is None else min_daily_energy_kWh,
    )


def sweep_linear_disutility(
    data: InputData,
    coefficients: list[float] | np.ndarray,
    plot_path: Path | str | None = None,
) -> pd.DataFrame:
    """Solve Q2 for each linear disutility coefficient and return a summary table.

    ``load_breakpoint_hours`` counts hours where the solution is at either supplied
    hourly load bound. The input data has no separate deviation-bound parameter.
    """
    if data.reference_load is None:
        raise ValueError("The linear disutility sweep requires a reference load profile.")

    rows: list[dict[str, float | int]] = []
    runs: dict[str, Results] = {}
    for coefficient in coefficients:
        coefficient = float(coefficient)
        if coefficient < 0:
            raise ValueError("Linear disutility coefficients must be non-negative.")
        scenario = replace(data, linear_disutility=coefficient)
        results = FlexibleConsumerModel(scenario).build().solve()
        runs[f"c_L={coefficient:g}"] = results
        metrics = results.daily_metrics
        rows.append({
            "c_L_DKK_per_kWh": coefficient,
            "daily_procurement_cost_DKK": float(metrics["daily_procurement_cost_DKK"]),
            "total_disutility_DKK": float(metrics["total_disutility_DKK"]),
            "daily_energy_consumed_kWh": float(metrics["daily_energy_consumed_kWh"]),
            "total_absolute_deviation_kWh": float(metrics["total_absolute_deviation_kWh"]),
            "load_breakpoint_hours": int(metrics["load_bound_binding_hours"]),
        })
    if plot_path is not None:
        plot_scenario_comparison(runs, save_to=plot_path)
    return pd.DataFrame(rows)


def sweep_quadratic_disutility(
    data: InputData,
    coefficients: list[float] | np.ndarray,
    plot_path: Path | str | None = None,
) -> pd.DataFrame:
    """Solve Q2 for each quadratic disutility coefficient and return a summary table."""
    if data.reference_load is None:
        raise ValueError("The quadratic disutility sweep requires a reference load profile.")

    rows: list[dict[str, float | int]] = []
    runs: dict[str, Results] = {}
    for coefficient in coefficients:
        coefficient = float(coefficient)
        if coefficient < 0:
            raise ValueError("Quadratic disutility coefficients must be non-negative.")
        scenario = replace(data, quadratic_disutility=coefficient)
        results = FlexibleConsumerModel(scenario).build().solve()
        runs[f"c_Q={coefficient:g}"] = results
        metrics = results.daily_metrics
        rows.append({
            "c_Q_DKK_per_kWh2": coefficient,
            "daily_procurement_cost_DKK": float(metrics["daily_procurement_cost_DKK"]),
            "total_disutility_DKK": float(metrics["total_disutility_DKK"]),
            "daily_energy_consumed_kWh": float(metrics["daily_energy_consumed_kWh"]),
            "total_absolute_deviation_kWh": float(metrics["total_absolute_deviation_kWh"]),
            "load_breakpoint_hours": int(metrics["load_bound_binding_hours"]),
        })
    if plot_path is not None:
        plot_scenario_comparison(runs, save_to=plot_path)
    return pd.DataFrame(rows)