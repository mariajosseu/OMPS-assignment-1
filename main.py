"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios
    python main.py --question Q3 --compare-unconstrained   # Question 3.(e)
    python main.py --question Q3 --sensitivity             # Question 3.(f)

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from src.data_loader import load_question, list_questions

from src.model import FlexibleConsumerModel, Results
from src.plotting import (
    plot_battery_comparison,
    plot_duals,
    plot_inputs,
    plot_load_scenarios,
    plot_min_energy_comparison,
    plot_scenario_comparison,
    plot_schedule,
    plot_sensitivity,
    plot_temporal_metrics,
    plot_temporal_schedules,
)
from src.scenarios import (
    compare_battery,
    sweep_battery_value,
    compare_temporal_price_profiles,
    drop_min_energy,
    scale_prices,
    sweep_linear_disutility,
    sweep_quadratic_disutility,
    set_load_preferences,
    set_quadratic_disutility,
)

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def run_base_case(question: str, out: Path, show: bool) -> Results | None:
    data = load_question(question)
    print(data.summary(), "\n")
    plot_inputs(data, save_to=out / "inputs.png")

    model = FlexibleConsumerModel(data).build()
    try:
        results = model.solve()
    except NotImplementedError as e:
        print(f"[skipped] {e}")
        return None

    print(results, "\n")
    results.save(out)
    plot_schedule(results, data, save_to=out / "schedule.png")
    plot_duals(results, data, save_to=out / "duals.png")
    if show:
        matplotlib.pyplot.show()
    return results


def run_min_energy_comparison(question: str, out: Path, tol: float = 1e-6) -> pd.DataFrame:
    """Question 3.(e): solve the base case with and without the minimum daily energy requirement.

    The unconstrained run is the Question 2.(c) consumer on the same data (``drop_min_energy``).
    Writes to ``out``:
      * ``q3e_comparison.png``   - loads vs reference, load shift, lambda_t vs mu;
      * ``q3e_daily.csv/.tex``   - daily metrics of both runs (DKK, kWh, hours);
      * ``q3e_hourly.csv``       - hourly loads, deviations, lambda_t and the KKT prediction.
    Sign convention (Gurobi, d objective / d rhs): mu >= 0 for the ">= E_min" constraint of the
    minimisation; lambda_t = dual of the power balance, the marginal cost of energy in hour t (DKK/kWh).
    Returns the daily-metrics table.
    """
    data = load_question(question)
    if data.min_daily_energy_kWh is None:
        raise ValueError(f"{question} has no minimum daily energy requirement to compare against.")
    runs = {
        "2c_unconstrained": FlexibleConsumerModel(drop_min_energy(data)).build().solve(),
        "Q3_with_Emin": FlexibleConsumerModel(data).build().solve(),
    }
    plot_min_energy_comparison(runs["Q3_with_Emin"], runs["2c_unconstrained"], data,
                               save_to=out / "q3e_comparison.png")

    ref = data.reference_load
    daily = {}
    for name, r in runs.items():
        dev = r.hourly["load"].to_numpy() - ref
        m = r.daily_metrics
        daily[name] = {
            "E_min_kWh": data.min_daily_energy_kWh if name == "Q3_with_Emin" else None,
            "daily_energy_consumed_kWh": m["daily_energy_consumed_kWh"],
            "energy_above_Emin_kWh": (
                m["daily_energy_consumed_kWh"] - data.min_daily_energy_kWh
                if name == "Q3_with_Emin" else None
            ),
            "mu_DKK_per_kWh": r.duals.get("daily_energy_min"),
            "daily_procurement_cost_DKK": m["daily_procurement_cost_DKK"],
            "total_disutility_DKK": m["total_disutility_DKK"],
            "objective_DKK": r.objective,
            "total_absolute_deviation_kWh": m["total_absolute_deviation_kWh"],
            "hours_above_reference": int((dev > tol).sum()),
            "energy_above_reference_kWh": float(dev[dev > tol].sum()),
            "hours_below_reference": int((dev < -tol).sum()),
            "hours_at_load_bound": m["load_bound_binding_hours"],
        }
    daily = pd.DataFrame(daily)
    daily.to_csv(out / "q3e_daily.csv", index_label="metric")
    (out / "q3e_daily.tex").write_text(daily.to_latex(float_format="%.3f", na_rep="-"), encoding="utf-8")

    con, unc = runs["Q3_with_Emin"].hourly, runs["2c_unconstrained"].hourly
    mu = runs["Q3_with_Emin"].duals["daily_energy_min"]
    # Interior KKT solution of Q3: 2 c_Q (l_t - ref_t) = mu - lambda_t, then clipped to the load bounds
    kkt_load = np.clip(ref + (mu - con["dual_balance"]) / (2 * data.quadratic_disutility),
                       data.load_min_kWh, data.load_max_kWh)
    hourly = pd.DataFrame({
        "price": con["price"],
        "reference_load": ref,
        "load_2c": unc["load"],
        "load_Q3": con["load"],
        "shift_Q3_minus_2c": con["load"] - unc["load"],
        "deviation_Q3": con["load"] - ref,
        "lambda_Q3": con["dual_balance"],
        "mu_minus_lambda": mu - con["dual_balance"],
        "load_Q3_from_KKT": kkt_load,
    })
    hourly.to_csv(out / "q3e_hourly.csv", index_label="hour")

    print("\nQuestion 3.(e) - daily metrics:\n", daily.to_string(float_format=lambda x: f"{x:.3f}"))
    print("\nHourly (Q3 vs 2.(c)):\n", hourly.round(3).to_string())
    print(f"\nmax |load_Q3 - KKT prediction| = {np.abs(hourly['load_Q3'] - kkt_load).max():.2e} kWh")
    return daily


def _min_energy_metrics(data, tol: float = 1e-6) -> tuple[dict, np.ndarray | None]:
    """Solve one scenario of the minimum-energy consumer and return (daily metrics, hourly load).

    Metrics: mu (DKK/kWh), energy (kWh), procurement cost, disutility and objective (DKK),
    hours above / below the reference and at L_max, energy above / below the reference (kWh),
    PV curtailed (kWh). An infeasible scenario returns ``{"status": "INFEASIBLE"}`` and no load.
    """
    try:
        r = FlexibleConsumerModel(data).build().solve()
    except RuntimeError:
        return {"status": "INFEASIBLE"}, None
    load = r.hourly["load"].to_numpy()
    dev = load - data.reference_load
    m = r.daily_metrics
    return {
        "status": r.status,
        "mu_DKK_per_kWh": r.duals.get("daily_energy_min", 0.0),
        "daily_energy_consumed_kWh": m["daily_energy_consumed_kWh"],
        "daily_procurement_cost_DKK": m["daily_procurement_cost_DKK"],
        "total_disutility_DKK": m["total_disutility_DKK"],
        "objective_DKK": r.objective,
        "hours_above_reference": int((dev > tol).sum()),
        "hours_below_reference": int((dev < -tol).sum()),
        "hours_at_max_load": int((load > data.load_max_kWh - tol).sum()),
        "energy_above_reference_kWh": float(dev[dev > tol].sum()),
        "energy_below_reference_kWh": float(-dev[dev < -tol].sum()),
        "pv_curtailed_kWh": float((r.hourly["pv_available"] - r.hourly["pv"]).sum()),
    }, load


def _run_sweep(name, param, param_label, values, make_scenario, profile_values, out, title,
               vlines=None, logx=False) -> pd.DataFrame:
    """Solve ``make_scenario(v)`` for every v in ``values``; save ``q3f_<name>.csv/.tex/.png``."""
    rows, profiles = [], {}
    for v in values:
        scenario = make_scenario(v)
        metrics, load = _min_energy_metrics(scenario)
        rows.append({param: v, **metrics})
        if load is not None and any(np.isclose(v, p) for p in profile_values):
            profiles[v] = load
    sweep = pd.DataFrame(rows)
    sweep.to_csv(out / f"q3f_{name}.csv", index=False)
    cols = [param, "mu_DKK_per_kWh", "daily_energy_consumed_kWh", "daily_procurement_cost_DKK",
            "total_disutility_DKK", "hours_above_reference", "hours_below_reference", "hours_at_max_load"]
    (out / f"q3f_{name}.tex").write_text(
        sweep[sweep["status"] == "OPTIMAL"][cols].to_latex(index=False, float_format="%.2f"), encoding="utf-8"
    )
    plot_sensitivity(sweep, param, param_label, profiles, make_scenario(values[0]).reference_load, title,
                     vlines=vlines, logx=logx, save_to=out / f"q3f_{name}.png")
    print(f"\nSensitivity to {param_label}:\n",
          sweep.drop(columns=["status"]).to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    return sweep


def run_min_energy_sensitivity(question: str, out: Path) -> dict[str, pd.DataFrame]:
    """Question 3.(f): three one-at-a-time sweeps around the base case, all else fixed.

    * ``Emin``   - E_min from 0 to 150 kWh (step 5, step 1 between 20 and 45 where the regimes change);
      above 24 * L_max = 144 kWh the problem is infeasible.
      Marked thresholds: E_unc (energy of the unconstrained 2.(c) consumer) and sum of the reference.
    * ``cQ``     - quadratic disutility coefficient c_Q from 0.05 to 20 DKK/kWh^2 (E_min = base).
    * ``spread`` - price spread factor around the unchanged daily mean (``scale_prices(keep_mean=True)``),
      from 0 (flat price) to 1.95, the largest factor that keeps every p_t above the export tariff.
    Writes ``q3f_<sweep>.csv``, ``.tex`` and ``.png`` to ``out``. Returns the three tables.
    """
    data = load_question(question)
    e_unc = FlexibleConsumerModel(drop_min_energy(data)).build().solve().hourly["load"].sum()
    ref_sum = data.reference_load.sum()
    return {
        "Emin": _run_sweep(
            "Emin", "E_min_kWh", "E_min [kWh]", list(np.union1d(np.arange(0.0, 151.0, 5.0), np.arange(20.0, 46.0, 1.0))),
            lambda v: set_load_preferences(data, min_daily_energy_kWh=v), [15.0, 28.0, 34.0, 40.0, 100.0], out,
            "Sensitivity to the minimum daily energy E_min",
            vlines={"E_unc": e_unc, "sum ref": ref_sum, "24 L_max": 24 * data.load_max_kWh},
        ),
        "cQ": _run_sweep(
            "cQ", "c_Q", "c_Q [DKK/kWh2]", [0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0],
            lambda v: set_quadratic_disutility(data, v), [0.1, 1.0, 10.0], out,
            f"Sensitivity to the disutility coefficient c_Q (E_min = {data.min_daily_energy_kWh:g} kWh)",
            logx=True,
        ),
        "spread": _run_sweep(
            "spread", "spread_factor", "price spread factor", [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 1.95],
            lambda v: scale_prices(data, v, keep_mean=True), [0.0, 1.0, 1.95], out,
            f"Sensitivity to the price spread, same mean (E_min = {data.min_daily_energy_kWh:g} kWh)",
        ),
    }
def run_q2e_experiment(
    data,
    out: Path,
    temporal_disutility: float,
    temporal_window_hours: int,
) -> None:
    """Solve the Q2(e) base case and save the matched temporal-price experiment."""
    result = FlexibleConsumerModel(
        data,
        temporal_disutility=temporal_disutility,
        temporal_window_hours=temporal_window_hours,
    ).build().solve()
    result.save(out, tag="Q2e_base")
    plot_schedule(result, data, save_to=out / "Q2e_schedule.png")

    summary, profiles, runs = compare_temporal_price_profiles(
        data,
        temporal_disutility=temporal_disutility,
        temporal_window_hours=temporal_window_hours,
    )
    summary.to_csv(out / "Q2e_temporal_price_comparison.csv", index=False)
    (out / "Q2e_temporal_price_comparison.tex").write_text(
        summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8"
    )
    plot_temporal_schedules(profiles, runs, save_to=out / "Q2e_temporal_schedules.png")
    plot_temporal_metrics(summary, save_to=out / "Q2e_temporal_metrics.png")
    print("\nQ2(e) temporal price comparison:\n", summary.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--question", default="Q1_caseA", choices=list_questions(), help="data case to use")
    parser.add_argument("--scenarios", action="store_true", help="also run the example sensitivity scenarios")
    parser.add_argument(
        "--linear-sweep",
        nargs="+",
        type=float,
        metavar="c_L",
        help="solve Q2_linear for the supplied c_L values and save linear_sweep.csv",
    )
    parser.add_argument(
        "--quadratic-sweep",
        nargs="+",
        type=float,
        metavar="c_Q",
        help="solve Q2_quadratic or Q3 (with the E_min constraint) for the supplied c_Q values and save quadratic_sweep.csv",
    )
    parser.add_argument(
        "--q2e-experiment",
        action="store_true",
        help="run the Q2(e) base case and matched alternating/block price experiment",
    )
    parser.add_argument(
        "--q2e-coefficient",
        type=float,
        default=0.1,
        metavar="KAPPA",
        help="Q2(e) rolling-deviation penalty in DKK/kWh^2 (default: 0.1)",
    )
    parser.add_argument(
        "--q2e-window-hours",
        type=int,
        default=3,
        metavar="HOURS",
        help="Q2(e) rolling window length (default: 3)",
    )
    parser.add_argument(
        "--battery-comparison",
        action="store_true",
        help="Q3(g): compare Q3_battery with the same consumer without its battery",
    )
    parser.add_argument(
        "--battery-sweep",
        action="store_true",
        help="Q3(g.v): sweep environment and battery parameters and report the battery value",
    )
    parser.add_argument(
        "--emin-sweep",
        nargs="+",
        type=float,
        metavar="E_MIN",
        help="solve Q3 for the supplied minimum daily energy values and save emin_sweep.csv",
    )
    parser.add_argument(
        "--compare-unconstrained",
        action="store_true",
        help="Q3.(e): compare the E_min-constrained base case with the unconstrained 2.(c) consumer",
    )
    parser.add_argument(
        "--sensitivity",
        action="store_true",
        help="Q3.(f): one-at-a-time sweeps of E_min, c_Q and the price spread",
    )
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.linear_sweep is not None:
        if args.question != "Q2_linear":
            parser.error("--linear-sweep requires --question Q2_linear")
        sweep = sweep_linear_disutility(
            load_question(args.question),
            args.linear_sweep,
            plot_path=out / "linear_sweep_comparison.png",
        )
        sweep.to_csv(out / "linear_sweep.csv", index=False)
        (out / "linear_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print("\nLinear disutility sweep:\n", sweep.to_string(index=False))
    if args.quadratic_sweep is not None:
        if args.question not in ("Q2_quadratic", "Q3"):
            parser.error("--quadratic-sweep requires --question Q2_quadratic or Q3")
        sweep = sweep_quadratic_disutility(
            load_question(args.question),
            args.quadratic_sweep,
            plot_path=out / "quadratic_sweep_comparison.png",
        )
        sweep.to_csv(out / "quadratic_sweep.csv", index=False)
        (out / "quadratic_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print("\nQuadratic disutility sweep:\n", sweep.to_string(index=False))
    if args.compare_unconstrained:
        if args.question != "Q3":
            parser.error("--compare-unconstrained requires --question Q3")
        run_min_energy_comparison(args.question, out)
    if args.sensitivity:
        if args.question != "Q3":
            parser.error("--sensitivity requires --question Q3")
        run_min_energy_sensitivity(args.question, out)
    if args.q2e_experiment:
        if args.question != "Q2_quadratic":
            parser.error("--q2e-experiment requires --question Q2_quadratic")
        run_q2e_experiment(
            load_question(args.question),
            out,
            temporal_disutility=args.q2e_coefficient,
            temporal_window_hours=args.q2e_window_hours,
        )
    if args.battery_comparison:
        if args.question != "Q3_battery":
            parser.error("--battery-comparison requires --question Q3_battery")
        data = load_question(args.question)
        summary, no_bat, bat = compare_battery(data)
        summary.to_csv(out / "battery_comparison.csv", index=False)
        (out / "battery_comparison.tex").write_text(summary.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        plot_battery_comparison(no_bat, bat, data, save_to=out / "battery_comparison.png")
        plot_load_scenarios({"no battery": no_bat, "with battery": bat},
                            save_to=out / "battery_load_comparison.png",
                            title="Actual load with and without the battery")
        print("\nBattery comparison:\n", summary.T.to_string(header=False))
        print(f"Value of the battery: {summary.net_utility_DKK[1] - summary.net_utility_DKK[0]:.3f} DKK/day")
    if args.battery_sweep:
        if args.question != "Q3_battery":
            parser.error("--battery-sweep requires --question Q3_battery")
        sweep = sweep_battery_value(load_question(args.question))
        sweep.to_csv(out / "battery_value_sweep.csv", index=False)
        (out / "battery_value_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        for parameter, group in sweep.groupby("parameter", sort=False):
            plot_scenario_comparison(
                values={f"{v:g}": x for v, x in zip(group["value"], group["battery_value_DKK"])},
                metric=f"battery value vs {parameter}",
                ylabel="battery value [DKK/day]",
                save_to=out / f"battery_value_{parameter}.png",
            )
        print("\nBattery value sweeps:\n", sweep.to_string(index=False))
    if args.emin_sweep is not None:
        if args.question != "Q3":
            parser.error("--emin-sweep requires --question Q3")
        data = load_question(args.question)
        rows = []
        runs = {}
        for minimum_energy in args.emin_sweep:
            if minimum_energy < 0:
                parser.error("--emin-sweep values must be non-negative")
            scenario = set_load_preferences(data, min_daily_energy_kWh=minimum_energy)
            results = FlexibleConsumerModel(scenario).build().solve()
            runs[f"E_min={minimum_energy:g}"] = results
            metrics = results.daily_metrics
            rows.append({
                "E_min_kWh": minimum_energy,
                "daily_energy_consumed_kWh": float(metrics["daily_energy_consumed_kWh"]),
                "objective_DKK": float(results.objective),
                "daily_procurement_cost_DKK": float(metrics["daily_procurement_cost_DKK"]),
                "total_disutility_DKK": float(metrics["total_disutility_DKK"]),
                "daily_energy_min_dual_DKK_per_kWh": float(results.duals["daily_energy_min"]),
            })
        sweep = pd.DataFrame(rows)
        sweep.to_csv(out / "emin_sweep.csv", index=False)
        (out / "emin_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        plot_scenario_comparison(runs, save_to=out / "emin_sweep_comparison.png")
        print("\nMinimum daily energy sweep:\n", sweep.to_string(index=False))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
