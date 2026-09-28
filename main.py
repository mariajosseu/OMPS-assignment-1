"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios
    python main.py --question Q3 --compare-unconstrained   # Question 3.(e)

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
    plot_duals, plot_inputs, plot_min_energy_comparison, plot_scenario_comparison, plot_schedule,
)
from src.scenarios import drop_min_energy, sweep_linear_disutility, sweep_quadratic_disutility

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
            "energy_above_Emin_kWh": m["daily_energy_consumed_kWh"] - data.min_daily_energy_kWh,
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
        help="solve Q2_quadratic for the supplied c_Q values and save quadratic_sweep.csv",
    )
    parser.add_argument(
        "--compare-unconstrained",
        action="store_true",
        help="Q3.(e): compare the E_min-constrained base case with the unconstrained 2.(c) consumer",
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
        if args.question != "Q2_quadratic":
            parser.error("--quadratic-sweep requires --question Q2_quadratic")
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
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
