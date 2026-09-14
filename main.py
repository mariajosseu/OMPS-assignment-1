"""Entry point: load one question's data, build and solve the model, save results and figures.

    python main.py                          # base case of Q1_caseA
    python main.py --question Q2_linear     # another case
    python main.py --scenarios              # also run the example sensitivity scenarios

Results (CSV, TXT, PNG) are written to ``results/<question>/``. Extend ``run_scenarios``
with your own scenarios, or add a new function per question, as your analysis grows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import copy

from src.data_loader import load_question, list_questions

#import modelQ2 file instead of model file
from src.modelQ2 import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import scale_prices, scale_pv, set_tariffs, sweep_linear_disutility

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


#Run scenario analysis for Q2_linear and Q2_quadratic
def run_scenarios(question: str, out: Path) -> dict[str, Results]:
    """Example sensitivity analysis. Replace with the scenarios you design in Question 1.g."""
    base = load_question(question)
    

    if question == "Q2_linear":
        def set_cL(val):
            d = copy.deepcopy(base)
            d.linear_disutility = val
            return d
            
        scenarios = {
            "cL_0": set_cL(0),
            "base": set_cL(1.43),      
            "cL_3": set_cL(3),
        }
        
    elif question == "Q2_quadratic":
        def set_cQ(val):
            d = copy.deepcopy(base)
            d.quadratic_disutility = val
            return d
            
        scenarios = {
            "cQ_0.05": set_cQ(0.05),
            "cQ_0.30": set_cQ(0.30),
            "base": base,         
            "cQ_5.00": set_cQ(5.00),
            "cQ_50.0": set_cQ(50.0),
        }
        
    else:
        scenarios = {
            "base": base,
            "flat_prices": scale_prices(base, factor=0.0, keep_mean=True),
            "double_spread": scale_prices(base, factor=2.0, keep_mean=True),
            "no_tariffs": set_tariffs(base, import_tariff=0.0, export_tariff=0.0),
            "no_pv": scale_pv(base, factor=0.0),
        }

    runs: dict[str, Results] = {}
    for name, data in scenarios.items():
        results = FlexibleConsumerModel(data).build().solve()
        results.save(out, tag=name)
        runs[name] = results

        load_list = results.hourly['load'].tolist()
        ref_list = data.reference_load
        

        energy_consumed = sum(load_list)
        

        abs_deviation = sum(abs(l - r) for l, r in zip(load_list, ref_list))
        
        # Total disutility
        if data.question == "Q2_linear":
            total_disutility = data.linear_disutility * abs_deviation
        elif data.question == "Q2_quadratic":
            total_disutility = data.quadratic_disutility * sum((l - r)**2 for l, r in zip(load_list, ref_list))
        else:
            total_disutility = 0.0
            
        procurement_cost = results.objective - total_disutility
        
        print(f"[{name}]")
        print(f"  - Objective (Total Cost) : {results.objective:6.2f} DKK")
        print(f"  - Procurement Cost       : {procurement_cost:6.2f} DKK")
        print(f"  - Total Disutility       : {total_disutility:6.2f} DKK")
        print(f"  - Energy Consumed        : {energy_consumed:6.1f} kWh")
        print(f"  - Absolute Deviation     : {abs_deviation:6.2f} kWh\n")

    plot_scenario_comparison(runs, "objective", save_to=out / "scenarios_cost.png")
    return runs


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
    parser.add_argument("--show", action="store_true", help="open the figures in a window")
    args = parser.parse_args()

    out = RESULTS_DIR / args.question
    out.mkdir(parents=True, exist_ok=True)
    if not args.show:
        matplotlib.use("Agg")

    base = run_base_case(args.question, out, args.show)
    if args.scenarios and base is not None:
        run_scenarios(args.question, out)
    if args.linear_sweep is not None:
        if args.question != "Q2_linear":
            parser.error("--linear-sweep requires --question Q2_linear")
        sweep = sweep_linear_disutility(load_question(args.question), args.linear_sweep)
        sweep.to_csv(out / "linear_sweep.csv", index=False)
        (out / "linear_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print("\nLinear disutility sweep:\n", sweep.to_string(index=False))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
