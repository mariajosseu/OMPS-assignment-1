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

from src.data_loader import load_question, list_questions

from src.model import FlexibleConsumerModel, Results
from src.plotting import plot_duals, plot_inputs, plot_scenario_comparison, plot_schedule
from src.scenarios import sweep_linear_disutility, sweep_quadratic_disutility

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
        sweep = sweep_linear_disutility(load_question(args.question), args.linear_sweep)
        sweep.to_csv(out / "linear_sweep.csv", index=False)
        (out / "linear_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print("\nLinear disutility sweep:\n", sweep.to_string(index=False))
    if args.quadratic_sweep is not None:
        if args.question != "Q2_quadratic":
            parser.error("--quadratic-sweep requires --question Q2_quadratic")
        sweep = sweep_quadratic_disutility(load_question(args.question), args.quadratic_sweep)
        sweep.to_csv(out / "quadratic_sweep.csv", index=False)
        (out / "quadratic_sweep.tex").write_text(sweep.to_latex(index=False, float_format="%.3f"), encoding="utf-8")
        print("\nQuadratic disutility sweep:\n", sweep.to_string(index=False))
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
