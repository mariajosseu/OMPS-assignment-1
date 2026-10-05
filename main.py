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
from src.plotting import (
    plot_duals,
    plot_inputs,
    plot_schedule,
    plot_temporal_metrics,
    plot_temporal_schedules,
)
from src.scenarios import (
    compare_temporal_price_profiles,
    sweep_linear_disutility,
    sweep_quadratic_disutility,
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
        help="solve Q2_quadratic for the supplied c_Q values and save quadratic_sweep.csv",
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
    if args.q2e_experiment:
        if args.question != "Q2_quadratic":
            parser.error("--q2e-experiment requires --question Q2_quadratic")
        run_q2e_experiment(
            load_question(args.question),
            out,
            temporal_disutility=args.q2e_coefficient,
            temporal_window_hours=args.q2e_window_hours,
        )
    print(f"\nOutputs written to {out}")


if __name__ == "__main__":
    main()
