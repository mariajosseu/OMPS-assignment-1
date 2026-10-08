# Demand-Side Flexibility in Active Distribution Grids

Hui-Ju Wang - s252794\
Maria Jose Uribe Pizarro - s252605\
Daniele Di Dio Roccazzella - s253213\
Gaia Franceschetti - s243134

Python models and experiments for DTU course 46750, Assignment 1. The project optimizes a flexible
consumer's 24-hour electricity schedule under hourly prices, PV availability, load preferences, and,
for the battery case, storage constraints. Gurobi solves the optimization models; Python exports the
hourly solutions, summary metrics, sensitivity tables, and figures.

## Cases

| Case | Model |
|---|---|
| `Q1_caseA` | Hourly consumption with PV cost below consumption utility |
| `Q1_caseB` | Hourly consumption with PV cost above consumption utility |
| `Q2_linear` | Flexible load with linear deviation cost from a reference profile |
| `Q2_quadratic` | Flexible load with quadratic deviation cost |
| `Q3` | Quadratic deviation cost with a minimum daily energy requirement |
| `Q3_battery` | Q3 consumer with battery storage |

## Project layout

| Path | Purpose |
|---|---|
| `main.py` | Command-line entry point for base cases and report experiments |
| `src/data_loader.py` | Loads a case and returns validated, unit-labelled `InputData` |
| `src/model.py` | Builds and solves the optimization model; extracts primal values, duals, and metrics |
| `src/scenarios.py` | Creates in-memory sensitivity cases and calculates experiment summaries |
| `src/plotting.py` | Creates Matplotlib figures for inputs, schedules, duals, and comparisons |
| `data/appliance_params.json` | Shared PV, flexible-load, and battery catalogue |
| `data/bus_params.json` | Shared hourly prices, tariffs, and grid limits |
| `data/params_<case>.json` | Case-specific selection of the consumer's appliances |
| `results/` | Generated CSV, TXT, LaTeX, and PNG outputs; ignored by Git |

Time series contain 24 hourly values. Power ratings use kW, hourly energy use uses kWh/h, daily
energy uses kWh, and prices and costs use DKK. The loader converts dimensionless profile ratios to
physical PV availability and reference load values. Input schemas and conversions are implemented
in `src/data_loader.py`.

## Setup

Use Python 3.11 or newer. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate              # macOS / Linux
# .venv\Scripts\activate              # Windows PowerShell
python -m pip install -r requirements.txt
```

Or create the Conda environment with `conda env create -f environment.yaml`, then activate it with
`conda activate 46750-a1`. A valid Gurobi license is required. The restricted license provided by
the Gurobi package is sufficient for these assignment-sized models; a DTU academic license can be
activated with `grbgetkey <your-key>` if needed.

## Reproduce results

Run commands from the repository root. A case command writes `<case>_hourly.csv`, `<case>_summary.txt`,
`inputs.png`, `schedule.png`, and `duals.png` to `results/<case>/`. Each experiment command below also
runs that case's base model before writing its additional outputs.

```bash
python main.py --question Q1_caseA
python main.py --question Q1_caseB
python main.py --question Q2_linear
python main.py --question Q2_quadratic
python main.py --question Q3
python main.py --question Q3_battery
```

| Experiment | Command | Additional outputs in `results/<case>/` |
|---|---|---|
| Linear disutility sweep | `python main.py --question Q2_linear --linear-sweep 0 0.5 1 1.43 2 3` | `linear_sweep.csv`, `linear_sweep.tex`, `linear_sweep_comparison.png`, `linear_load_comparison.png` |
| Q2(e) temporal-price comparison | `python main.py --question Q2_quadratic --q2e-experiment` | `Q2e_temporal_price_comparison.csv`, `.tex`, `Q2e_schedule.png`, `Q2e_temporal_schedules.png`, `Q2e_temporal_metrics.png`, tagged base result files |
| Minimum-energy sweep | `python main.py --question Q3 --emin-sweep 0 10 20 30 40 50 60` | `emin_sweep.csv`, `emin_sweep.tex`, `emin_sweep_comparison.png` |
| Quadratic-disutility sweep | `python main.py --question Q3 --quadratic-sweep 0.1 0.5 1` | `quadratic_sweep.csv`, `quadratic_sweep.tex`, `quadratic_sweep_comparison.png`, `quadratic_load_comparison.png` |
| Price-spread sweep | `python main.py --question Q3 --spread-sweep 0 0.5 1 1.5 2` | `spread_sweep.csv`, `spread_sweep.tex`, `spread_sweep_comparison.png` |
| Battery comparison | `python main.py --question Q3_battery --battery-comparison` | `battery_comparison.csv`, `.tex`, `battery_comparison.png`, `battery_load_comparison.png` |
| Battery value sweep | `python main.py --question Q3_battery --battery-sweep` | `battery_value_sweep.csv`, `.tex`, and parameter comparison PNGs |

The Q2(e) defaults are a temporal penalty of `0.1 DKK/kWh^2` and a three-hour window. Use
`--q2e-coefficient` and `--q2e-window-hours` to override them. `python main.py --help` lists all
available options. Add `--show` to open figures in a window.

Generated files under `results/` are excluded from version control. Re-run the corresponding command
to regenerate them from the tracked source and input data.

## Python API

The same loader, model, and plotting functions can be used in scripts or notebooks:

```python
from src.data_loader import load_question
from src.model import FlexibleConsumerModel
from src.plotting import plot_schedule

data = load_question("Q1_caseA")
results = FlexibleConsumerModel(data).build().solve()
print(results)
print(results.hourly)  # hourly variables and prices; energy rates are in kWh/h
plot_schedule(results, data)
```

`Results` also contains daily metrics and scalar constraint duals. Objective and cost values are in
DKK/day, daily energy is in kWh/day, and hourly primal values are in kWh/h. The precise metric names
and definitions are in `src/model.py`.

## Extending the project

Keep data loading, model construction and solving, scenario generation, and plotting in their existing
modules. Give every new function a short docstring describing its inputs, outputs, and units (or state
when an argument is dimensionless). Add a brief description here for every new module or data file.
Sensitivity cases should normally be derived in `src/scenarios.py`, leaving the source JSON unchanged.