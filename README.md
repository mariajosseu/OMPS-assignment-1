# 46750 - Assignment 1: Demand-Side Flexibility in Active Distribution Grids

Starter repository for **Group Assignment 1** of *46750 - Optimization in Modern Power Systems* (DTU).
It contains the input data for every question (in `data/`), a small and working Python code base to
build on, and the instructions below. The structure is a suggestion: adapt it to your needs, but keep
it documented (update this README) so that your code stays reproducible and easy to grade.

**Getting your own copy.** This is a public template: on the repository page, click
**Use this template -> Create a new repository** to create your group's own repository under one
member's GitHub account (recommended - the group can then work with git), or **Code -> Download ZIP**
to work without GitHub. Using the Python starter code is recommended but not required; the input data
in `data/` must be used as provided.

**Submitting your code.** Choose one of the two, and say in your report which one you chose:
attach your complete project code as a single `.zip` file to your submission in DTU Learn, or give
the link to your group's GitHub repository on the front page of your report. A linked repository must
be accessible to the graders (public, or private with the teaching team invited) and must not be
modified after the deadline - the last commit before the deadline is what is graded.

## 1. Setup

### 1.1 Python environment

Use one of the two options (both install the same packages).

**Option A - pip and venv**
```bash
python -m venv venv
source venv/bin/activate            # macOS / Linux
# venv\Scripts\activate.bat         # Windows cmd
# venv\Scripts\Activate.ps1         # Windows PowerShell
pip install -r requirements.txt
```

**Option B - conda**
```bash
conda env create -f environment.yaml
conda activate 46750-a1
```

### 1.2 Gurobi licence

The code uses [Gurobi](https://www.gurobi.com) through the `gurobipy` package. The package ships with a
restricted licence that is large enough for this assignment (models up to 2000 variables/constraints), so
`python main.py` works out of the box. For unrestricted use, request a free
[academic licence](https://www.gurobi.com/academia/academic-program-and-licenses/) with your DTU e-mail
and activate it with `grbgetkey <your-key>` (on the DTU network or VPN).

### 1.3 Check that everything works
```bash
python main.py
```
This loads the `Q1_caseA` case from `data/`, prints a summary of the input data and saves the
input figure to `results/Q1_caseA/`. Until you complete the model (see Section 3) it prints
`[skipped] The model has no constraints ...` and stops there - that is expected.

## 2. Repository structure

```
main.py                  Entry point: load data -> build model -> solve -> save results and figures
src/
  data_loader.py         load_question("Q1_caseA") -> InputData (all parameters, with units)
  model.py               FlexibleConsumerModel: build() / solve() -> Results (primal + dual values)
  scenarios.py           Helpers that derive sensitivity scenarios from a base InputData
  plotting.py            Figures for inputs, optimal schedule, duals and scenario comparisons
data/
  appliance_params.json  SHARED catalogue: every PV system, flexible load and battery - see Section 4
  bus_params.json        SHARED grid connection: prices (both days) and tariffs
  params_Q1_caseA.json   One small file per case: composes a consumer from the catalogue
  params_Q1_caseB.json
  params_Q2_linear.json
  params_Q2_quadratic.json
  params_Q3.json
  params_Q3_battery.json
results/                 Written by main.py (git-ignored)
requirements.txt, environment.yaml, LICENSE, .gitignore
```

The four modules mirror the workflow you are asked to implement and document: *data loading*,
*model building*, *solving and extracting results*, *plotting*. Keep them separate as your code grows -
for example one model class per question in `src/model.py` (or one file per question), and one function per
experiment in `main.py`.

## 3. How to use the code

**Run everything for one case**
```bash
python main.py --question Q1_caseA              # base case
python main.py --question Q1_caseA --scenarios  # + example sensitivity scenarios
python main.py --show                           # open the figures in a window
```

**Use it from a notebook or your own script** (run from the repository root):
```python
from src.data_loader import load_question
from src.model import FlexibleConsumerModel
from src.plotting import plot_schedule, plot_duals
from src.scenarios import scale_prices

data = load_question("Q1_caseA")
print(data.summary())

results = FlexibleConsumerModel(data).build().solve()
print(results)                    # objective, daily totals, scalar duals
results.hourly                    # DataFrame: one row per hour with variables, prices and hourly duals
plot_schedule(results, data)

high_spread = scale_prices(data, factor=2.0, keep_mean=True)
results_hs = FlexibleConsumerModel(high_spread).build().solve()
```

**What you need to implement.** `FlexibleConsumerModel.build()` in `src/model.py` is entirely `TODO`:
identify and declare the decision variables of your formulation, then add the objective and the
constraints - the gurobipy pattern for each step is shown in comments (including the `vtype=` to use
if you ever declare binary variables). Complete it with your formulation from Question 1, then extend
or subclass it for the following questions.
Everything downstream (solving, extraction of primal and dual values, saving, plotting) already works.

**Conventions that make the primal and dual values come out for free**

* Store every variable family in `self.var[<name>]`. `solve()` returns the hourly values of each family
  as a column of `results.hourly`. gurobipy indexes names automatically - `m.addVars(T, name="p_import")`
  creates `p_import[0]` ... `p_import[23]`, and the same holds for `m.addConstrs(..., name=...)` - so
  per-hour names come for free. Families named `import`, `export`, `load`, `pv` make the standard plots
  of `src/plotting.py` work out of the box. With binary or integer variables (`vtype=GRB.BINARY`) the
  model becomes a MILP and dual values are no longer defined - `solve()` then skips them.

* Store every constraint family in `self.con[<name>]`. `solve()` returns the dual value (Gurobi attribute
  `Pi`) of every hourly constraint as a column `dual_<name>` of `results.hourly`, and of every single
  constraint in `results.duals`.
* Write the bounds you want a dual for as explicit constraints (`m.addConstr(...)`), not as variable bounds
  (`lb=`, `ub=`). Gurobi reports the sensitivity of a variable bound in the reduced cost (`RC`), not in `Pi`.
* No quadratic *constraint* is needed in this assignment: the quadratic disutility sits in the objective, and
  the duals of the (linear) constraints of a QP need nothing special. If you ever add one (`m.addQConstr(...)`),
  its dual is in the attribute `QCPi` and Gurobi only computes it when the parameter `QCPDual` is 1 -
  `model.py` sets it and reads the right attribute for you.
  Gurobi's sign convention is d(objective)/d(right-hand side): for a "<=" constraint in a minimization the value
  is non-positive; state the convention you use when you report multipliers.

## 4. Input data

All input data lives in `data/`: two **shared** files plus one small **case** file per
question. The shared files form a catalogue of every appliance used anywhere in the assignment;
each `params_<case>.json` composes the consumer of one case by picking appliances from the
catalogue by ID. All time series have 24 hourly values (hour 0 to 23).

**Units.** Every field name carries its unit. Fields ending in `_kW` are power ratings (kW);
fields ending in `_kWh_per_hour` (or `_KWh_per_hour` - same unit despite the capital K) are hourly
energy bounds (kWh consumed within one hour); fields ending in `_kWh` are energy amounts;
`_DKK_per_kWh` are prices/costs and `_DKK_per_kWh2` is the quadratic disutility coefficient.
**Capacity factors and ratios** (`hourly_profile_ratio`, `reference_load_capacity_factor`,
`initial_state_of_charge_ratio`, and the `_next_day` profiles) are **dimensionless fractions in
[0, 1] of the corresponding maximum** - multiply by it to get physical values. With 1-hour periods,
a power rating in kW and an energy amount in kWh/h are numerically interchangeable; the loader
does every conversion for you and exposes physical quantities:
`pv_available = max_power_kW x hourly_profile_ratio` (kWh/h),
`reference_load = max_load_KWh_per_hour x reference_load_capacity_factor` (kWh/h), and
`battery_initial_soc_kWh = storage_capacity_kWh x initial_state_of_charge_ratio` (kWh).

| Question in the assignment | Case | Appliances (from the catalogue) |
|---|---|---|
| Question 1 - hourly consumption decision (utility u, PV cost c_PV), case A: c_PV < u | `Q1_caseA` | `PV_01_A` + `FL_01` (a load with `consumption_utility_DKK_per_kWh`; no reference profile, no daily requirement) |
| Question 1, case B: a more expensive PV, c_PV > u | `Q1_caseB` | `PV_01_B` + `FL_01` (`PV_01_B` differs from `PV_01_A` only in `marginal_cost_DKK_per_kWh`) |
| Question 2 - linear disutility of deviating from a reference profile | `Q2_linear` | `PV_01_A` + `FL_02_L` (reference profile, `linear_disutility_DKK_per_kWh`; same hourly bounds as Question 1; no utility) |
| Question 2 - quadratic disutility | `Q2_quadratic` | `PV_01_A` + `FL_02_Q` (as `FL_02_L` with `quadratic_disutility_DKK_per_kWh2` instead) |
| Question 3 - minimum daily energy requirement | `Q3` | `PV_01_A` + `FL_03` (as `FL_02_Q` plus `min_total_energy_per_day_kWh`, the E_min of the daily requirement) |
| Question 3.(g) - the same consumer with a small battery | `Q3_battery` | `PV_01_A` + `FL_03` + `BAT_01` |

No data is shipped for the bonus question 2.(e) on purpose: choosing your own cost parameters (and
justifying them) is part of that question - start from the `Q2_quadratic` case and override in code.

`params_<case>.json` - the case file
: `consumers[]`: `consumer_id`, `connection_bus` (a `bus_id` of `bus_params.json`), `list_appliances`
  (IDs of the consumer's DERs, loads and storages in `appliance_params.json`). `hyperparameters` is
  unused in Assignment 1 (`null`).

`appliance_params.json` - the shared catalogue, one list per appliance type (`null` if none)
: **DERs**: `DER_id`, `DER_type` (`"PV"`), `max_power_kW` (peak power, kW), `marginal_cost_DKK_per_kWh` (cost of every kWh produced), `hourly_profile_ratio` (dimensionless fraction of `max_power_kW` available each hour; available energy in kWh/h = `max_power_kW` x ratio), `hourly_profile_ratio_next_day` (see *Next-day forecasts* below)
: **loads**: `load_id`, `load_type`, `max_load_KWh_per_hour` and `min_load_KWh_per_hour` (hourly consumption bounds, kWh/h), `consumption_utility_DKK_per_kWh` (value of every kWh consumed, Question 1), `linear_disutility_DKK_per_kWh` and `quadratic_disutility_DKK_per_kWh2` (Questions 2-3), `min_total_energy_per_day_kWh` (the E_min of Question 3), `reference_load_capacity_factor` (dimensionless fraction of `max_load_KWh_per_hour`; preferred consumption in kWh/h = max load x factor), `reference_hourly_profile_ratio_next_day` (see *Next-day forecasts*). Fields a load does not use are `null`. The hourly bounds are the same for every load: deviations from the reference are limited only by them.
: **storages**: `storage_id`, `storage_type` (`"battery"`), `storage_capacity_kWh`, `max_charging_power_kW`, `max_discharging_power_kW`, `charging_efficiency` and `discharging_efficiency` (fraction of the energy transferred that survives each conversion - a full round trip keeps η_ch·η_dis), `initial_state_of_charge_ratio` (fraction of `storage_capacity_kWh` in the battery at the start of the day). **No final state of charge is given on purpose**: how to treat the battery's state of charge at the end of the day (constraint or objective) is your modeling choice in Question 3.(g).
: **heat_pumps**: not used in Assignment 1 (`null`)

`bus_params.json` - grid and market conditions at the connection bus
: `bus_id`, `import_tariff_DKK_per_kWh`, `export_tariff_DKK_per_kWh`, `energy_price_DKK_per_kWh` (24 values), `energy_price_next_day_DKK_per_kWh` (see *Next-day forecasts*), `max_import_kW`, `max_export_kW` (`null` = no grid limit in Assignment 1)

**Next-day forecasts.** Three fields carry the forecast for the *following* day:
`energy_price_next_day_DKK_per_kWh` (in `bus_params.json`), `hourly_profile_ratio_next_day` (in the
DER entries) and `reference_hourly_profile_ratio_next_day` (in the load entries). They are **not
needed for the base models**: they are provided in case you want to explore the end-of-horizon
choice of Question 3.(g) with a look-ahead (for instance a rolling-horizon or a two-day experiment).
`load_question()` exposes them as `energy_price_next_day`, `pv_available_next_day` and
`reference_load_next_day`.

`load_question()` in `src/data_loader.py` shows exactly how each attribute is derived from these files.

**Modifying or adding data.** For sensitivity analyses, prefer deriving scenarios in code
(`src/scenarios.py`) over editing the JSON files - it keeps the base case intact and the experiment
reproducible. If you do add appliances or case files, follow the same structure (a new entry in the
catalogue, a new `params_<case>.json`) and document them here.

## 5. What is expected of your code

Your code is part of the submission (as a repository link or a `.zip`, see the top of this README).
The graders should be able to open it, follow this README, and reproduce every number and figure in
your report. In practice:

* keep the separation between data loading, model building, solving and plotting;
* document every function you add (a short docstring stating inputs, outputs and units is enough), and
  describe any new module or data file in this README;
* make each experiment of the report runnable with a single command (e.g. `python main.py --question ...`
  or one notebook cell), and save its outputs under `results/`;
* commit regularly and with meaningful messages - the git history is also a record of everyone's contribution.
