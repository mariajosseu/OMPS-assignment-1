"""Optimisation model of a single flexible consumer, implemented with gurobipy.

The class below separates the three steps you will repeat for every question:

    model = FlexibleConsumerModel(data)   # 1. hand over the input data
    model.build()                         # 2. declare variables, objective, constraints
    results = model.solve()               # 3. optimise and collect primal AND dual values

``build()`` is the only method you need to complete for Question 1; the other questions
are variations of it (a different objective, an extra constraint). Copy this file or
subclass ``FlexibleConsumerModel`` and override ``build()`` to keep one model per question.

Two conventions make the dual variables easy to read out afterwards:

* Every constraint family is stored in ``self.con`` under a descriptive name, e.g.
  ``self.con["balance"] = self.m.addConstrs(...)``. ``solve()`` then returns the dual value
  (shadow price, Gurobi attribute ``Pi``) of every constraint in ``self.con`` automatically.
* Bounds that you want a dual for must be written as explicit constraints (``addConstr``),
  not as variable bounds (``lb=``/``ub=``). Gurobi reports the sensitivity of a variable
  bound in the reduced cost (``RC``), not in ``Pi``.
* Duals of quadratic constraints (``m.addQConstr``) are read from ``QCPi`` and require ``QCPDual = 1`` (set below).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB

from .data_loader import InputData


@dataclass
class Results:
    """Primal and dual solution of one model run."""

    question: str
    status: str
    objective: float
    hourly: pd.DataFrame                   # one row per hour: variables, prices, hourly duals
    duals: dict[str, float] = field(default_factory=dict)   # duals of non-hourly constraints
    meta: dict = field(default_factory=dict)                 # anything else worth keeping (scenario name, ...)
    daily_metrics: dict[str, object] = field(default_factory=dict)

    def save(self, folder: Path | str, tag: str = "") -> None:
        """Write ``hourly`` to CSV and the scalar values to a small text file."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"{self.question}{'_' + tag if tag else ''}"
        self.hourly.to_csv(folder / f"{stem}_hourly.csv", index_label="hour")
        with open(folder / f"{stem}_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"status    : {self.status}\nobjective : {self.objective:.4f} DKK\n")
            for k, v in self.daily_metrics.items():
                f.write(f"{k} : {v}\n")
            for k, v in self.duals.items():
                f.write(f"dual[{k}] : {v:.4f}\n")

    def __str__(self) -> str:
        cols = [c for c in self.hourly.columns if not c.startswith("dual_")]
        return (
            f"status: {self.status} | objective: {self.objective:.2f} DKK\n"
            f"daily totals (kWh): " + ", ".join(f"{c}={self.hourly[c].sum():.1f}" for c in cols if c in ("import", "export", "load", "pv"))
            + (f"\nmetrics: {self.daily_metrics}" if self.daily_metrics else "")
            + (f"\nduals: {self.duals}" if self.duals else "")
        )


def _dual(c) -> float:
    """Dual value of a linear (``Pi``) or quadratic (``QCPi``, requires QCPDual=1) constraint."""
    return c.QCPi if isinstance(c, gp.QConstr) else c.Pi


class FlexibleConsumerModel:
    """Consumption problem of one consumer over a 24-hour horizon (Question 1); extend it for Questions 2 and 3."""

    def __init__(self, data: InputData, name: str = "flexible_consumer", verbose: bool = False):
        self.data = data
        self.T = range(data.n_hours)
        self.m = gp.Model(name)
        self.m.Params.OutputFlag = 1 if verbose else 0
        self.m.Params.QCPDual = 1          # only relevant if you add a quadratic constraint (none is needed in Assignment 1)
        self.var: dict[str, gp.tupledict | gp.Var] = {}   # decision variables by name
        self.con: dict[str, gp.tupledict | gp.Constr] = {}  # constraints by name (duals read from here)

    # ------------------------------------------------------------------ 2. build
    def build(self) -> "FlexibleConsumerModel":
        """Build the hourly consumer LP for Question 1 or Question 2(b)."""
        d, m, T = self.data, self.m, self.T

        # --- Decision variables --------------------------------------------------------
        self.var["load"] = m.addVars(T, lb=-GRB.INFINITY, name="load")
        self.var["pv"] = m.addVars(T, lb=-GRB.INFINITY, name="pv")
        self.var["import"] = m.addVars(T, lb=-GRB.INFINITY, name="import")
        self.var["export"] = m.addVars(T, lb=-GRB.INFINITY, name="export")

        is_q2_linear = d.reference_load is not None and d.linear_disutility is not None
        is_q2_quadratic = d.reference_load is not None and d.quadratic_disutility is not None
        if d.reference_load is not None and not (is_q2_linear or is_q2_quadratic):
            raise NotImplementedError("Question 2 requires linear or quadratic disutility data.")
        if is_q2_linear:
            self.var["deviation"] = m.addVars(T, lb=-GRB.INFINITY, name="deviation")

        # --- Objective ---------------------------------------------------------------
        load = self.var["load"]
        pv = self.var["pv"]
        imported = self.var["import"]
        exported = self.var["export"]
        hourly_cost = ((d.energy_price[t] + d.import_tariff) * imported[t]
                       - (d.energy_price[t] - d.export_tariff) * exported[t]
                       + d.pv_marginal_cost * pv[t]
                       for t in T)
        if is_q2_linear:
            objective = gp.quicksum(hourly_cost) + d.linear_disutility * gp.quicksum(
                self.var["deviation"][t] for t in T
            )
        elif is_q2_quadratic:
            objective = gp.quicksum(
                hourly_cost
            ) + d.quadratic_disutility * gp.quicksum(
                (load[t] - d.reference_load[t]) ** 2 for t in T
            )
        elif d.consumption_utility is not None:
            objective = gp.quicksum(hourly_cost) - d.consumption_utility * gp.quicksum(load[t] for t in T)
        else:
            raise ValueError("Input data must define either consumption utility or linear disutility.")
        m.setObjective(objective, GRB.MINIMIZE)

        # --- Constraints -------------------------------------------------------------
        self.con["balance"] = m.addConstrs(
            (pv[t] + imported[t] == load[t] + exported[t] for t in T), name="balance"
        )
        self.con["load_min"] = m.addConstrs(
            (load[t] >= d.load_min_kWh for t in T), name="load_min"
        )
        self.con["load_max"] = m.addConstrs(
            (load[t] <= d.load_max_kWh for t in T), name="load_max"
        )

        if d.min_daily_energy_kWh is not None:
            self.con["daily_energy_min"] = m.addConstr(
                gp.quicksum(load[t] for t in T) >= d.min_daily_energy_kWh, name="daily_energy_min"
            )
        self.con["pv_min"] = m.addConstrs((pv[t] >= 0 for t in T), name="pv_min")
        self.con["pv_max"] = m.addConstrs(
            (pv[t] <= d.pv_available[t] for t in T), name="pv_max"
        )
        self.con["import_min"] = m.addConstrs((imported[t] >= 0 for t in T), name="import_min")
        self.con["export_min"] = m.addConstrs((exported[t] >= 0 for t in T), name="export_min")

        if d.max_import_kW is not None:
            self.con["import_max"] = m.addConstrs(
                (imported[t] <= d.max_import_kW for t in T), name="import_max"
            )
        if d.max_export_kW is not None:
            self.con["export_max"] = m.addConstrs(
                (exported[t] <= d.max_export_kW for t in T), name="export_max"
            )
        if is_q2_linear:
            deviation = self.var["deviation"]
            self.con["deviation_positive"] = m.addConstrs(
                (deviation[t] >= load[t] - d.reference_load[t] for t in T),
                name="deviation_positive",
            )
            self.con["deviation_negative"] = m.addConstrs(
                (deviation[t] >= d.reference_load[t] - load[t] for t in T),
                name="deviation_negative",
            )
            self.con["deviation_nonnegative"] = m.addConstrs(
                (deviation[t] >= 0 for t in T), name="deviation_nonnegative"
            )

        m.update()
        return self

    # ------------------------------------------------------------------ 3. solve
    def solve(self) -> Results:
        """Optimise and return primal values, objective and dual values."""
        m = self.m
        m.update()
        if m.NumConstrs == 0 and m.NumQConstrs == 0:
            raise NotImplementedError(
                "The model has no constraints: complete FlexibleConsumerModel.build() in src/model.py first."
            )
        m.optimize()
        status = _status_name(m.Status)
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"Optimisation ended with status {status}. Check the model (m.computeIIS() helps for infeasibility).")
        return self._extract_results(status)

    # --------------------------------------------------------------- extraction
    def _extract_results(self, status: str) -> Results:
        d, T = self.data, list(self.T)
        hourly = pd.DataFrame(index=pd.Index(T, name="hour"))
        hourly["price"] = d.energy_price
        hourly["pv_available"] = d.pv_available
        if d.reference_load is not None:
            hourly["reference_load"] = d.reference_load

        # Primal values: every hourly variable family in self.var becomes a column
        for name, v in self.var.items():
            if isinstance(v, gp.tupledict):
                hourly[name] = [v[t].X for t in T]
        scalars = {name: v.X for name, v in self.var.items() if isinstance(v, gp.Var)}

        # Dual values: every constraint family in self.con becomes a 'dual_<name>' column or scalar
        duals: dict[str, float] = {}
        for name, c in self.con.items():
            try:
                if isinstance(c, gp.tupledict):
                    hourly[f"dual_{name}"] = [_dual(c[t]) for t in T]
                else:
                    duals[name] = _dual(c)
            except (AttributeError, gp.GurobiError):
                # No duals available (e.g. model with integer variables)
                pass

        daily_metrics = self._daily_metrics(hourly)

        return Results(
            question=d.question,
            status=status,
            objective=self.m.ObjVal,
            hourly=hourly,
            duals=duals,
            meta={"scalar_variables": scalars},
            daily_metrics=daily_metrics,
        )

    def _daily_metrics(self, hourly: pd.DataFrame) -> dict[str, object]:
        """Compute daily cost, disutility, energy and bound diagnostics from a solution."""
        d = self.data
        procurement_cost = float(
            (hourly["import"] * (d.energy_price + d.import_tariff)
             - hourly["export"] * (d.energy_price - d.export_tariff)
               + hourly["pv"] * d.pv_marginal_cost).sum()
           )
        total_absolute_deviation = float(
            np.abs(hourly["load"].to_numpy() - d.reference_load).sum()
        ) if d.reference_load is not None else 0.0
        if "deviation" in hourly:
            total_disutility = float(d.linear_disutility * total_absolute_deviation)
        elif d.reference_load is not None and d.quadratic_disutility is not None:
            total_disutility = float(
                d.quadratic_disutility
                * np.square(hourly["load"].to_numpy() - d.reference_load).sum()
            )
        else:
            total_disutility = 0.0
        at_load_bound = np.isclose(hourly["load"], d.load_min_kWh, atol=1e-7) | np.isclose(
            hourly["load"], d.load_max_kWh, atol=1e-7
        )
        return {
            "daily_procurement_cost_DKK": procurement_cost,
            "total_disutility_DKK": total_disutility,
            "daily_energy_consumed_kWh": float(hourly["load"].sum()),
            "total_absolute_deviation_kWh": total_absolute_deviation,
            "deviation_bound_binding_hours": None,
            "deviation_bound_note": (
                "No explicit deviation bound is provided; the model only has load bounds. "
                "The reasonable related diagnostic is load_bound_binding_hours."
            ),
            "load_bound_binding_hours": int(at_load_bound.sum()),
        }


_STATUS = {
    GRB.OPTIMAL: "OPTIMAL", GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED",
    GRB.INF_OR_UNBD: "INF_OR_UNBD", GRB.TIME_LIMIT: "TIME_LIMIT", GRB.SUBOPTIMAL: "SUBOPTIMAL",
    GRB.NUMERIC: "NUMERIC", GRB.INTERRUPTED: "INTERRUPTED",
}


def _status_name(code: int) -> str:
    return _STATUS.get(code, f"STATUS_{code}")
