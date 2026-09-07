"""Load the input data of one case (``data/params_<case>.json``) into a single object.

Usage
-----
>>> from src.data_loader import load_question
>>> data = load_question("Q1_caseA")
>>> data.energy_price          # numpy array of 24 hourly prices, DKK/kWh
>>> data.pv_available          # numpy array of 24 hourly PV availability, kWh/h
>>> data.summary()             # human-readable overview

Data layout (see the README): ``data/`` holds two SHARED files --
``appliance_params.json`` (the catalogue of every PV system, flexible load and battery used
anywhere in the assignment) and ``bus_params.json`` (grid connection: prices and tariffs) --
plus one small ``params_<case>.json`` file per case that composes a consumer by picking
appliances from the catalogue.

The loader reads the two shared files and the case file, checks that they are consistent
(24 hourly values, ratios within [0, 1], appliance IDs that exist in the catalogue) and
exposes the quantities needed by the optimisation models as attributes with explicit units.
The raw dictionaries are kept in ``data.raw`` in case you need a field that is not exposed
as an attribute.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Repository root = parent of the ``src`` folder. Works whatever the current working directory.
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"

SHARED_FILES = (
    "appliance_params.json",
    "bus_params.json",
)


@dataclass
class InputData:
    """All parameters of one consumer for one case, with units in the attribute names."""

    question: str
    consumer_id: str
    hours: np.ndarray                     # 0 .. 23

    # Grid / market conditions (from bus_params.json)
    energy_price: np.ndarray              # DKK/kWh, one value per hour
    import_tariff: float                  # DKK/kWh charged on every kWh imported
    export_tariff: float                  # DKK/kWh charged on every kWh exported
    max_import_kW: float | None           # None = no grid limit in Assignment 1
    max_export_kW: float | None           # None = no grid limit in Assignment 1

    # PV system (from the "DERs" catalogue)
    pv_max_kW: float
    pv_profile: np.ndarray                # ratio of pv_max_kW available each hour, in [0, 1]
    pv_available: np.ndarray              # kWh/h = pv_max_kW * pv_profile
    pv_marginal_cost: float               # DKK per kWh of PV actually produced (0 if not given)

    # Flexible load (from the "loads" catalogue)
    load_max_kWh: float                   # maximum hourly consumption, kWh/h
    load_min_kWh: float                   # minimum hourly consumption, kWh/h
    consumption_utility: float | None     # DKK per kWh consumed (Question 1; None otherwise)
    min_daily_energy_kWh: float | None    # minimum energy to consume over the day (Question 3)
    reference_load: np.ndarray | None     # preferred hourly consumption, kWh/h
                                          # = load_max_kWh * reference_load_capacity_factor
    linear_disutility: float | None       # DKK per kWh of absolute deviation from reference_load
    quadratic_disutility: float | None    # DKK per kWh^2 of deviation from reference_load

    # Battery (Q3_battery only; from the "storages" catalogue; all None when absent)
    battery_capacity_kWh: float | None = None
    battery_max_charge_kW: float | None = None
    battery_max_discharge_kW: float | None = None
    battery_charging_efficiency: float | None = None      # fraction of charged energy reaching the battery
    battery_discharging_efficiency: float | None = None   # fraction of discharged energy reaching the consumer
    battery_initial_soc_kWh: float | None = None          # state of charge at the start of the day
    battery_final_soc_kWh: float | None = None            # None ON PURPOSE: the end-of-horizon treatment is
                                                          # your modeling choice (Question 3.(g))

    # Next-day forecasts (for exploring the end-of-horizon choice of Question 3.(g))
    energy_price_next_day: np.ndarray | None = None       # DKK/kWh, 24 values for the following day
    pv_available_next_day: np.ndarray | None = None       # kWh/h available the following day
    reference_load_next_day: np.ndarray | None = None     # preferred consumption the following day, kWh/h

    # Everything that was read, unprocessed
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------ helpers
    @property
    def n_hours(self) -> int:
        return len(self.hours)

    def summary(self) -> str:
        """Compact textual overview of the dataset."""
        lines = [
            f"Case                 : {self.question}",
            f"Consumer             : {self.consumer_id}",
            f"Energy price         : mean {self.energy_price.mean():.2f}, min {self.energy_price.min():.2f}, "
            f"max {self.energy_price.max():.2f} DKK/kWh",
            f"Import/export tariff : {self.import_tariff:.2f} / {self.export_tariff:.2f} DKK/kWh",
            f"PV                   : {self.pv_max_kW:.1f} kW peak, {self.pv_available.sum():.1f} kWh available/day, "
            f"marginal cost {self.pv_marginal_cost:.2f} DKK/kWh",
            f"Load bounds          : {self.load_min_kWh:.1f} - {self.load_max_kWh:.1f} kWh/h",
            f"Consumption utility  : {self.consumption_utility}",
            f"Min daily energy     : {self.min_daily_energy_kWh}",
            f"Reference profile    : {'yes (%.1f kWh/day)' % self.reference_load.sum() if self.reference_load is not None else 'no'}",
            f"Disutility lin/quad  : {self.linear_disutility} / {self.quadratic_disutility}",
            f"Battery              : "
            + (f"{self.battery_capacity_kWh:.1f} kWh, {self.battery_max_charge_kW:.1f}/"
               f"{self.battery_max_discharge_kW:.1f} kW, eta {self.battery_charging_efficiency:.2f}/"
               f"{self.battery_discharging_efficiency:.2f}, initial SoC {self.battery_initial_soc_kWh:.1f} kWh"
               if self.battery_capacity_kWh is not None else "no"),
            f"Next-day forecasts   : {'yes' if self.energy_price_next_day is not None else 'no'}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------- loading
def _read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _first_with(items: list[dict], key: str, value: str, what: str) -> dict:
    """Return the first dict in ``items`` whose ``key`` equals ``value``."""
    for item in items:
        if item.get(key) == value:
            return item
    raise KeyError(f"No {what} with {key} == {value!r}")


def _as_array(values, name: str, n_hours: int = 24) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.shape != (n_hours,):
        raise ValueError(f"{name} must have {n_hours} hourly values, got shape {arr.shape}")
    return arr


def _case_name(question: str) -> str:
    """Accept ``"Q1_caseA"``, ``"params_Q1_caseA"`` or ``"params_Q1_caseA.json"``."""
    name = question.removesuffix(".json")
    return name.removeprefix("params_")


def load_question(question: str, data_dir: Path | str = DATA_DIR, consumer_index: int = 0) -> InputData:
    """Read the shared catalogue plus ``params_<question>.json`` and return an :class:`InputData`.

    Parameters
    ----------
    question : str
        Name of the case, e.g. ``"Q1_caseA"`` (the ``params_``/``.json`` wrapping is optional).
    data_dir : Path, optional
        Data directory (defaults to ``<repo>/data``).
    consumer_index : int, optional
        Which consumer of the case file to load (Assignment 1 has one per case).
    """
    data_dir = Path(data_dir)
    case = _case_name(question)
    case_file = data_dir / f"params_{case}.json"
    if not case_file.exists():
        raise FileNotFoundError(
            f"Unknown case {question!r} (no {case_file.name} in {data_dir}). "
            f"Available: {list_questions(data_dir)}"
        )

    raw = {name.removesuffix(".json"): _read_json(data_dir / name) for name in SHARED_FILES}
    raw["params"] = _read_json(case_file)

    consumer = raw["params"]["consumers"][consumer_index]
    cid = consumer["consumer_id"]
    appliance_ids = list(consumer["list_appliances"])
    catalogue = raw["appliance_params"]

    # --- bus / market
    bus = _first_with(raw["bus_params"], "bus_id", consumer["connection_bus"], "bus")
    price = _as_array(bus["energy_price_DKK_per_kWh"], "energy_price_DKK_per_kWh")
    if (price < 0).any():
        raise ValueError("Energy prices are assumed non-negative in Assignment 1")

    def opt_float(d: dict, key: str) -> float | None:
        v = d.get(key)
        return None if v is None else float(v)

    # --- PV (the consumer's DER of type PV in the catalogue)
    ders = catalogue.get("DERs") or []
    pv = next((d for d in ders if d["DER_id"] in appliance_ids and d["DER_type"] == "PV"), None)
    if pv is None:
        raise KeyError(f"None of {appliance_ids} is a PV system in the 'DERs' catalogue")
    pv_max = float(pv["max_power_kW"])
    pv_profile = _as_array(pv["hourly_profile_ratio"], "PV hourly_profile_ratio")
    if (pv_profile < 0).any() or (pv_profile > 1).any():
        raise ValueError("PV hourly_profile_ratio must lie in [0, 1]")

    # --- flexible load (in the catalogue; parameters not used by a case are null)
    loads = catalogue.get("loads") or []
    load = next((l for l in loads if l["load_id"] in appliance_ids), None)
    if load is None:
        raise KeyError(f"None of {appliance_ids} is in the 'loads' catalogue")
    load_max = float(load["max_load_KWh_per_hour"])
    ref_factor = load.get("reference_load_capacity_factor")
    reference_load = (None if ref_factor is None
                      else load_max * _as_array(ref_factor, "reference_load_capacity_factor"))
    if reference_load is not None and ((reference_load < 0).any() or (reference_load > load_max).any()):
        raise ValueError("reference_load_capacity_factor must lie in [0, 1]")

    # --- battery (Q3_battery only; None elsewhere)
    storages = catalogue.get("storages") or []
    bat = next((s for s in storages if s.get("storage_id") in appliance_ids), None)
    bat_kwargs: dict[str, Any] = {}
    if bat is not None:
        cap = float(bat["storage_capacity_kWh"])
        bat_kwargs = dict(
            battery_capacity_kWh=cap,
            battery_max_charge_kW=float(bat["max_charging_power_kW"]),
            battery_max_discharge_kW=float(bat["max_discharging_power_kW"]),
            battery_charging_efficiency=float(bat["charging_efficiency"]),
            battery_discharging_efficiency=float(bat["discharging_efficiency"]),
            battery_initial_soc_kWh=cap * float(bat["initial_state_of_charge_ratio"]),
        )
        # battery_final_soc_kWh stays None on purpose (your choice in Question 3.(g)).

    # --- next-day forecasts (present in the shared files for all cases)
    nd_price = bus.get("energy_price_next_day_DKK_per_kWh")
    nd_pv = pv.get("hourly_profile_ratio_next_day")
    nd_ref = load.get("reference_hourly_profile_ratio_next_day")
    nd_kwargs = dict(
        energy_price_next_day=None if nd_price is None
            else _as_array(nd_price, "energy_price_next_day_DKK_per_kWh"),
        pv_available_next_day=None if nd_pv is None
            else pv_max * _as_array(nd_pv, "hourly_profile_ratio_next_day"),
        reference_load_next_day=None if nd_ref is None
            else load_max * _as_array(nd_ref, "reference_hourly_profile_ratio_next_day"),
    )

    return InputData(
        question=case,
        consumer_id=cid,
        hours=np.arange(len(price)),
        energy_price=price,
        import_tariff=float(bus["import_tariff_DKK_per_kWh"]),
        export_tariff=float(bus["export_tariff_DKK_per_kWh"]),
        max_import_kW=opt_float(bus, "max_import_kW"),
        max_export_kW=opt_float(bus, "max_export_kW"),
        pv_max_kW=pv_max,
        pv_profile=pv_profile,
        pv_available=pv_max * pv_profile,
        pv_marginal_cost=float(pv.get("marginal_cost_DKK_per_kWh") or 0.0),
        load_max_kWh=load_max,
        load_min_kWh=float(load.get("min_load_KWh_per_hour") or 0.0),
        consumption_utility=opt_float(load, "consumption_utility_DKK_per_kWh"),
        min_daily_energy_kWh=opt_float(load, "min_total_energy_per_day_kWh"),
        reference_load=reference_load,
        linear_disutility=opt_float(load, "linear_disutility_DKK_per_kWh"),
        quadratic_disutility=opt_float(load, "quadratic_disutility_DKK_per_kWh2"),
        **bat_kwargs,
        **nd_kwargs,
        raw=raw,
    )


def list_questions(data_dir: Path | str = DATA_DIR) -> list[str]:
    """Names of all cases available under ``data/`` (one per ``params_*.json``)."""
    return sorted(p.stem.removeprefix("params_") for p in Path(data_dir).glob("params_*.json"))


if __name__ == "__main__":  # quick manual check:  python -m src.data_loader
    for q in list_questions():
        print(load_question(q).summary(), end="\n\n")
