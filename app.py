"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SIOP DIGITAL TWIN — Enterprise Planning Platform                    ║
║         Tier 1 Automotive OEM Manufacturing                                 ║
║                                                                              ║
║  Stack : Streamlit · Pandas · Plotly · NumPy                                ║
║  Model : 12-month rolling horizon · 3 products · 3 OEM customers            ║
║  Logic : Demand → Capacity Constraint → Supply Plan → Financial P&L         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:  streamlit run app.py
"""

# ─────────────────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import Dict, Tuple
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be the first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SIOP Digital Twin",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# COLOUR PALETTE  (consistent across all charts)
# ─────────────────────────────────────────────────────────────────────────────
PALETTE = {
    "navy":     "#1F3864",
    "blue":     "#2E75B6",
    "lblue":    "#BDD7EE",
    "green":    "#00B050",
    "dgreen":   "#375623",
    "orange":   "#ED7D31",
    "amber":    "#FFC000",
    "red":      "#C00000",
    "purple":   "#7030A0",
    "lgray":    "#F2F2F2",
    "mgray":    "#D9D9D9",
    "dgray":    "#595959",
    "baseline": "#2E75B6",
    "scenario": "#ED7D31",
    "positive": "#00B050",
    "negative": "#C00000",
}

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
MONTHS = pd.date_range("2025-01-01", periods=12, freq="MS")
MONTH_LABELS = [m.strftime("%b %Y") for m in MONTHS]
N_MONTHS = 12

PRODUCTS = {
    "P001": "Front Suspension Bracket",
    "P002": "Transmission Mount",
    "P003": "Engine Cradle Assembly",
}
CUSTOMERS = {
    "OEM-A": "Ford (F-150)",
    "OEM-B": "GM (Silverado)",
    "OEM-C": "Stellantis (Ram)",
}
MACHINES = {
    "MC-01": "Stamping Press",
    "MC-02": "Robotic MIG Weld",
    "MC-03": "CMM Inspection",
}


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 1 — SYNTHETIC DATA GENERATOR (replaces ERP / EDI data feed)
# ═════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading ERP master data…")
def generate_master_data() -> Dict[str, pd.DataFrame]:
    """
    Generates a complete set of synthetic ERP master data tables.
    Cached so it only runs once per session (simulates a slow ERP API call).

    Returns
    -------
    dict of DataFrames:
        products, bom, routings, machines, inventory,
        demand_forecast, pricing, raw_materials
    """
    rng = np.random.default_rng(42)

    # ── PRODUCT MASTER ────────────────────────────────────────────────────
    products = pd.DataFrame([
        {"product_code": "P001", "product_name": "Front Suspension Bracket",
         "uom": "EA", "product_family": "Chassis", "weight_kg": 4.2},
        {"product_code": "P002", "product_name": "Transmission Mount",
         "uom": "EA", "product_family": "Powertrain", "weight_kg": 3.8},
        {"product_code": "P003", "product_name": "Engine Cradle Assembly",
         "uom": "EA", "product_family": "Powertrain", "weight_kg": 11.5},
    ])

    # ── BILL OF MATERIALS ─────────────────────────────────────────────────
    bom = pd.DataFrame([
        # P001
        {"product_code": "P001", "component_code": "RM-001",
         "component_name": "Hot Roll Steel Coil",
         "qty_per_unit": 8.5, "uom": "kg", "scrap_pct": 0.03},
        {"product_code": "P001", "component_code": "RM-002",
         "component_name": "Cold Roll Sheet",
         "qty_per_unit": 2.0, "uom": "kg", "scrap_pct": 0.02},
        {"product_code": "P001", "component_code": "RM-005",
         "component_name": "Hardware Kit",
         "qty_per_unit": 1.0, "uom": "EA", "scrap_pct": 0.00},
        # P002
        {"product_code": "P002", "component_code": "RM-001",
         "component_name": "Hot Roll Steel Coil",
         "qty_per_unit": 6.2, "uom": "kg", "scrap_pct": 0.03},
        {"product_code": "P002", "component_code": "RM-003",
         "component_name": "Steel Tubing 50mm",
         "qty_per_unit": 1.5, "uom": "kg", "scrap_pct": 0.02},
        {"product_code": "P002", "component_code": "RM-005",
         "component_name": "Hardware Kit",
         "qty_per_unit": 1.0, "uom": "EA", "scrap_pct": 0.00},
        # P003
        {"product_code": "P003", "component_code": "RM-001",
         "component_name": "Hot Roll Steel Coil",
         "qty_per_unit": 12.0, "uom": "kg", "scrap_pct": 0.04},
        {"product_code": "P003", "component_code": "RM-003",
         "component_name": "Steel Tubing 50mm",
         "qty_per_unit": 3.5, "uom": "kg", "scrap_pct": 0.02},
        {"product_code": "P003", "component_code": "RM-004",
         "component_name": "Iron Casting",
         "qty_per_unit": 2.0, "uom": "EA", "scrap_pct": 0.01},
        {"product_code": "P003", "component_code": "RM-005",
         "component_name": "Hardware Kit",
         "qty_per_unit": 2.0, "uom": "EA", "scrap_pct": 0.00},
    ])
    bom["effective_qty"] = bom["qty_per_unit"] * (1 + bom["scrap_pct"])

    # ── RAW MATERIAL MASTER ───────────────────────────────────────────────
    raw_materials = pd.DataFrame([
        {"component_code": "RM-001", "name": "Hot Roll Steel Coil",
         "supplier": "SteelCo USA",      "lead_time_wks": 4,
         "unit_cost": 0.85,  "current_stock": 45000, "safety_stock": 15000,
         "moq": 10000, "uom": "kg"},
        {"component_code": "RM-002", "name": "Cold Roll Sheet",
         "supplier": "SteelCo USA",      "lead_time_wks": 4,
         "unit_cost": 1.20,  "current_stock": 22000, "safety_stock": 8000,
         "moq": 5000,  "uom": "kg"},
        {"component_code": "RM-003", "name": "Steel Tubing 50mm",
         "supplier": "TubeMaster Inc",   "lead_time_wks": 6,
         "unit_cost": 2.45,  "current_stock": 8500,  "safety_stock": 3000,
         "moq": 2000,  "uom": "kg"},
        {"component_code": "RM-004", "name": "Iron Casting",
         "supplier": "Precision Cast.",  "lead_time_wks": 8,
         "unit_cost": 18.50, "current_stock": 1200,  "safety_stock": 500,
         "moq": 500,   "uom": "EA"},
        {"component_code": "RM-005", "name": "Hardware Kit",
         "supplier": "FastenerWorld",    "lead_time_wks": 3,
         "unit_cost": 4.20,  "current_stock": 12000, "safety_stock": 4000,
         "moq": 1000,  "uom": "EA"},
    ])
    raw_materials["available_stock"] = (
        raw_materials["current_stock"] - raw_materials["safety_stock"]
    ).clip(lower=0)

    # ── MACHINE ROUTINGS ──────────────────────────────────────────────────
    routings = pd.DataFrame([
        {"product_code": "P001", "machine_id": "MC-01",
         "machine_name": "Stamping Press",   "cycle_min": 2.5,
         "labor_rate": 35.0, "machine_rate": 85.0},
        {"product_code": "P001", "machine_id": "MC-02",
         "machine_name": "Robotic MIG Weld", "cycle_min": 4.0,
         "labor_rate": 40.0, "machine_rate": 60.0},
        {"product_code": "P001", "machine_id": "MC-03",
         "machine_name": "CMM Inspection",   "cycle_min": 0.8,
         "labor_rate": 45.0, "machine_rate": 55.0},
        {"product_code": "P002", "machine_id": "MC-01",
         "machine_name": "Stamping Press",   "cycle_min": 3.2,
         "labor_rate": 35.0, "machine_rate": 85.0},
        {"product_code": "P002", "machine_id": "MC-02",
         "machine_name": "Robotic MIG Weld", "cycle_min": 5.5,
         "labor_rate": 40.0, "machine_rate": 60.0},
        {"product_code": "P002", "machine_id": "MC-03",
         "machine_name": "CMM Inspection",   "cycle_min": 1.2,
         "labor_rate": 45.0, "machine_rate": 55.0},
        {"product_code": "P003", "machine_id": "MC-01",
         "machine_name": "Stamping Press",   "cycle_min": 4.8,
         "labor_rate": 35.0, "machine_rate": 85.0},
        {"product_code": "P003", "machine_id": "MC-02",
         "machine_name": "Robotic MIG Weld", "cycle_min": 8.5,
         "labor_rate": 40.0, "machine_rate": 60.0},
        {"product_code": "P003", "machine_id": "MC-03",
         "machine_name": "CMM Inspection",   "cycle_min": 2.0,
         "labor_rate": 45.0, "machine_rate": 55.0},
    ])
    # Standard routing cost per unit
    routings["std_cost_per_unit"] = (
        routings["cycle_min"] / 60.0
        * (routings["labor_rate"] + routings["machine_rate"])
    )

    # ── MACHINE CAPACITY PARAMETERS ───────────────────────────────────────
    machines = pd.DataFrame([
        {"machine_id": "MC-01", "machine_name": "Stamping Press",
         "shifts": 2, "hrs_per_shift": 8, "days_per_month": 21, "oee": 0.82},
        {"machine_id": "MC-02", "machine_name": "Robotic MIG Weld",
         "shifts": 2, "hrs_per_shift": 8, "days_per_month": 21, "oee": 0.88},
        {"machine_id": "MC-03", "machine_name": "CMM Inspection",
         "shifts": 1, "hrs_per_shift": 8, "days_per_month": 21, "oee": 0.92},
    ])
    machines["available_hrs_month"] = (
        machines["shifts"]
        * machines["hrs_per_shift"]
        * machines["days_per_month"]
        * machines["oee"]
    )

    # ── PRICING / STANDARD COST MASTER ────────────────────────────────────
    pricing = pd.DataFrame([
        {"product_code": "P001", "selling_price": 285.00,
         "std_material_cost": 48.50, "std_labor_cost": 18.20,
         "std_overhead_cost": 22.50},
        {"product_code": "P002", "selling_price": 420.00,
         "std_material_cost": 62.30, "std_labor_cost": 28.40,
         "std_overhead_cost": 35.10},
        {"product_code": "P003", "selling_price": 875.00,
         "std_material_cost": 135.80, "std_labor_cost": 52.60,
         "std_overhead_cost": 68.40},
    ])
    pricing["total_std_cost"] = (
        pricing["std_material_cost"]
        + pricing["std_labor_cost"]
        + pricing["std_overhead_cost"]
    )
    pricing["std_gross_profit"] = pricing["selling_price"] - pricing["total_std_cost"]
    pricing["std_gm_pct"] = pricing["std_gross_profit"] / pricing["selling_price"]

    # ── DEMAND FORECAST (12 months × product × customer) ─────────────────
    base_monthly = {
        ("P001", "OEM-A"): [3280, 3340, 3200, 3480, 3560, 3280, 3120,
                             3600, 3680, 3400, 3240, 3480],
        ("P001", "OEM-B"): [2880, 2780, 2960, 2840, 2920, 2760, 3000,
                             2880, 2800, 2860, 2960, 3040],
        ("P001", "OEM-C"): [1920, 2000, 1840, 1960, 2040, 1880, 1980,
                             2040, 1920, 1860, 2000, 1940],
        ("P002", "OEM-A"): [1220, 1280, 1180, 1240, 1320, 1160, 1260,
                             1360, 1200, 1240, 1280, 1220],
        ("P002", "OEM-B"): [1060, 1020, 1100, 1040, 1080, 1020, 1120,
                             1060, 1000, 1080, 1040, 1100],
        ("P002", "OEM-C"): [780, 820, 760, 800, 840, 740, 800, 860,
                             780, 800, 820, 780],
        ("P003", "OEM-A"): [312, 328, 300, 320, 340, 288, 320, 360,
                             300, 320, 340, 312],
        ("P003", "OEM-B"): [272, 260, 288, 272, 280, 260, 288, 272,
                             248, 280, 264, 288],
        ("P003", "OEM-C"): [192, 208, 180, 200, 220, 168, 200, 232,
                             180, 200, 208, 192],
    }
    demand_rows = []
    for (prod, cust), vals in base_monthly.items():
        for mi, (month, qty) in enumerate(zip(MONTHS, vals)):
            demand_rows.append({
                "month":          month,
                "month_label":    MONTH_LABELS[mi],
                "product_code":   prod,
                "customer_code":  cust,
                "product_name":   PRODUCTS[prod],
                "customer_name":  CUSTOMERS[cust],
                "forecast_qty":   qty,
            })
    demand_forecast = pd.DataFrame(demand_rows)

    # ── OPENING INVENTORY (finished goods) ────────────────────────────────
    inventory = pd.DataFrame([
        {"product_code": "P001", "product_name": PRODUCTS["P001"],
         "opening_stock": 2400, "safety_stock": 800,
         "avg_monthly_demand": 2700},
        {"product_code": "P002", "product_name": PRODUCTS["P002"],
         "opening_stock": 900, "safety_stock": 300,
         "avg_monthly_demand": 1030},
        {"product_code": "P003", "product_name": PRODUCTS["P003"],
         "opening_stock": 280, "safety_stock": 80,
         "avg_monthly_demand": 256},
    ])

    return {
        "products":        products,
        "bom":             bom,
        "routings":        routings,
        "machines":        machines,
        "raw_materials":   raw_materials,
        "demand_forecast": demand_forecast,
        "pricing":         pricing,
        "inventory":       inventory,
    }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 2 — DIGITAL BRAIN: CALCULATION ENGINE
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class ScenarioParams:
    """
    Encapsulates all scenario levers adjustable by the planner.
    Passed into the engine for both baseline (all defaults) and scenario runs.
    """
    # Demand
    demand_multiplier:      float = 1.00   # global demand shift
    oem_a_multiplier:       float = 1.00   # OEM-A specific
    oem_b_multiplier:       float = 1.00   # OEM-B specific
    oem_c_multiplier:       float = 1.00   # OEM-C specific

    # Cost
    steel_cost_delta_pct:   float = 0.00   # % change in RM-001/RM-002 cost
    freight_cost_per_unit:  float = 0.00   # expedited freight adder ($/unit)
    overhead_delta_pct:     float = 0.00   # overhead absorption change

    # Capacity
    capacity_multiplier:    float = 1.00   # OEE/capacity factor (0.8 = -20%)
    overtime_hrs_month:     float = 0.00   # additional hours per machine per month

    # Working Capital
    ar_days:                float = 45.0   # accounts receivable days
    ap_days:                float = 60.0   # accounts payable days
    inventory_turns_target: float = 8.0    # target inventory turns

    # Exception Thresholds
    capacity_alert_pct:     float = 0.95   # alert if utilisation > this
    margin_alert_pct:       float = 0.20   # alert if GM% < this


class SIOPEngine:
    """
    Vectorised calculation engine implementing the SIOP knowledge graph logic.

    Data flows in a strict dependency order:
        Demand Plan
            → Capacity Loading (constraint evaluation)
            → Constrained Supply Plan
            → Material Requirements (BOM explosion)
            → Financial P&L (Revenue → COGS → GM → Working Capital)

    All operations use Pandas vectorisation — no Python loops over rows.
    """

    def __init__(self, master: Dict[str, pd.DataFrame]):
        self.products      = master["products"]
        self.bom           = master["bom"]
        self.routings      = master["routings"]
        self.machines      = master["machines"]
        self.raw_materials = master["raw_materials"]
        self.demand_base   = master["demand_forecast"]
        self.pricing       = master["pricing"]
        self.inventory     = master["inventory"]

    # ── STEP 1: DEMAND PLAN ───────────────────────────────────────────────
    def build_demand_plan(self, params: ScenarioParams) -> pd.DataFrame:
        """
        Applies scenario multipliers to the baseline forecast.
        Returns a month × product aggregated demand DataFrame.
        """
        df = self.demand_base.copy()

        # Customer-level multipliers
        cust_mult = {
            "OEM-A": params.demand_multiplier * params.oem_a_multiplier,
            "OEM-B": params.demand_multiplier * params.oem_b_multiplier,
            "OEM-C": params.demand_multiplier * params.oem_c_multiplier,
        }
        df["mult"] = df["customer_code"].map(cust_mult)
        df["adjusted_qty"] = (df["forecast_qty"] * df["mult"]).round(0)

        # Aggregate to month × product
        agg = (
            df.groupby(["month", "month_label", "product_code", "product_name"])
            ["adjusted_qty"].sum()
            .reset_index()
            .rename(columns={"adjusted_qty": "demand_qty"})
        )
        return agg

    # ── STEP 2: CAPACITY LOADING ──────────────────────────────────────────
    def compute_capacity_loading(
        self,
        demand: pd.DataFrame,
        params: ScenarioParams,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Explodes demand through routings to calculate required machine-hours
        vs. available machine-hours.

        Returns
        -------
        loading : month × machine utilisation DataFrame
        machine_avail : updated available hours (with capacity factor + OT)
        """
        # Adjusted available hours
        mach = self.machines.copy()
        mach["avail_hrs"] = (
            mach["available_hrs_month"] * params.capacity_multiplier
            + params.overtime_hrs_month
        )

        # Required hours: merge demand with routings, sum by machine × month
        req = demand.merge(
            self.routings[["product_code", "machine_id", "machine_name", "cycle_min"]],
            on="product_code",
            how="left",
        )
        req["required_hrs"] = req["demand_qty"] * req["cycle_min"] / 60.0

        loading = (
            req.groupby(["month", "month_label", "machine_id", "machine_name"])
            ["required_hrs"].sum()
            .reset_index()
        )
        loading = loading.merge(
            mach[["machine_id", "avail_hrs"]],
            on="machine_id",
            how="left",
        )
        loading["utilisation_pct"] = (
            loading["required_hrs"] / loading["avail_hrs"]
        ).clip(upper=2.0)  # cap at 200% for display sanity

        return loading, mach

    # ── STEP 3: CONSTRAINED SUPPLY PLAN ──────────────────────────────────
    def build_supply_plan(
        self,
        demand: pd.DataFrame,
        loading: pd.DataFrame,
        params: ScenarioParams,
    ) -> pd.DataFrame:
        """
        Constrains demand by the most-loaded machine (bottleneck).
        For each month, the binding machine determines max producible units.
        """
        mach_avail = self.machines.copy()
        mach_avail["avail_hrs"] = (
            mach_avail["available_hrs_month"] * params.capacity_multiplier
            + params.overtime_hrs_month
        )

        # For each product × month, find the bottleneck cycle time
        # (the machine with the least surplus capacity)
        demand_w = demand.copy()

        rows = []
        for prod in demand_w["product_code"].unique():
            rt = self.routings[self.routings["product_code"] == prod]
            prod_demand = demand_w[demand_w["product_code"] == prod].copy()

            for _, row in prod_demand.iterrows():
                d_qty = row["demand_qty"]
                max_possible = []
                for _, r in rt.iterrows():
                    mc = mach_avail[mach_avail["machine_id"] == r["machine_id"]]
                    if len(mc) > 0:
                        avail_h = mc.iloc[0]["avail_hrs"]
                        max_from_machine = (avail_h * 60.0) / r["cycle_min"]
                        max_possible.append(max_from_machine)
                bottleneck_max = min(max_possible) if max_possible else d_qty
                constrained_qty = min(d_qty, bottleneck_max)
                shortage = max(0.0, d_qty - constrained_qty)
                rows.append({
                    "month":           row["month"],
                    "month_label":     row["month_label"],
                    "product_code":    prod,
                    "product_name":    row["product_name"],
                    "demand_qty":      d_qty,
                    "constrained_qty": constrained_qty,
                    "shortage_qty":    shortage,
                    "service_level":   constrained_qty / d_qty if d_qty > 0 else 1.0,
                })

        supply = pd.DataFrame(rows)
        return supply

    # ── STEP 4: INVENTORY PROJECTION ─────────────────────────────────────
    def project_inventory(self, supply: pd.DataFrame) -> pd.DataFrame:
        """
        Rolls forward finished-goods inventory month by month.
        Production = constrained_qty (we produce what we can).
        Shipment = min(demand, opening_stock + production).
        """
        rows = []
        opening = dict(zip(
            self.inventory["product_code"],
            self.inventory["opening_stock"],
        ))
        safety = dict(zip(
            self.inventory["product_code"],
            self.inventory["safety_stock"],
        ))

        for month_label in MONTH_LABELS:
            month_data = supply[supply["month_label"] == month_label]
            for prod in PRODUCTS:
                row = month_data[month_data["product_code"] == prod]
                if len(row) == 0:
                    continue
                row = row.iloc[0]
                open_stock = opening.get(prod, 0)
                production = row["constrained_qty"]
                demand     = row["demand_qty"]
                available  = open_stock + production
                shipped    = min(available, demand)
                close_stock = max(0.0, available - shipped)
                ss          = safety.get(prod, 0)
                rows.append({
                    "month_label":   month_label,
                    "product_code":  prod,
                    "product_name":  row["product_name"],
                    "opening_stock": open_stock,
                    "production":    production,
                    "demand":        demand,
                    "shipped":       shipped,
                    "closing_stock": close_stock,
                    "safety_stock":  ss,
                    "below_safety":  close_stock < ss,
                })
                opening[prod] = close_stock

        return pd.DataFrame(rows)

    # ── STEP 5: FINANCIAL P&L ENGINE ─────────────────────────────────────
    def compute_financials(
        self,
        supply: pd.DataFrame,
        params: ScenarioParams,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Translates every constrained unit into financial outcomes.
        Financial-first: Revenue → COGS (Material, Labor, OH, Freight) →
        Gross Profit → Working Capital.

        Returns
        -------
        monthly_pnl : month-level P&L DataFrame
        product_pnl : product-level summary
        """
        # Build adjusted pricing
        pr = self.pricing.copy()

        # Apply RM cost delta (steel-heavy products P001, P002, P003 all use steel)
        # Recalculate material cost from BOM
        bom_cost = (
            self.bom
            .merge(
                self.raw_materials[["component_code", "unit_cost"]],
                on="component_code",
                how="left",
            )
        )
        # Apply steel cost multiplier to RM-001 and RM-002
        steel_mask = bom_cost["component_code"].isin(["RM-001", "RM-002"])
        bom_cost.loc[steel_mask, "unit_cost"] *= (1 + params.steel_cost_delta_pct)

        bom_cost["line_cost"] = bom_cost["effective_qty"] * bom_cost["unit_cost"]
        recalc_mat = (
            bom_cost.groupby("product_code")["line_cost"]
            .sum()
            .reset_index()
            .rename(columns={"line_cost": "material_cost_recalc"})
        )
        pr = pr.merge(recalc_mat, on="product_code", how="left")

        # Apply overhead delta
        pr["adj_overhead"] = pr["std_overhead_cost"] * (1 + params.overhead_delta_pct)
        pr["adj_total_cogs"] = (
            pr["material_cost_recalc"]
            + pr["std_labor_cost"]
            + pr["adj_overhead"]
            + params.freight_cost_per_unit   # expedited freight adder
        )
        pr["adj_gross_profit"] = pr["selling_price"] - pr["adj_total_cogs"]
        pr["adj_gm_pct"]       = pr["adj_gross_profit"] / pr["selling_price"]

        # Merge into supply plan
        fin = supply.merge(
            pr[["product_code", "selling_price", "material_cost_recalc",
                "std_labor_cost", "adj_overhead", "adj_total_cogs",
                "adj_gross_profit", "adj_gm_pct"]],
            on="product_code",
            how="left",
        )

        # Revenue & cost lines (on constrained volume only — financial reality)
        fin["revenue"]         = fin["constrained_qty"] * fin["selling_price"]
        fin["material_cogs"]   = fin["constrained_qty"] * fin["material_cost_recalc"]
        fin["labor_cogs"]      = fin["constrained_qty"] * fin["std_labor_cost"]
        fin["overhead_cogs"]   = fin["constrained_qty"] * fin["adj_overhead"]
        fin["freight_cost"]    = fin["constrained_qty"] * params.freight_cost_per_unit
        fin["total_cogs"]      = fin["constrained_qty"] * fin["adj_total_cogs"]
        fin["gross_profit"]    = fin["constrained_qty"] * fin["adj_gross_profit"]

        # Lost revenue from shortages
        fin["lost_revenue"]    = fin["shortage_qty"] * fin["selling_price"]

        # Monthly P&L aggregation
        monthly_pnl = (
            fin.groupby(["month", "month_label"])
            .agg(
                revenue       = ("revenue",       "sum"),
                material_cogs = ("material_cogs",  "sum"),
                labor_cogs    = ("labor_cogs",     "sum"),
                overhead_cogs = ("overhead_cogs",  "sum"),
                freight_cost  = ("freight_cost",   "sum"),
                total_cogs    = ("total_cogs",     "sum"),
                gross_profit  = ("gross_profit",   "sum"),
                lost_revenue  = ("lost_revenue",   "sum"),
            )
            .reset_index()
        )
        monthly_pnl["gm_pct"]       = monthly_pnl["gross_profit"] / monthly_pnl["revenue"]
        monthly_pnl["month_label"]  = pd.Categorical(
            monthly_pnl["month_label"], categories=MONTH_LABELS, ordered=True
        )
        monthly_pnl = monthly_pnl.sort_values("month_label")

        # Working capital
        monthly_pnl["ar_balance"]   = monthly_pnl["revenue"]      * params.ar_days / 30.0
        monthly_pnl["ap_balance"]   = monthly_pnl["material_cogs"] * params.ap_days / 30.0

        # Product-level summary
        product_pnl = (
            fin.groupby(["product_code", "product_name"])
            .agg(
                total_revenue  = ("revenue",       "sum"),
                total_cogs     = ("total_cogs",    "sum"),
                total_gp       = ("gross_profit",  "sum"),
                total_units    = ("constrained_qty","sum"),
                total_shortage = ("shortage_qty",  "sum"),
            )
            .reset_index()
        )
        product_pnl["gm_pct"] = product_pnl["total_gp"] / product_pnl["total_revenue"]

        return monthly_pnl, product_pnl

    # ── MASTER RUN ────────────────────────────────────────────────────────
    def run(self, params: ScenarioParams) -> dict:
        """
        Executes the full calculation chain and returns all outputs.
        """
        demand    = self.build_demand_plan(params)
        loading, mach_avail = self.compute_capacity_loading(demand, params)
        supply    = self.build_supply_plan(demand, loading, params)
        inventory = self.project_inventory(supply)
        monthly_pnl, product_pnl = self.compute_financials(supply, params)

        return {
            "demand":      demand,
            "loading":     loading,
            "mach_avail":  mach_avail,
            "supply":      supply,
            "inventory":   inventory,
            "monthly_pnl": monthly_pnl,
            "product_pnl": product_pnl,
        }


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 3 — EXCEPTION MANAGEMENT ENGINE
# ═════════════════════════════════════════════════════════════════════════════

def detect_exceptions(
    results: dict,
    params: ScenarioParams,
    plan_label: str,
) -> list:
    """
    Scans all plan outputs and returns a list of exception dicts.
    Alerts are classified as CRITICAL / WARNING / INFO.
    """
    exceptions = []
    loading     = results["loading"]
    monthly_pnl = results["monthly_pnl"]
    inventory   = results["inventory"]
    supply      = results["supply"]

    # ── CAPACITY ALERTS ───────────────────────────────────────────────────
    cap_breaches = loading[loading["utilisation_pct"] > params.capacity_alert_pct]
    for _, row in cap_breaches.iterrows():
        severity = "CRITICAL" if row["utilisation_pct"] > 1.0 else "WARNING"
        exceptions.append({
            "plan":      plan_label,
            "severity":  severity,
            "category":  "Capacity",
            "month":     row["month_label"],
            "machine":   row["machine_name"],
            "message":   (
                f"{row['machine_name']} utilisation at "
                f"{row['utilisation_pct']:.1%} in {row['month_label']} — "
                f"{'OVER CAPACITY' if row['utilisation_pct'] > 1.0 else 'approaching limit'}"
            ),
            "value":     row["utilisation_pct"],
            "threshold": params.capacity_alert_pct,
        })

    # ── MARGIN ALERTS ─────────────────────────────────────────────────────
    margin_breaches = monthly_pnl[monthly_pnl["gm_pct"] < params.margin_alert_pct]
    for _, row in margin_breaches.iterrows():
        exceptions.append({
            "plan":      plan_label,
            "severity":  "CRITICAL" if row["gm_pct"] < params.margin_alert_pct * 0.75 else "WARNING",
            "category":  "Margin",
            "month":     row["month_label"],
            "machine":   "–",
            "message":   (
                f"Gross margin {row['gm_pct']:.1%} is below "
                f"{params.margin_alert_pct:.0%} threshold in {row['month_label']}"
            ),
            "value":     row["gm_pct"],
            "threshold": params.margin_alert_pct,
        })

    # ── INVENTORY ALERTS ──────────────────────────────────────────────────
    inv_breaches = inventory[inventory["below_safety"]]
    for _, row in inv_breaches.iterrows():
        exceptions.append({
            "plan":      plan_label,
            "severity":  "WARNING",
            "category":  "Inventory",
            "month":     row["month_label"],
            "machine":   row["product_code"],
            "message":   (
                f"{row['product_name']} closing stock ({row['closing_stock']:.0f}) "
                f"below safety stock ({row['safety_stock']:.0f}) in {row['month_label']}"
            ),
            "value":     row["closing_stock"],
            "threshold": row["safety_stock"],
        })

    # ── SERVICE LEVEL ALERTS ──────────────────────────────────────────────
    sl_breaches = supply[supply["service_level"] < 0.98]
    for _, row in sl_breaches.iterrows():
        exceptions.append({
            "plan":      plan_label,
            "severity":  "WARNING",
            "category":  "Service Level",
            "month":     row["month_label"],
            "machine":   row["product_code"],
            "message":   (
                f"{row['product_name']} service level {row['service_level']:.1%} "
                f"(shortage: {row['shortage_qty']:.0f} units) in {row['month_label']}"
            ),
            "value":     row["service_level"],
            "threshold": 0.98,
        })

    return exceptions


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 4 — PLOTLY CHART LIBRARY
# ═════════════════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Arial", size=11, color="#1F3864"),
    margin=dict(l=40, r=20, t=50, b=40),
    legend=dict(bgcolor="rgba(255,255,255,0.8)", borderwidth=1),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor="#F2F2F2", zeroline=False),
)


def chart_revenue_cogs(
    base: pd.DataFrame,
    scen: pd.DataFrame,
) -> go.Figure:
    """Grouped bar: monthly Revenue and Gross Profit — Baseline vs Scenario."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=("Net Revenue ($)", "Gross Profit ($)"),
        vertical_spacing=0.12,
    )
    # Revenue
    fig.add_trace(go.Bar(
        name="Baseline Revenue", x=MONTH_LABELS,
        y=base["revenue"], marker_color=PALETTE["baseline"],
        opacity=0.85, offsetgroup="base",
    ), row=1, col=1)
    fig.add_trace(go.Bar(
        name="Scenario Revenue", x=MONTH_LABELS,
        y=scen["revenue"], marker_color=PALETTE["scenario"],
        opacity=0.85, offsetgroup="scen",
    ), row=1, col=1)
    # Gross Profit
    fig.add_trace(go.Bar(
        name="Baseline GP", x=MONTH_LABELS,
        y=base["gross_profit"], marker_color=PALETTE["baseline"],
        opacity=0.70, showlegend=False, offsetgroup="base",
    ), row=2, col=1)
    fig.add_trace(go.Bar(
        name="Scenario GP", x=MONTH_LABELS,
        y=scen["gross_profit"], marker_color=PALETTE["scenario"],
        opacity=0.70, showlegend=False, offsetgroup="scen",
    ), row=2, col=1)

    fig.update_layout(
        **CHART_LAYOUT,
        title_text="Revenue & Gross Profit: Baseline vs Scenario",
        barmode="group",
        height=480,
    )
    fig.update_yaxes(tickformat="$,.0f")
    return fig


def chart_gm_pct(
    base: pd.DataFrame,
    scen: pd.DataFrame,
    threshold: float,
) -> go.Figure:
    """Line chart: Gross Margin % trend with threshold line."""
    fig = go.Figure()
    fig.add_hline(
        y=threshold, line_dash="dash",
        line_color=PALETTE["red"], opacity=0.7,
        annotation_text=f"Margin Alert: {threshold:.0%}",
        annotation_position="bottom right",
    )
    fig.add_trace(go.Scatter(
        name="Baseline GM%", x=MONTH_LABELS,
        y=base["gm_pct"], mode="lines+markers",
        line=dict(color=PALETTE["baseline"], width=2.5),
        marker=dict(size=7),
    ))
    fig.add_trace(go.Scatter(
        name="Scenario GM%", x=MONTH_LABELS,
        y=scen["gm_pct"], mode="lines+markers",
        line=dict(color=PALETTE["scenario"], width=2.5),
        marker=dict(size=7),
        fill="tonexty",
        fillcolor="rgba(237,125,49,0.08)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title_text="Gross Margin % — 12-Month Trend",
        height=340,
        yaxis=dict(tickformat=".1%", showgrid=True, gridcolor="#F2F2F2"),
    )
    return fig


def chart_waterfall(
    base_pnl: pd.DataFrame,
    scen_pnl: pd.DataFrame,
) -> go.Figure:
    """
    Waterfall chart showing scenario P&L bridge vs baseline.
    Shows how each cost/revenue driver contributes to GP delta.
    """
    b = base_pnl.sum(numeric_only=True)
    s = scen_pnl.sum(numeric_only=True)

    rev_delta   = s["revenue"]         - b["revenue"]
    mat_delta   = -(s["material_cogs"] - b["material_cogs"])
    lab_delta   = -(s["labor_cogs"]    - b["labor_cogs"])
    oh_delta    = -(s["overhead_cogs"] - b["overhead_cogs"])
    frt_delta   = -(s["freight_cost"]  - b["freight_cost"])
    gp_delta    =   s["gross_profit"]  - b["gross_profit"]

    labels  = ["Baseline GP", "Volume / Mix", "Material Cost Δ",
               "Labor Cost Δ", "Overhead Δ", "Freight Δ", "Scenario GP"]
    measures = ["absolute", "relative", "relative",
                "relative", "relative", "relative", "total"]
    values  = [b["gross_profit"], rev_delta, mat_delta,
               lab_delta, oh_delta, frt_delta, s["gross_profit"]]

    colours = []
    for i, (m, v) in enumerate(zip(measures, values)):
        if m == "absolute" or m == "total":
            colours.append(PALETTE["navy"] if i == 0 else PALETTE["scenario"])
        else:
            colours.append(PALETTE["positive"] if v >= 0 else PALETTE["negative"])

    fig = go.Figure(go.Waterfall(
        name="P&L Bridge",
        orientation="v",
        measure=measures,
        x=labels,
        y=values,
        text=[f"${v:+,.0f}" for v in values],
        textposition="outside",
        connector=dict(line=dict(color=PALETTE["mgray"], width=1, dash="dot")),
        increasing=dict(marker_color=PALETTE["positive"]),
        decreasing=dict(marker_color=PALETTE["negative"]),
        totals=dict(marker_color=PALETTE["scenario"]),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title_text="Gross Profit Bridge: Baseline → Scenario (12-Month Cumulative)",
        height=420,
        yaxis=dict(tickformat="$,.0f", showgrid=True, gridcolor="#F2F2F2"),
    )
    return fig


def chart_capacity_heatmap(loading: pd.DataFrame) -> go.Figure:
    """
    Heatmap of machine utilisation % — month × machine center.
    Red = over 100%, Amber = 85-100%, Green = healthy.
    """
    pivot = loading.pivot_table(
        index="machine_name",
        columns="month_label",
        values="utilisation_pct",
        aggfunc="mean",
    )
    pivot = pivot[[m for m in MONTH_LABELS if m in pivot.columns]]

    colorscale = [
        [0.00, "#00B050"],  # green  < 80%
        [0.80, "#00B050"],
        [0.85, "#FFC000"],  # amber  85%
        [1.00, "#FFC000"],
        [1.05, "#C00000"],  # red    > 100%
        [2.00, "#C00000"],
    ]

    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=list(pivot.columns),
        y=list(pivot.index),
        colorscale=colorscale,
        zmin=0, zmax=1.2,
        text=[[f"{v:.0%}" for v in row] for row in pivot.values],
        texttemplate="%{text}",
        textfont=dict(size=10, color="white"),
        colorbar=dict(
            tickformat=".0%",
            title="Utilisation",
        ),
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title_text="Machine Capacity Utilisation Heatmap",
        height=260,
        xaxis=dict(tickangle=-45),
    )
    return fig


def chart_inventory_projection(
    base_inv: pd.DataFrame,
    scen_inv: pd.DataFrame,
    product_code: str,
) -> go.Figure:
    """
    Area chart: Closing inventory vs safety stock for a selected product.
    """
    b = base_inv[base_inv["product_code"] == product_code].copy()
    s = scen_inv[scen_inv["product_code"] == product_code].copy()

    prod_name = PRODUCTS.get(product_code, product_code)
    safety_stock = b["safety_stock"].iloc[0] if len(b) > 0 else 0

    fig = go.Figure()
    fig.add_hline(
        y=safety_stock, line_dash="dash",
        line_color=PALETTE["red"], opacity=0.8,
        annotation_text="Safety Stock",
        annotation_position="right",
    )
    fig.add_trace(go.Scatter(
        name="Baseline Closing Stock", x=b["month_label"],
        y=b["closing_stock"], mode="lines+markers",
        line=dict(color=PALETTE["baseline"], width=2.5),
        fill="tozeroy", fillcolor="rgba(46,117,182,0.12)",
    ))
    fig.add_trace(go.Scatter(
        name="Scenario Closing Stock", x=s["month_label"],
        y=s["closing_stock"], mode="lines+markers",
        line=dict(color=PALETTE["scenario"], width=2.5),
        fill="tozeroy", fillcolor="rgba(237,125,49,0.12)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title_text=f"Finished Goods Inventory Projection — {prod_name}",
        height=320,
        yaxis_title="Units",
    )
    return fig


def chart_cogs_breakdown(pnl: pd.DataFrame, label: str) -> go.Figure:
    """Stacked bar showing COGS composition vs revenue."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Material", x=pnl["month_label"],
        y=pnl["material_cogs"],
        marker_color=PALETTE["navy"],
    ))
    fig.add_trace(go.Bar(
        name="Labor", x=pnl["month_label"],
        y=pnl["labor_cogs"],
        marker_color=PALETTE["blue"],
    ))
    fig.add_trace(go.Bar(
        name="Overhead", x=pnl["month_label"],
        y=pnl["overhead_cogs"],
        marker_color=PALETTE["lblue"],
    ))
    fig.add_trace(go.Bar(
        name="Freight", x=pnl["month_label"],
        y=pnl["freight_cost"],
        marker_color=PALETTE["orange"],
    ))
    fig.add_trace(go.Scatter(
        name="Revenue", x=pnl["month_label"],
        y=pnl["revenue"], mode="lines+markers",
        line=dict(color=PALETTE["dgreen"], width=3),
        yaxis="y",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title_text=f"COGS Composition vs Revenue — {label}",
        barmode="stack",
        height=360,
        yaxis_tickformat="$,.0f",
    )
    return fig


def chart_working_capital(
    base: pd.DataFrame,
    scen: pd.DataFrame,
) -> go.Figure:
    """Line chart comparing net working capital."""
    base_nwc = base["ar_balance"] - base["ap_balance"]
    scen_nwc = scen["ar_balance"] - scen["ap_balance"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name="Baseline Net WC", x=MONTH_LABELS,
        y=base_nwc, mode="lines+markers",
        line=dict(color=PALETTE["baseline"], width=2.5),
    ))
    fig.add_trace(go.Scatter(
        name="Scenario Net WC", x=MONTH_LABELS,
        y=scen_nwc, mode="lines+markers",
        line=dict(color=PALETTE["scenario"], width=2.5),
        fill="tonexty",
        fillcolor="rgba(237,125,49,0.08)",
    ))
    fig.update_layout(
        **CHART_LAYOUT,
        title_text="Net Working Capital (AR − AP)",
        height=320,
        yaxis_tickformat="$,.0f",
    )
    return fig


def chart_service_level(supply: pd.DataFrame, label: str) -> go.Figure:
    """Grouped bar: service level % by product per month."""
    pivot = supply.pivot_table(
        index="month_label",
        columns="product_name",
        values="service_level",
        aggfunc="mean",
    )
    pivot = pivot.reindex(MONTH_LABELS)

    colors = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"]]
    fig = go.Figure()
    for i, col in enumerate(pivot.columns):
        fig.add_trace(go.Bar(
            name=col, x=pivot.index,
            y=pivot[col],
            marker_color=colors[i % len(colors)],
            opacity=0.85,
        ))
    fig.add_hline(y=0.98, line_dash="dash",
                  line_color=PALETTE["red"], opacity=0.7,
                  annotation_text="98% Target")
    fig.update_layout(
        **CHART_LAYOUT,
        title_text=f"Service Level % by Product — {label}",
        barmode="group",
        height=320,
        yaxis=dict(tickformat=".1%", range=[0.85, 1.02]),
    )
    return fig


# ═════════════════════════════════════════════════════════════════════════════
# MODULE 5 — STREAMLIT UI
# ═════════════════════════════════════════════════════════════════════════════

def format_delta(value: float, fmt: str = ",.0f", prefix: str = "$") -> str:
    """Formats a delta value with sign and currency prefix."""
    sign = "+" if value >= 0 else ""
    if abs(value) >= 1_000_000:
        return f"{sign}{prefix}{value/1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{sign}{prefix}{value/1_000:.1f}K"
    return f"{sign}{prefix}{value:{fmt}}"


def render_metric_card(
    col,
    label: str,
    base_val: float,
    scen_val: float,
    fmt: str = ",.0f",
    prefix: str = "$",
    invert_delta: bool = False,
) -> None:
    """
    Renders a Streamlit metric card with scenario vs baseline comparison.
    invert_delta=True: lower is better (e.g. costs, working capital tied-up).
    """
    delta = scen_val - base_val
    if invert_delta:
        delta = -delta
    with col:
        st.metric(
            label=label,
            value=f"{prefix}{scen_val:>10,.0f}" if prefix == "$" else f"{scen_val:.1%}",
            delta=format_delta(
                scen_val - base_val, fmt=fmt, prefix=prefix
            ),
            delta_color="normal" if not invert_delta else "inverse",
        )


def render_exception_panel(exceptions: list) -> None:
    """
    Renders the exception management panel with severity-coloured alerts.
    """
    if not exceptions:
        st.success("✅  No exceptions detected — all thresholds clear.")
        return

    critical = [e for e in exceptions if e["severity"] == "CRITICAL"]
    warnings  = [e for e in exceptions if e["severity"] == "WARNING"]

    if critical:
        st.markdown(
            f"<div style='background:{PALETTE['red']};color:white;"
            f"padding:8px 14px;border-radius:6px;font-weight:600;"
            f"margin-bottom:8px;'>🚨  {len(critical)} CRITICAL exception(s)</div>",
            unsafe_allow_html=True,
        )
        for e in critical:
            st.error(
                f"**[{e['category']}] {e['month']}** — {e['message']}"
            )

    if warnings:
        st.markdown(
            f"<div style='background:{PALETTE['amber']};color:{PALETTE['navy']};"
            f"padding:8px 14px;border-radius:6px;font-weight:600;"
            f"margin-bottom:8px;'>⚠️  {len(warnings)} WARNING(s)</div>",
            unsafe_allow_html=True,
        )
        for e in warnings:
            st.warning(
                f"**[{e['category']}] {e['month']}** — {e['message']}"
            )


def sidebar_scenario_controls() -> ScenarioParams:
    """
    Renders the scenario control panel in the sidebar.
    Returns a ScenarioParams object with all user-selected values.
    """
    st.sidebar.markdown(
        f"""
        <div style='background:{PALETTE["navy"]};color:white;
        padding:12px 16px;border-radius:8px;margin-bottom:16px;'>
        <h2 style='margin:0;font-size:1.1rem;'>🎛  Scenario Controls</h2>
        <p style='margin:4px 0 0 0;font-size:0.75rem;opacity:0.8;'>
        Adjust levers below — dashboard updates live
        </p></div>
        """,
        unsafe_allow_html=True,
    )

    # ── DEMAND LEVERS ─────────────────────────────────────────────────────
    st.sidebar.markdown("### 📦 Demand Scenario")
    demand_multiplier = st.sidebar.slider(
        "Global Demand Shift",
        min_value=0.60, max_value=1.50,
        value=1.00, step=0.05,
        format="%.2f×",
        help="Applies uniformly to all products and customers.",
    )
    with st.sidebar.expander("OEM-Level Adjustments"):
        oem_a = st.sidebar.slider("OEM-A (Ford) Multiplier", 0.70, 1.40, 1.00, 0.05)
        oem_b = st.sidebar.slider("OEM-B (GM) Multiplier",   0.70, 1.40, 1.00, 0.05)
        oem_c = st.sidebar.slider("OEM-C (Stellantis) Multiplier", 0.70, 1.40, 1.00, 0.05)

    # ── COST LEVERS ───────────────────────────────────────────────────────
    st.sidebar.markdown("### 💰 Cost Scenario")
    steel_delta = st.sidebar.slider(
        "Steel RM Cost Change",
        min_value=-0.30, max_value=0.60,
        value=0.00, step=0.05,
        format="%+.0%",
        help="Applies to RM-001 (HR Steel) and RM-002 (CR Sheet).",
    )
    freight_adder = st.sidebar.number_input(
        "Expedited Freight Cost ($/unit)",
        min_value=0.0, max_value=50.0,
        value=0.0, step=1.0,
        help="Added to COGS for every unit produced (e.g., air-freight premium).",
    )
    overhead_delta = st.sidebar.slider(
        "Overhead Absorption Change",
        min_value=-0.20, max_value=0.30,
        value=0.00, step=0.05,
        format="%+.0%",
    )

    # ── CAPACITY LEVERS ───────────────────────────────────────────────────
    st.sidebar.markdown("### 🏭 Capacity Scenario")
    cap_multiplier = st.sidebar.slider(
        "Capacity Availability Factor",
        min_value=0.60, max_value=1.20,
        value=1.00, step=0.05,
        format="%.2f×",
        help="<1.0 = equipment downtime / maintenance. >1.0 = third shift.",
    )
    overtime_hrs = st.sidebar.number_input(
        "Overtime Hours / Machine / Month",
        min_value=0.0, max_value=80.0,
        value=0.0, step=8.0,
    )

    # ── WORKING CAPITAL LEVERS ────────────────────────────────────────────
    st.sidebar.markdown("### 🏦 Working Capital")
    ar_days = st.sidebar.number_input(
        "AR Collection Days", min_value=15.0, max_value=90.0,
        value=45.0, step=5.0,
    )
    ap_days = st.sidebar.number_input(
        "AP Payment Days", min_value=15.0, max_value=120.0,
        value=60.0, step=5.0,
    )

    # ── EXCEPTION THRESHOLDS ──────────────────────────────────────────────
    st.sidebar.markdown("### 🚨 Alert Thresholds")
    cap_alert = st.sidebar.slider(
        "Capacity Alert Trigger", 0.70, 1.00, 0.95, 0.01, format="%.0%"
    )
    margin_alert = st.sidebar.slider(
        "Margin Alert Trigger", 0.05, 0.40, 0.20, 0.01, format="%.0%"
    )

    # ── SCENARIO PRESETS ──────────────────────────────────────────────────
    st.sidebar.markdown("### ⚡ Quick Scenario Presets")
    preset = st.sidebar.selectbox(
        "Load Preset",
        options=[
            "Custom",
            "📈 Demand Surge +20%",
            "📉 Demand Drop -20%",
            "⛓️  Supply Chain Crunch",
            "🔧 Equipment Downtime -25%",
            "🚢 Freight Crisis +$15/unit",
            "🏗️  Steel Spike +30%",
            "💥 Perfect Storm",
        ],
        index=0,
    )
    # Apply presets (override sliders when preset selected)
    preset_params = {
        "Custom":                    {},
        "📈 Demand Surge +20%":      {"demand_multiplier": 1.20},
        "📉 Demand Drop -20%":       {"demand_multiplier": 0.80},
        "⛓️  Supply Chain Crunch":    {"demand_multiplier": 1.15, "cap_multiplier": 0.85},
        "🔧 Equipment Downtime -25%": {"cap_multiplier": 0.75},
        "🚢 Freight Crisis +$15/unit":{"freight_adder": 15.0},
        "🏗️  Steel Spike +30%":       {"steel_delta": 0.30},
        "💥 Perfect Storm":           {"demand_multiplier": 1.20, "cap_multiplier": 0.80,
                                       "steel_delta": 0.25, "freight_adder": 12.0},
    }
    if preset != "Custom":
        overrides = preset_params.get(preset, {})
        demand_multiplier  = overrides.get("demand_multiplier", demand_multiplier)
        cap_multiplier     = overrides.get("cap_multiplier",    cap_multiplier)
        steel_delta        = overrides.get("steel_delta",       steel_delta)
        freight_adder      = overrides.get("freight_adder",     freight_adder)

    return ScenarioParams(
        demand_multiplier      = demand_multiplier,
        oem_a_multiplier       = oem_a if preset == "Custom" else 1.0,
        oem_b_multiplier       = oem_b if preset == "Custom" else 1.0,
        oem_c_multiplier       = oem_c if preset == "Custom" else 1.0,
        steel_cost_delta_pct   = steel_delta,
        freight_cost_per_unit  = freight_adder,
        overhead_delta_pct     = overhead_delta,
        capacity_multiplier    = cap_multiplier,
        overtime_hrs_month     = overtime_hrs,
        ar_days                = ar_days,
        ap_days                = ap_days,
        capacity_alert_pct     = cap_alert,
        margin_alert_pct       = margin_alert,
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ═════════════════════════════════════════════════════════════════════════════

def main():
    # ── HEADER ───────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style='background:linear-gradient(135deg, {PALETTE["navy"]} 0%,
        {PALETTE["blue"]} 100%);padding:20px 28px;border-radius:10px;
        margin-bottom:20px;'>
        <h1 style='color:white;margin:0;font-size:1.7rem;'>
        🏭  SIOP Digital Twin
        </h1>
        <p style='color:rgba(255,255,255,0.75);margin:6px 0 0 0;font-size:0.9rem;'>
        Enterprise Sales, Inventory & Operations Planning Platform
        · Tier 1 Automotive OEM  · 12-Month Rolling Horizon
        · <strong style="color:{PALETTE["amber"]};">Financial-First Decision Engine</strong>
        </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── LOAD DATA & INITIALISE ENGINE ────────────────────────────────────
    master = generate_master_data()
    engine = SIOPEngine(master)

    # ── SIDEBAR: SCENARIO CONTROLS ────────────────────────────────────────
    scen_params  = sidebar_scenario_controls()
    base_params  = ScenarioParams()    # all defaults = baseline

    # ── SESSION STATE: RUN CALCULATIONS ──────────────────────────────────
    # We cache the baseline in session state so it only recomputes when
    # master data changes. The scenario always recomputes on widget change.
    if "baseline_results" not in st.session_state:
        with st.spinner("Computing baseline plan…"):
            st.session_state["baseline_results"] = engine.run(base_params)

    base_results = st.session_state["baseline_results"]

    with st.spinner("Running scenario plan…"):
        scen_results = engine.run(scen_params)

    b_pnl  = base_results["monthly_pnl"]
    s_pnl  = scen_results["monthly_pnl"]
    b_prod = base_results["product_pnl"]
    s_prod = scen_results["product_pnl"]

    # Exception detection
    scen_exceptions = detect_exceptions(scen_results, scen_params, "Scenario")
    base_exceptions = detect_exceptions(base_results, base_params, "Baseline")

    # ── TABS ──────────────────────────────────────────────────────────────
    tab_overview, tab_financial, tab_capacity, tab_supply, tab_exceptions, tab_data = st.tabs([
        "📊 Overview",
        "💰 Financial Deep-Dive",
        "🏭 Capacity & Supply",
        "📦 Inventory & Service",
        "🚨 Exception Management",
        "🔍 Raw Data",
    ])

    # ══════════════════════════════════════════════════════════════════════
    # TAB 1 — OVERVIEW
    # ══════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.markdown("#### 12-Month Cumulative KPIs — Baseline vs Scenario")

        b_sum = b_pnl.sum(numeric_only=True)
        s_sum = s_pnl.sum(numeric_only=True)
        b_gm  = b_sum["gross_profit"] / b_sum["revenue"]
        s_gm  = s_sum["gross_profit"] / s_sum["revenue"]
        b_sl  = base_results["supply"]["service_level"].mean()
        s_sl  = scen_results["supply"]["service_level"].mean()
        b_cap = base_results["loading"]["utilisation_pct"].max()
        s_cap = scen_results["loading"]["utilisation_pct"].max()

        kpi_cols = st.columns(6)
        render_metric_card(kpi_cols[0], "📈 Revenue",
                           b_sum["revenue"], s_sum["revenue"])
        render_metric_card(kpi_cols[1], "💵 Gross Profit",
                           b_sum["gross_profit"], s_sum["gross_profit"])
        render_metric_card(kpi_cols[2], "📉 Total COGS",
                           b_sum["total_cogs"], s_sum["total_cogs"],
                           invert_delta=True)
        kpi_cols[3].metric(
            "🎯 Gross Margin",
            f"{s_gm:.1%}",
            delta=f"{(s_gm - b_gm)*100:+.1f}pp",
            delta_color="normal",
        )
        kpi_cols[4].metric(
            "🚚 Avg Service Level",
            f"{s_sl:.1%}",
            delta=f"{(s_sl - b_sl)*100:+.1f}pp",
        )
        kpi_cols[5].metric(
            "⚙️ Peak Capacity Util",
            f"{s_cap:.1%}",
            delta=f"{(s_cap - b_cap)*100:+.1f}pp",
            delta_color="inverse",
        )

        # Exception banner at top
        if scen_exceptions:
            crit_count = sum(1 for e in scen_exceptions if e["severity"] == "CRITICAL")
            warn_count = sum(1 for e in scen_exceptions if e["severity"] == "WARNING")
            if crit_count > 0:
                st.error(
                    f"🚨 **{crit_count} CRITICAL** and **{warn_count} WARNING** "
                    f"exceptions detected — see the **Exception Management** tab for details."
                )
            else:
                st.warning(
                    f"⚠️ **{warn_count} WARNING** exception(s) — see Exception Management tab."
                )

        st.divider()

        # Revenue & GP chart
        st.plotly_chart(
            chart_revenue_cogs(b_pnl, s_pnl),
            use_container_width=True,
        )

        col_wf, col_gm = st.columns([3, 2])
        with col_wf:
            st.plotly_chart(
                chart_waterfall(b_pnl, s_pnl),
                use_container_width=True,
            )
        with col_gm:
            st.plotly_chart(
                chart_gm_pct(b_pnl, s_pnl, scen_params.margin_alert_pct),
                use_container_width=True,
            )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 2 — FINANCIAL DEEP-DIVE
    # ══════════════════════════════════════════════════════════════════════
    with tab_financial:
        st.markdown("#### Financial P&L Detail — Baseline vs Scenario")

        col_b_fin, col_s_fin = st.columns(2)
        with col_b_fin:
            st.plotly_chart(
                chart_cogs_breakdown(b_pnl, "Baseline"),
                use_container_width=True,
            )
        with col_s_fin:
            st.plotly_chart(
                chart_cogs_breakdown(s_pnl, "Scenario"),
                use_container_width=True,
            )

        # Product P&L table
        st.markdown("#### Product-Level P&L Summary — Scenario")
        s_prod_disp = s_prod.copy()
        s_prod_disp["Revenue ($)"]     = s_prod_disp["total_revenue"].apply(lambda x: f"${x:,.0f}")
        s_prod_disp["COGS ($)"]        = s_prod_disp["total_cogs"].apply(lambda x: f"${x:,.0f}")
        s_prod_disp["Gross Profit ($)"]= s_prod_disp["total_gp"].apply(lambda x: f"${x:,.0f}")
        s_prod_disp["GM %"]            = s_prod_disp["gm_pct"].apply(lambda x: f"{x:.1%}")
        s_prod_disp["Units Produced"]  = s_prod_disp["total_units"].apply(lambda x: f"{x:,.0f}")
        s_prod_disp["Shortage Units"]  = s_prod_disp["total_shortage"].apply(lambda x: f"{x:,.0f}")
        st.dataframe(
            s_prod_disp[["product_code","product_name",
                          "Revenue ($)","COGS ($)","Gross Profit ($)",
                          "GM %","Units Produced","Shortage Units"]],
            hide_index=True,
            use_container_width=True,
        )

        # Working capital
        st.markdown("#### Working Capital Trend")
        col_wc1, col_wc2 = st.columns([2, 1])
        with col_wc1:
            st.plotly_chart(
                chart_working_capital(b_pnl, s_pnl),
                use_container_width=True,
            )
        with col_wc2:
            st.markdown("##### WC KPIs — Scenario (12-Mo Avg)")
            b_ar  = b_pnl["ar_balance"].mean()
            s_ar  = s_pnl["ar_balance"].mean()
            b_ap  = b_pnl["ap_balance"].mean()
            s_ap  = s_pnl["ap_balance"].mean()
            st.metric("Avg AR Balance",  f"${s_ar:,.0f}",  delta=f"${s_ar-b_ar:+,.0f}")
            st.metric("Avg AP Balance",  f"${s_ap:,.0f}",  delta=f"${s_ap-b_ap:+,.0f}")
            net_b = b_ar - b_ap
            net_s = s_ar - s_ap
            st.metric("Avg Net WC (AR-AP)", f"${net_s:,.0f}", delta=f"${net_s-net_b:+,.0f}",
                      delta_color="inverse")

    # ══════════════════════════════════════════════════════════════════════
    # TAB 3 — CAPACITY & SUPPLY
    # ══════════════════════════════════════════════════════════════════════
    with tab_capacity:
        st.markdown("#### Machine Capacity Loading")

        col_bh, col_sh = st.columns(2)
        with col_bh:
            st.markdown("**Baseline**")
            st.plotly_chart(
                chart_capacity_heatmap(base_results["loading"]),
                use_container_width=True,
            )
        with col_sh:
            st.markdown("**Scenario**")
            st.plotly_chart(
                chart_capacity_heatmap(scen_results["loading"]),
                use_container_width=True,
            )

        st.markdown("#### Bottleneck Analysis")
        # Show worst-utilised machine per month for scenario
        worst = (
            scen_results["loading"]
            .sort_values("utilisation_pct", ascending=False)
            .groupby("month_label")
            .first()
            .reset_index()[["month_label", "machine_name", "utilisation_pct",
                             "required_hrs", "avail_hrs"]]
        )
        worst["month_label"] = pd.Categorical(
            worst["month_label"], categories=MONTH_LABELS, ordered=True
        )
        worst = worst.sort_values("month_label")
        worst["Utilisation"] = worst["utilisation_pct"].apply(lambda x: f"{x:.1%}")
        worst["Required Hrs"] = worst["required_hrs"].apply(lambda x: f"{x:,.1f}")
        worst["Available Hrs"] = worst["avail_hrs"].apply(lambda x: f"{x:,.1f}")
        worst["Surplus / (Gap) Hrs"] = (
            worst["avail_hrs"] - worst["required_hrs"]
        ).apply(lambda x: f"{x:+,.1f}")

        def style_util(val):
            try:
                v = float(val.replace("%",""))/100
                if v > 1.0: return "background-color:#C00000;color:white;font-weight:bold"
                if v > 0.95: return "background-color:#FFC000;color:#1F3864"
                return "background-color:#00B050;color:white"
            except Exception:
                return ""

        styled = worst[["month_label","machine_name","Utilisation",
                         "Required Hrs","Available Hrs","Surplus / (Gap) Hrs"]]
        st.dataframe(
            styled.rename(columns={"month_label":"Month","machine_name":"Bottleneck Machine"}),
            hide_index=True,
            use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 4 — INVENTORY & SERVICE LEVEL
    # ══════════════════════════════════════════════════════════════════════
    with tab_supply:
        st.markdown("#### Finished Goods Inventory Projection")

        selected_product = st.selectbox(
            "Select Product",
            options=list(PRODUCTS.keys()),
            format_func=lambda x: f"{x} — {PRODUCTS[x]}",
        )
        st.plotly_chart(
            chart_inventory_projection(
                base_results["inventory"],
                scen_results["inventory"],
                selected_product,
            ),
            use_container_width=True,
        )

        col_sl_b, col_sl_s = st.columns(2)
        with col_sl_b:
            st.plotly_chart(
                chart_service_level(base_results["supply"], "Baseline"),
                use_container_width=True,
            )
        with col_sl_s:
            st.plotly_chart(
                chart_service_level(scen_results["supply"], "Scenario"),
                use_container_width=True,
            )

        st.markdown("#### Supply Plan Detail — Scenario")
        sp_disp = scen_results["supply"].copy()
        sp_disp["Demand"] = sp_disp["demand_qty"].apply(lambda x: f"{x:,.0f}")
        sp_disp["Constrained Supply"] = sp_disp["constrained_qty"].apply(lambda x: f"{x:,.0f}")
        sp_disp["Shortage"] = sp_disp["shortage_qty"].apply(lambda x: f"{x:,.0f}")
        sp_disp["Service Level"] = sp_disp["service_level"].apply(lambda x: f"{x:.1%}")
        st.dataframe(
            sp_disp[["month_label","product_name","Demand",
                      "Constrained Supply","Shortage","Service Level"]]
            .rename(columns={"month_label":"Month","product_name":"Product"}),
            hide_index=True,
            use_container_width=True,
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAB 5 — EXCEPTION MANAGEMENT
    # ══════════════════════════════════════════════════════════════════════
    with tab_exceptions:
        st.markdown("#### 🚨 Active Exception Log — Scenario Plan")

        col_exc1, col_exc2 = st.columns([2, 1])
        with col_exc1:
            render_exception_panel(scen_exceptions)
        with col_exc2:
            st.markdown("##### Exception Summary")
            categories = ["Capacity", "Margin", "Inventory", "Service Level"]
            counts = {
                cat: sum(1 for e in scen_exceptions if e["category"] == cat)
                for cat in categories
            }
            fig_exc = go.Figure(go.Bar(
                x=list(counts.keys()),
                y=list(counts.values()),
                marker_color=[
                    PALETTE["red"] if v > 0 else PALETTE["green"]
                    for v in counts.values()
                ],
                text=list(counts.values()),
                textposition="outside",
            ))
            fig_exc.update_layout(
                **CHART_LAYOUT,
                title_text="Exceptions by Category",
                height=280,
                yaxis_title="Count",
                showlegend=False,
            )
            st.plotly_chart(fig_exc, use_container_width=True)

        if scen_exceptions:
            st.divider()
            st.markdown("##### Exception Detail Table")
            exc_df = pd.DataFrame(scen_exceptions)[
                ["severity", "category", "month", "machine", "message",
                 "value", "threshold"]
            ]
            exc_df.columns = ["Severity", "Category", "Month",
                               "Asset/Item", "Exception Message",
                               "Actual Value", "Threshold"]
            st.dataframe(exc_df, hide_index=True, use_container_width=True)

        st.divider()
        st.markdown("#### 📋 Baseline Exception Log (for comparison)")
        render_exception_panel(base_exceptions)

    # ══════════════════════════════════════════════════════════════════════
    # TAB 6 — RAW DATA EXPLORER
    # ══════════════════════════════════════════════════════════════════════
    with tab_data:
        st.markdown("#### Master Data Inspector")
        data_choice = st.selectbox(
            "Select dataset",
            options=[
                "Demand Forecast (Raw)",
                "Scenario Demand Plan",
                "Monthly P&L — Baseline",
                "Monthly P&L — Scenario",
                "Capacity Loading — Scenario",
                "Supply Plan — Scenario",
                "Inventory Projection — Scenario",
                "Bill of Materials",
                "Machine Routings",
                "Raw Material Master",
                "Pricing Master",
            ],
        )
        dataset_map = {
            "Demand Forecast (Raw)":         master["demand_forecast"],
            "Scenario Demand Plan":          scen_results["demand"],
            "Monthly P&L — Baseline":        b_pnl,
            "Monthly P&L — Scenario":        s_pnl,
            "Capacity Loading — Scenario":   scen_results["loading"],
            "Supply Plan — Scenario":        scen_results["supply"],
            "Inventory Projection — Scenario": scen_results["inventory"],
            "Bill of Materials":             master["bom"],
            "Machine Routings":              master["routings"],
            "Raw Material Master":           master["raw_materials"],
            "Pricing Master":                master["pricing"],
        }
        df_show = dataset_map[data_choice]
        st.markdown(
            f"**{len(df_show):,} rows × {len(df_show.columns)} columns** — "
            f"use column headers to sort."
        )
        st.dataframe(df_show, use_container_width=True, height=420)

        # Download button
        csv = df_show.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇️ Download {data_choice} as CSV",
            data=csv,
            file_name=f"siop_{data_choice.lower().replace(' ','_')}.csv",
            mime="text/csv",
        )

    # ── FOOTER ────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <hr style='margin-top:32px;border-color:{PALETTE["mgray"]};'>
        <div style='text-align:center;color:{PALETTE["dgray"]};font-size:0.75rem;
        padding-bottom:16px;'>
        SIOP Digital Twin · Tier 1 Automotive OEM ·
        Powered by Streamlit · Plotly · Pandas ·
        Calculation Engine: {len(PRODUCTS)} products ·
        {len(MACHINES)} machine centres · {N_MONTHS}-month horizon ·
        {len(master["bom"])} BOM lines · {len(master["raw_materials"])} RM components
        </div>
        """,
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
