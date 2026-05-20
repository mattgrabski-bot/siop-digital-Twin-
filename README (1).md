# 🏭 SIOP Digital Twin — Enterprise Knowledge Graph

A production-grade **Sales, Inventory & Operations Planning (SIOP) Digital Twin**
built with Streamlit, Pandas, and Plotly. Modeled after enterprise platforms like
**o9 Solutions** and **Kinaxis RapidResponse** — runs entirely on your local machine
or in a GitHub Codespace with zero external dependencies.

---

## 🚀 Quick Start (Local / GitHub Codespaces)

```bash
# 1. Clone or copy files into a directory
# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app opens at **http://localhost:8501** automatically.

---

## 🧩 Architecture

```
app.py
├── generate_synthetic_erp_data()   # Cached ERP master data (Demand / BOM / Cost / Capacity)
├── ScenarioParams (dataclass)      # Immutable snapshot of all sidebar levers
├── DigitalTwinEngine               # Vectorized calculation engine (the "Digital Brain")
│   ├── build_scenario_demand()     # Demand plan with shift applied
│   ├── build_scenario_costs()      # BOM cost recalculation (broad + critical RM shocks)
│   ├── build_capacity_analysis()   # Plant utilisation vs. required hours
│   ├── build_pl_bridge()           # Full 12-month P&L: Baseline → Scenario
│   ├── build_inventory_projection()# Closing inventory & working capital
│   ├── detect_exceptions()         # Automated alert engine
│   └── run()                       # Orchestrates full calculation chain
├── Chart Library (Plotly)
│   ├── chart_demand_comparison()   # Grouped bar: Baseline vs Scenario demand
│   ├── chart_capacity_utilisation()# Area chart with alert band
│   ├── chart_pl_waterfall()        # Gross Profit Bridge waterfall
│   ├── chart_inventory_trend()     # 12-month inventory projection
│   ├── chart_margin_by_sku()       # Horizontal bar: GM% by SKU
│   └── chart_working_capital()     # Working capital ($M) trend
└── Streamlit UI
    ├── render_header()
    ├── render_sidebar() → ScenarioParams
    ├── render_kpi_strip()
    ├── render_exceptions()
    └── render_cost_detail_table()
```

---

## 🎛️ Scenario Levers

| Lever | Range | Models |
|---|---|---|
| Global Demand Shift | -40% to +40% | OEM upside / ramp-down |
| Broad RM Cost Spike | 0–60% | Commodity inflation |
| Critical Component Spike | 0–80% | Chip / lithium / magnet shortage |
| Expedited Air Freight | 0–50% of volume | Supply disruption response |
| Overtime Hours / Month | 0–400 hrs | Capacity augmentation |
| Additional Scrap Rate | 0–5% | Quality excursion / new-part yield |
| ASP Change | -10% to +10% | Pass-through pricing / OEM rebate |
| Margin Alert Threshold | Configurable | Exception trigger |
| Capacity Alert Threshold | Configurable | Exception trigger |

---

## 📊 Dashboard Panels

1. **KPI Strip** — Revenue, Gross Profit, GM%, Peak Utilisation (Baseline vs Scenario)
2. **Exception Management** — Auto-generated alerts with severity & recommended actions
3. **Demand Analysis** — 12-month grouped bar (Baseline vs Scenario)
4. **Capacity Utilisation** — Plant utilisation % with alert band
5. **P&L Waterfall** — Gross Profit Bridge: Volume | Price | RM Cost impacts
6. **Margin by SKU** — Horizontal bar with red highlighting below threshold
7. **Inventory Trend** — 12-month closing stock (Baseline vs Scenario)
8. **Working Capital** — Inventory value ($M) over horizon
9. **Capacity Detail Table** — Month-by-month hours & status
10. **SKU Cost Decomposition** — RM / Labor / Overhead breakdown with delta
11. **BOM Supply Risk Explorer** — Component-level origin & risk classification

---

## 🗂️ Synthetic ERP Data

The app generates a Tier-1 automotive dataset on startup (no file imports needed):

- **5 SKUs**: EV Drive Module, Battery Pack Assembly, Brake Control Module,
  ADAS Camera Unit, Electric Power Steering
- **12-month demand plan** with seasonal indices and noise
- **BOM** with 18 components, 5 countries of origin, Critical/Standard risk flags
- **Standard cost card** with RM / Labor / Overhead breakdown and ASP
- **Inventory projections** with opening/closing stock and days-on-hand
- **Plant capacity** across 3 assembly lines with maintenance schedules
- **Freight rates** for 9 lanes (Ocean / Air / Road) across 5 origin countries
