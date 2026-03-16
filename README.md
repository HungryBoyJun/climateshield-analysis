# ClimateShield — Data Science Pipeline

> **AI Urban Climate Risk Platform for Durham Region, Ontario**
> Ontario Tech University · Brilliant Catalyst Smart Communities Challenge · March 2026 · Track 3: Climate Resilience Urban Ecosystem

---

## Overview

This repository contains the full reproducible data science pipeline backing the **ClimateShield** pitch deck. Every chart, statistic, and KPI in our presentation is derived from this codebase — no numbers are asserted without a traceable model or cited data source.

The pipeline covers:
- **Data acquisition** from official Canadian open data sources (Statistics Canada, ECCC, TRCA, GEI Consultants, FCM)
- **Exploratory data analysis** and feature engineering
- **Four machine learning models** for climate risk prediction
- **10 publication-quality charts** used directly in the pitch deck

---

## Repository Structure

```
climateshield-analysis/
├── run_pipeline.py              # ← Single entry point
├── src/
│   ├── data_loader.py           # Data acquisition (live fetch + offline fallback)
│   ├── eda_feature_engineering.py  # EDA + feature engineering
│   ├── ml_models.py             # ML model training & evaluation
│   └── visualizations.py       # Chart generation
├── data/
│   ├── raw/                     # Raw CSVs from source datasets
│   └── processed/               # Engineered feature sets
└── outputs/
    ├── charts/                  # 10 publication-quality charts (.png)
    └── models/                  # Trained models (.pkl) + results (.json)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/YOUR-ORG/climateshield-analysis.git
cd climateshield-analysis

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn scipy openpyxl

# Run full pipeline (data → EDA → ML → charts)
python run_pipeline.py

# Options
python run_pipeline.py --data-only    # Data + EDA only
python run_pipeline.py --no-charts   # Data + EDA + ML (skip charts)
```

---

## Data Sources

All data is sourced from official Canadian government and peer-reviewed reports.

| Dataset | Source | URL |
|---|---|---|
| Durham Population Projections | Statistics Canada 17-10-0057-01 + Durham Region Official Plan 2023 | https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1710005701 |
| Extreme Heat Days (historical) | ECCC AHCCD Station 6156732 (Oshawa) | https://climate.weather.gc.ca/historical_data/search_historic_data_e.html |
| Heat Day Projections (RCP 8.5) | TRCA Climate Projections for Durham Region, 2024 | https://trca.ca/conservation/watershed-management/climate-change/ |
| Flood Risk Infrastructure | GEI Consultants — Durham Flood Risk Roads Study, 2022 | https://www.durham.ca/en/regional-government/ |
| TRCA Flood Ready Durham | TRCA / NDMP-funded assessment | https://trca.ca/conservation/flood-risk-management/ |
| Insured Climate Losses | IBC Severe Weather Reports 2015–2023 | https://www.ibc.ca/news-insights/news/severe-weather-drove-record-losses |
| Federal Climate Funding | FCM Green Municipal Fund Annual Reports | https://fcm.ca/en/programs/green-municipal-fund/results |
| Infrastructure Canada Funding | Infrastructure Canada Climate Resilience Program | https://www.infrastructure.gc.ca/plan/icp-pic-eng.html |
| Durham Climate Adaptation Plan | Federation of Canadian Municipalities, Dec 2025 | https://fcm.ca |

> **Note on data availability:** The pipeline first attempts to fetch live data from official URLs. When network access is unavailable (CI/CD, sandboxed environments), it falls back to embedded reference datasets reconstructed directly from the cited reports. All embedded values are individually traceable to named source documents.

---

## ML Models

### Model 1 — Extreme Heat Projection (`heat_projection_model.pkl`)
- **Algorithm:** Polynomial Regression (degree=2) + StandardScaler
- **Features:** Year
- **Target:** Extreme heat days (Tmax ≥ 30°C) per year
- **Validation:** Cross-validated against TRCA 2024 RCP 8.5 ensemble checkpoints
- **Output:** Heat day forecasts to 2051 with 90% confidence intervals

### Model 2 — Flood Event Frequency (`flood_frequency_model.pkl`)
- **Algorithm:** Random Forest Regressor (200 estimators)
- **Features:** Heat days, heat days lag-1, insured losses, response time index
- **Target:** Annual flood events in Durham Region
- **Top feature:** Insured losses (46.7% importance)

### Model 3 — Hospitalisation Risk (`hospitalisation_model.pkl`)
- **Algorithm:** Gradient Boosting Regressor (150 estimators)
- **Features:** Heat days, lag-1, lag-2, insured losses, flood events
- **Target:** Heat-related hospitalisations (Ontario)
- **Use case:** Quantifies avoided harm from early warning (1,267 avoided in 2015–2023 backtest)

### Model 4 — Response Time Deterioration (`response_time_model.pkl`)
- **Algorithm:** Ridge Regression (α=1.0)
- **Features:** Heat days, flood events, insured losses
- **Target:** Average emergency response time (hrs)
- **Key finding:** Status quo trajectory reaches 9.2 hrs avg by 2024–2030; ClimateShield scenario reduces to 3.7 hrs

---

## Charts Generated

| File | Description |
|---|---|
| `01_heat_days_historical_projected.png` | ECCC historical + ML forecast + TRCA RCP 8.5 validation |
| `02_population_growth_exposure.png` | StatCan population + flood exposure trajectory overlay |
| `03_infrastructure_vulnerability.png` | Composite vulnerability scores by asset category |
| `04_flood_risk_by_category.png` | % at risk + CAD replacement value by infrastructure type |
| `05_insured_losses_trend.png` | IBC insured losses 2015–2023 + trend + cumulative |
| `06_response_time_deterioration.png` | Status quo vs ClimateShield response time scenario |
| `07_hospitalisation_scenario.png` | Baseline vs avoided harm with early warning system |
| `08_funding_growth.png` | FCM + Infrastructure Canada municipal climate funding growth |
| `09_climate_risk_composite_dashboard.png` | 6-panel composite risk dashboard |
| `10_model_accuracy_summary.png` | ML model R² and RMSE comparison |

---

## KPI Backing — How the Pitch Numbers Were Derived

| KPI (Pitch Deck) | Model / Source |
|---|---|
| 60% faster flood response | Model 4: Ridge regression status quo vs ClimateShield scenario |
| 50K+ residents reached | Population × composite exposure index (Model feature) |
| 85% prediction accuracy | Model 3: Gradient Boosting held-out evaluation target |
| 30% infrastructure risk reduction | Vulnerability score improvement with real-time monitoring |
| $2.4B+ annual flood damage | IBC Severe Weather Reports 2015–2023 (avg) |
| +22 extra heat days by 2050 | Model 1 forecast − 2023 baseline (14 → 36 days) |
| 1,267 hospitalisations avoided | Model 3 scenario analysis (backtest 2015–2023) |

---

## Team

**TEAM CEN-TECH (OntarioTech and Centennial College) — Team 3: Climate Resilience Urban Ecosystem**
Brilliant Catalyst Smart Communities Challenge · March 13-14, 2026

- Zainab Lawa — Software Engineering, 3rd year, Ontario Tech
- Siddharta Shukla — Design Lead, Centennial College
- Parmilla Shams — CFO, Centennial College
- Michael Gomez — Architecture, Centennial College
- Jun De Guzman — Data Analyst, Masters, Ontario Tech

---

## License

MIT License — open for collaboration with municipalities, conservation authorities, and research partners.


## Enhanced analysis files

This version includes a stronger real-data notebook and script that extend the original `ClimateShield_RealData_Modeling.ipynb`:

- `notebooks/ClimateShield_Enhanced_Analysis.ipynb`
- `scripts/run_enhanced_analysis.py`
- `outputs/enhanced/`

The enhanced workflow adds:
- richer temporal and interaction features,
- baseline vs tree vs neural benchmark comparisons,
- infrastructure hotspot ranking,
- exposed-population outlook,
- scenario evidence that maps more directly to the pitch deck.
