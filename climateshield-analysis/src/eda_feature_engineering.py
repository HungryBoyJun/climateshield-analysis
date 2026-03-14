"""
eda_feature_engineering.py
==========================
ClimateShield — Exploratory Data Analysis & Feature Engineering

Generates:
  - Descriptive statistics for all datasets
  - Correlation analysis between climate variables
  - Engineered features for ML model inputs
  - Processed datasets saved to data/processed/

Features engineered:
  - 5-year rolling mean of extreme heat days
  - Population-weighted flood exposure index
  - Climate risk composite score per year
  - Decade-over-decade warming acceleration
  - Infrastructure vulnerability score
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def engineer_heat_features(heat_df: pd.DataFrame, temp_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer time-series features for heat risk modelling.
    """
    hist = heat_df[heat_df['data_type'] == 'historical'].copy()
    merged = hist.merge(temp_df, on='year', how='left')

    # Rolling statistics
    merged['heat_rolling_5yr_mean'] = (
        merged['extreme_heat_days'].rolling(5, min_periods=3).mean()
    )
    merged['heat_rolling_5yr_std'] = (
        merged['extreme_heat_days'].rolling(5, min_periods=3).std()
    )

    # Year-over-year delta
    merged['heat_yoy_delta'] = merged['extreme_heat_days'].diff()

    # Decade label
    merged['decade'] = (merged['year'] // 10) * 10

    # Warming anomaly relative to 1990-2000 baseline
    baseline_temp = merged[merged['year'].between(1990, 2000)]['mean_temp_c'].mean()
    merged['temp_anomaly'] = merged['mean_temp_c'] - baseline_temp

    # Compound risk flag: heat days > 15 AND temp anomaly > 0.5
    merged['compound_risk_flag'] = (
        (merged['extreme_heat_days'] > 15) & (merged['temp_anomaly'] > 0.5)
    ).astype(int)

    # Normalised heat index (0-1)
    merged['heat_index_norm'] = (
        merged['extreme_heat_days'] / merged['extreme_heat_days'].max()
    )

    out_path = os.path.join(PROCESSED_DIR, 'heat_features.csv')
    merged.to_csv(out_path, index=False)
    print(f"  [EDA] Heat features engineered — {len(merged.columns)} features, saved to {out_path}")
    return merged


def engineer_population_risk_features(pop_df: pd.DataFrame,
                                       flood_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute population-weighted flood exposure index.
    Combines projected population growth with infrastructure flood risk.
    """
    # Weighted exposure index: sum(pct_at_risk * replacement_value) / total_value
    total_value = flood_df['estimated_replacement_value_m_cad'].sum()
    flood_df = flood_df.copy()
    flood_df['exposure_weight'] = (
        (flood_df['pct_at_risk'] / 100) *
        (flood_df['estimated_replacement_value_m_cad'] / max(total_value, 1))
    )
    composite_exposure = flood_df['exposure_weight'].sum()

    # Apply to population trajectory
    pop_df = pop_df.copy()
    pop_df['exposed_population_est'] = (
        pop_df['population'] * composite_exposure
    ).astype(int)
    pop_df['flood_risk_composite'] = composite_exposure
    pop_df['pop_growth_rate'] = pop_df['population'].pct_change() * 100

    out_path = os.path.join(PROCESSED_DIR, 'population_risk_features.csv')
    pop_df.to_csv(out_path, index=False)
    print(f"  [EDA] Population risk features engineered — composite exposure index: {composite_exposure:.3f}")
    return pop_df


def engineer_climate_impact_features(impact_df: pd.DataFrame,
                                      heat_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge climate impact data with heat observations.
    Engineer lag features for predictive modelling.
    """
    heat_hist = heat_df[heat_df['data_type'] == 'historical'][['year', 'extreme_heat_days']].copy()
    merged = impact_df.merge(heat_hist, on='year', how='left')

    # Lag features (prior year heat days → current year hospitalizations)
    merged['heat_days_lag1'] = merged['extreme_heat_days'].shift(1)
    merged['heat_days_lag2'] = merged['extreme_heat_days'].shift(2)

    # Hospitalisation rate per heat day
    merged['hosp_per_heat_day'] = (
        merged['heat_hospitalizations_ontario'] /
        merged['extreme_heat_days'].replace(0, np.nan)
    )

    # Losses per flood event
    merged['loss_per_flood_event'] = (
        merged['insured_losses_m_cad'] /
        merged['flood_events_durham'].replace(0, np.nan)
    )

    # Cumulative loss trajectory
    merged['cumulative_losses_m'] = merged['insured_losses_m_cad'].cumsum()

    # Response time deterioration index (higher = worse)
    baseline_rt = merged['avg_response_time_hrs'].iloc[0]
    merged['response_time_index'] = merged['avg_response_time_hrs'] / baseline_rt

    out_path = os.path.join(PROCESSED_DIR, 'impact_features.csv')
    merged.to_csv(out_path, index=False)
    print(f"  [EDA] Impact features engineered — {len(merged.columns)} features")
    return merged


def engineer_infrastructure_vulnerability(flood_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute composite vulnerability score per infrastructure category.
    Score = (pct_at_risk / 100) * value_weight * monitoring_penalty
    """
    df = flood_df.copy()
    total_val = df['estimated_replacement_value_m_cad'].replace(0, 1).sum()
    df['value_weight'] = df['estimated_replacement_value_m_cad'].replace(0, 1) / total_val

    # Monitoring penalty: None=1.5x, Partial=1.2x, Full=1.0x
    monitoring_map = {'None': 1.5, 'Partial': 1.2, 'Full': 1.0}
    df['monitoring_penalty'] = df['current_monitoring'].map(monitoring_map)

    df['vulnerability_score'] = (
        (df['pct_at_risk'] / 100) *
        df['value_weight'] *
        df['monitoring_penalty']
    )
    # Normalise to 0-100
    df['vulnerability_score_norm'] = (
        df['vulnerability_score'] / df['vulnerability_score'].max() * 100
    ).round(1)

    # Priority tier
    df['priority_tier'] = pd.cut(
        df['vulnerability_score_norm'],
        bins=[0, 33, 66, 100],
        labels=['Low', 'Medium', 'High']
    )

    out_path = os.path.join(PROCESSED_DIR, 'infrastructure_vulnerability.csv')
    df.to_csv(out_path, index=False)
    print(f"  [EDA] Infrastructure vulnerability scores computed")
    return df


def descriptive_stats(datasets: dict) -> pd.DataFrame:
    """
    Summary statistics across all datasets.
    """
    summary_rows = []
    for name, df in datasets.items():
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary_rows.append({
                'dataset': name,
                'variable': col,
                'n': df[col].count(),
                'mean': round(df[col].mean(), 3),
                'std':  round(df[col].std(), 3),
                'min':  round(df[col].min(), 3),
                'p25':  round(df[col].quantile(0.25), 3),
                'median': round(df[col].median(), 3),
                'p75':  round(df[col].quantile(0.75), 3),
                'max':  round(df[col].max(), 3),
            })
    summary = pd.DataFrame(summary_rows)
    out_path = os.path.join(PROCESSED_DIR, 'descriptive_statistics.csv')
    summary.to_csv(out_path, index=False)
    print(f"  [EDA] Descriptive statistics computed — {len(summary)} variable summaries")
    return summary


def run_eda(datasets: dict) -> dict:
    print("\n🔬 EDA & Feature Engineering")
    print("=" * 50)

    heat_features      = engineer_heat_features(datasets['heat'], datasets['temperature'])
    pop_risk_features  = engineer_population_risk_features(datasets['population'], datasets['flood_risk'])
    impact_features    = engineer_climate_impact_features(datasets['impacts'], datasets['heat'])
    infra_vuln         = engineer_infrastructure_vulnerability(datasets['flood_risk'])
    stats              = descriptive_stats(datasets)

    print(f"\n✅ Feature engineering complete. Processed files saved to data/processed/")

    return {
        'heat_features':     heat_features,
        'pop_risk':          pop_risk_features,
        'impact_features':   impact_features,
        'infra_vuln':        infra_vuln,
        'descriptive_stats': stats,
    }


if __name__ == '__main__':
    datasets = load_all()
    run_eda(datasets)
