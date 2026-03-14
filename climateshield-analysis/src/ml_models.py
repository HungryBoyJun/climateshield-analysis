"""
ml_models.py
============
ClimateShield — Machine Learning Pipeline

Models trained:
  1. ExtremeHeatProjectionModel
     - Linear regression + polynomial features on historical ECCC data
     - Validates against TRCA 2024 RCP 8.5 projections
     - Outputs: heat day forecasts to 2051 with confidence intervals

  2. FloodEventFrequencyModel
     - Random Forest Regressor predicting annual flood events
     - Features: population, heat days, insured losses lag
     - Cross-validated with TimeSeriesSplit

  3. HospitalisationRiskModel
     - Gradient Boosting predicting heat-related hospitalisations
     - Features: heat days, lag features, population growth
     - Used to quantify avoided-harm impact of ClimateShield alerts

  4. ResponseTimeDeteriorationModel
     - Ridge regression — models how avg response time worsens
       as flood events & heat days increase (status quo trajectory)
     - Used to justify KPI: 60% response time improvement

All models:
  - Saved to outputs/models/ as serialised objects
  - Evaluated with appropriate metrics (R², RMSE, MAE)
  - Results saved to outputs/models/model_results.json
"""

import pandas as pd
import numpy as np
import os
import sys
import json
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
import pickle

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all
from eda_feature_engineering import run_eda

MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

results_log = {}


def evaluate_model(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"    R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}")
    return {'r2': round(r2, 4), 'rmse': round(rmse, 4), 'mae': round(mae, 4)}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1: Extreme Heat Projection (Polynomial Regression)
# ─────────────────────────────────────────────────────────────────────────────
def train_heat_projection_model(heat_features: pd.DataFrame) -> dict:
    """
    Polynomial (degree=2) regression on historical heat days vs year.
    Generates projections to 2051 with ±1σ confidence band.
    Validated against TRCA 2024 RCP 8.5 ensemble values.
    """
    print("\n  [Model 1] Extreme Heat Projection")

    df = heat_features.dropna(subset=['extreme_heat_days']).copy()
    X = df[['year']].values
    y = df['extreme_heat_days'].values

    # TimeSeriesSplit CV
    tscv = TimeSeriesSplit(n_splits=4)
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=2, include_bias=False)),
        ('scaler', StandardScaler()),
        ('reg', LinearRegression())
    ])
    cv_scores = cross_val_score(pipe, X, y, cv=tscv, scoring='r2')
    print(f"    CV R² scores: {np.round(cv_scores, 3)}  |  Mean={cv_scores.mean():.3f}")

    pipe.fit(X, y)
    y_pred_train = pipe.predict(X)
    metrics = evaluate_model('heat_projection', y, y_pred_train)
    metrics['cv_r2_mean'] = round(cv_scores.mean(), 4)
    metrics['cv_r2_std']  = round(cv_scores.std(), 4)

    # Project to 2051
    future_years = np.arange(2024, 2052).reshape(-1, 1)
    future_pred  = pipe.predict(future_years)

    # Residual std for confidence interval
    residual_std = np.std(y - y_pred_train)
    future_lower = future_pred - 1.645 * residual_std   # 90% CI
    future_upper = future_pred + 1.645 * residual_std

    # Validate against TRCA 2024 RCP 8.5 checkpoints
    trca_checkpoints = {2025: 21, 2030: 24, 2035: 27, 2040: 31, 2045: 34, 2050: 38}
    validation = []
    fut_df = pd.DataFrame({'year': future_years.flatten(), 'predicted': future_pred,
                            'lower_90': future_lower, 'upper_90': future_upper})
    for yr, trca_val in trca_checkpoints.items():
        model_val = fut_df[fut_df['year'] == yr]['predicted'].values[0]
        within_ci = (future_lower[fut_df['year'] == yr][0] <= trca_val <=
                     future_upper[fut_df['year'] == yr][0])
        validation.append({'year': yr, 'trca_rcp85': trca_val,
                           'model_pred': round(model_val, 1), 'within_90ci': bool(within_ci)})
    val_df = pd.DataFrame(validation)
    print(f"    TRCA validation: {val_df['within_90ci'].sum()}/{len(val_df)} checkpoints within 90% CI")

    # Save
    out_path = os.path.join(MODELS_DIR, 'heat_projection_model.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(pipe, f)

    fut_df.to_csv(os.path.join(MODELS_DIR, 'heat_projections_2024_2051.csv'), index=False)
    val_df.to_csv(os.path.join(MODELS_DIR, 'heat_trca_validation.csv'), index=False)

    results_log['heat_projection'] = {**metrics, 'trca_checkpoints_validated': int(val_df['within_90ci'].sum())}
    print(f"    → Model saved: {out_path}")
    return {'model': pipe, 'projections': fut_df, 'validation': val_df}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2: Flood Event Frequency (Random Forest)
# ─────────────────────────────────────────────────────────────────────────────
def train_flood_frequency_model(impact_features: pd.DataFrame) -> dict:
    """
    Random Forest predicting annual flood events in Durham.
    Features: heat days, heat day lag, insured losses, response time index.
    """
    print("\n  [Model 2] Flood Event Frequency (Random Forest)")

    df = impact_features.dropna(subset=['heat_days_lag1']).copy()

    feature_cols = ['extreme_heat_days', 'heat_days_lag1',
                    'insured_losses_m_cad', 'response_time_index']
    X = df[feature_cols].values
    y = df['flood_events_durham'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=3)
    rf = RandomForestRegressor(n_estimators=200, max_depth=4,
                                random_state=42, min_samples_leaf=2)
    cv_scores = cross_val_score(rf, X_scaled, y, cv=tscv, scoring='r2')
    print(f"    CV R² scores: {np.round(cv_scores, 3)}  |  Mean={cv_scores.mean():.3f}")

    rf.fit(X_scaled, y)
    y_pred = rf.predict(X_scaled)
    metrics = evaluate_model('flood_frequency', y, y_pred)
    metrics['cv_r2_mean'] = round(cv_scores.mean(), 4)

    # Feature importances
    importances = pd.DataFrame({
        'feature': feature_cols,
        'importance': np.round(rf.feature_importances_, 4)
    }).sort_values('importance', ascending=False)
    print(f"    Feature importances:\n{importances.to_string(index=False)}")

    # Save
    model_bundle = {'model': rf, 'scaler': scaler, 'features': feature_cols}
    with open(os.path.join(MODELS_DIR, 'flood_frequency_model.pkl'), 'wb') as f:
        pickle.dump(model_bundle, f)
    importances.to_csv(os.path.join(MODELS_DIR, 'flood_feature_importances.csv'), index=False)

    results_log['flood_frequency'] = {**metrics, 'top_feature': importances.iloc[0]['feature']}
    return {'model': rf, 'scaler': scaler, 'importances': importances}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 3: Hospitalisation Risk (Gradient Boosting)
# ─────────────────────────────────────────────────────────────────────────────
def train_hospitalisation_model(impact_features: pd.DataFrame,
                                 pop_risk: pd.DataFrame) -> dict:
    """
    Gradient Boosting predicting heat-related hospitalisations in Ontario.
    Used to quantify avoided harm via early warning (KPI: 50K+ residents reached).
    """
    print("\n  [Model 3] Hospitalisation Risk (Gradient Boosting)")

    df = impact_features.dropna(subset=['heat_days_lag1']).copy()

    feature_cols = ['extreme_heat_days', 'heat_days_lag1', 'heat_days_lag2',
                    'insured_losses_m_cad', 'flood_events_durham']
    df_clean = df.dropna(subset=feature_cols)
    X = df_clean[feature_cols].values
    y = df_clean['heat_hospitalizations_ontario'].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=3)
    gb = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                    max_depth=3, random_state=42,
                                    subsample=0.85)
    cv_scores = cross_val_score(gb, X_scaled, y, cv=tscv, scoring='r2')
    print(f"    CV R² scores: {np.round(cv_scores, 3)}  |  Mean={cv_scores.mean():.3f}")

    gb.fit(X_scaled, y)
    y_pred = gb.predict(X_scaled)
    metrics = evaluate_model('hospitalisation', y, y_pred)

    # Scenario analysis: with vs without early warning
    # Assumption: 30% reduction in high-heat-day hospitalisations with alerts
    df_clean = df_clean.copy()
    df_clean['predicted_hosp_baseline'] = y_pred
    df_clean['predicted_hosp_with_alerts'] = y_pred * 0.70   # 30% reduction
    df_clean['avoided_hospitalisations']  = (y_pred - y_pred * 0.70).round(0)
    scenario_path = os.path.join(MODELS_DIR, 'hospitalisation_scenario.csv')
    df_clean[['year', 'extreme_heat_days', 'predicted_hosp_baseline',
              'predicted_hosp_with_alerts', 'avoided_hospitalisations']].to_csv(scenario_path, index=False)
    total_avoided = df_clean['avoided_hospitalisations'].sum()
    print(f"    Scenario: {total_avoided:.0f} total hospitalisations avoided (2015-2023 backtest)")

    with open(os.path.join(MODELS_DIR, 'hospitalisation_model.pkl'), 'wb') as f:
        pickle.dump({'model': gb, 'scaler': scaler, 'features': feature_cols}, f)

    results_log['hospitalisation'] = {**metrics, 'avoided_hospitalisations_backtest': int(total_avoided)}
    return {'model': gb, 'scaler': scaler, 'scenario': df_clean}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 4: Response Time Deterioration (Ridge Regression)
# ─────────────────────────────────────────────────────────────────────────────
def train_response_time_model(impact_features: pd.DataFrame) -> dict:
    """
    Ridge regression: response time as function of flood events + heat days.
    Status quo trajectory shows continued deterioration.
    ClimateShield KPI target: cut response time by 60% (7.8 hrs → <90 min).
    """
    print("\n  [Model 4] Response Time Deterioration (Ridge)")

    df = impact_features.dropna(subset=['extreme_heat_days']).copy()

    feature_cols = ['extreme_heat_days', 'flood_events_durham', 'insured_losses_m_cad']
    X = df[feature_cols].values
    y = df['avg_response_time_hrs'].values

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge',  Ridge(alpha=1.0))
    ])
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = cross_val_score(pipe, X, y, cv=tscv, scoring='r2')
    print(f"    CV R² scores: {np.round(cv_scores, 3)}  |  Mean={cv_scores.mean():.3f}")

    pipe.fit(X, y)
    y_pred = pipe.predict(X)
    metrics = evaluate_model('response_time', y, y_pred)

    # Status quo projection to 2030 (no ClimateShield)
    future_scenarios = pd.DataFrame({
        'year': range(2024, 2031),
        'extreme_heat_days': [22, 23, 24, 25, 26, 27, 28],
        'flood_events_durham': [9, 10, 10, 11, 11, 12, 13],
        'insured_losses_m_cad': [3800, 4200, 4600, 5100, 5600, 6200, 6900],
    })
    X_future = future_scenarios[feature_cols].values
    future_scenarios['projected_response_time_hrs_status_quo'] = pipe.predict(X_future)
    future_scenarios['projected_response_time_hrs_with_climateshield'] = (
        future_scenarios['projected_response_time_hrs_status_quo'] * 0.40  # 60% reduction
    )
    future_scenarios.to_csv(os.path.join(MODELS_DIR, 'response_time_scenarios.csv'), index=False)

    avg_sq  = future_scenarios['projected_response_time_hrs_status_quo'].mean()
    avg_cs  = future_scenarios['projected_response_time_hrs_with_climateshield'].mean()
    print(f"    Status quo 2024-2030 avg: {avg_sq:.2f} hrs | With ClimateShield: {avg_cs:.2f} hrs")

    with open(os.path.join(MODELS_DIR, 'response_time_model.pkl'), 'wb') as f:
        pickle.dump(pipe, f)

    results_log['response_time'] = {
        **metrics,
        'status_quo_avg_hrs': round(avg_sq, 2),
        'climateshield_avg_hrs': round(avg_cs, 2),
        'reduction_pct': round((1 - avg_cs / avg_sq) * 100, 1)
    }
    return {'model': pipe, 'scenarios': future_scenarios}


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_all_models(eda_outputs: dict) -> dict:
    print("\n🤖 Machine Learning Pipeline")
    print("=" * 50)

    model_outputs = {
        'heat':          train_heat_projection_model(eda_outputs['heat_features']),
        'flood':         train_flood_frequency_model(eda_outputs['impact_features']),
        'hospitalisation': train_hospitalisation_model(eda_outputs['impact_features'],
                                                        eda_outputs['pop_risk']),
        'response_time': train_response_time_model(eda_outputs['impact_features']),
    }

    # Save results log
    results_path = os.path.join(MODELS_DIR, 'model_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_log, f, indent=2)

    print(f"\n✅ All models trained. Results saved to outputs/models/model_results.json")
    print("\n📊 Model Summary:")
    for name, res in results_log.items():
        print(f"   {name:25s}  R²={res['r2']:.3f}  RMSE={res['rmse']:.3f}")

    return model_outputs


if __name__ == '__main__':
    datasets = load_all()
    eda_outputs = run_eda(datasets)
    run_all_models(eda_outputs)
