"""
visualizations.py
=================
ClimateShield — Chart Generation Pipeline

Generates all charts used in the pitch deck, backed by real data analysis.
Output: high-resolution PNGs in outputs/charts/

Charts produced:
  01_heat_days_historical_projected.png   — ECCC + TRCA projection with ML forecast
  02_population_growth_exposure.png       — StatCan population + flood exposure overlay
  03_infrastructure_vulnerability.png     — Ranked bar chart with priority tiers
  04_flood_risk_by_category.png          — Horizontal bar with replacement values
  05_insured_losses_trend.png            — IBC data + trend line
  06_response_time_deterioration.png     — Status quo vs ClimateShield scenarios
  07_hospitalisation_scenario.png        — Avoided harm analysis
  08_funding_growth.png                  — FCM + Infrastructure Canada growth
  09_climate_risk_composite.png          — Multi-variable composite dashboard
  10_model_accuracy_summary.png          — ML model performance comparison
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
import os
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_all
from eda_feature_engineering import run_eda
from ml_models import run_all_models

CHARTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'charts')
os.makedirs(CHARTS_DIR, exist_ok=True)

# ─── Design System ────────────────────────────────────────────────────────────
DARK_BG    = '#0A1628'
CARD_BG    = '#0F2844'
TEAL       = '#0D9488'
TEAL_LIGHT = '#14B8A6'
MINT       = '#5EEAD4'
WHITE      = '#FFFFFF'
OFF_WHITE  = '#E2E8F0'
MID_GRAY   = '#64748B'
AMBER      = '#F59E0B'
RED        = '#EF4444'
BLUE       = '#3B82F6'

plt.rcParams.update({
    'figure.facecolor':  DARK_BG,
    'axes.facecolor':    CARD_BG,
    'axes.edgecolor':    '#1A3A5C',
    'axes.labelcolor':   OFF_WHITE,
    'axes.titlecolor':   WHITE,
    'xtick.color':       MID_GRAY,
    'ytick.color':       MID_GRAY,
    'text.color':        OFF_WHITE,
    'grid.color':        '#1A3A5C',
    'grid.alpha':        0.6,
    'font.family':       'DejaVu Sans',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.spines.left':  False,
    'axes.spines.bottom': False,
})

def save_chart(fig, name, dpi=180):
    path = os.path.join(CHARTS_DIR, name)
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor=DARK_BG)
    plt.close(fig)
    print(f"  [chart] Saved: {name}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 01 — Heat Days: Historical + ML Projection + TRCA Validation
# ─────────────────────────────────────────────────────────────────────────────
def chart_heat_projection(heat_df, model_outputs):
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)

    hist = heat_df[heat_df['data_type'] == 'historical']
    proj_trca = heat_df[heat_df['data_type'] == 'projected_RCP85']
    ml_proj   = model_outputs['heat']['projections']
    val_df    = model_outputs['heat']['validation']

    # Historical bars — color-coded by decade
    decade_colors = {1990: TEAL, 2000: TEAL_LIGHT, 2010: AMBER, 2020: RED}
    for _, row in hist.iterrows():
        decade = (int(row['year']) // 10) * 10
        color  = decade_colors.get(decade, TEAL)
        ax.bar(row['year'], row['extreme_heat_days'], color=color, alpha=0.85, width=0.8, zorder=3)

    # ML projection line + CI band
    ax.plot(ml_proj['year'], ml_proj['predicted'], color=MINT, lw=2.5,
            linestyle='--', label='ML Projection (Poly Reg)', zorder=5)
    ax.fill_between(ml_proj['year'], ml_proj['lower_90'], ml_proj['upper_90'],
                    color=MINT, alpha=0.12, label='90% Confidence Interval')

    # TRCA RCP 8.5 validation points
    ax.scatter(proj_trca['year'], proj_trca['extreme_heat_days'],
               color=AMBER, s=80, zorder=6, marker='D', label='TRCA 2024 RCP 8.5')

    # Annotations
    ax.axvline(2023.5, color=MID_GRAY, lw=1, linestyle=':', alpha=0.7)
    ax.text(2016, 35, 'Historical\n(ECCC)', color=MID_GRAY, fontsize=9, ha='center')
    ax.text(2037, 35, 'Projected →', color=MINT, fontsize=9, ha='center')

    # Reference threshold
    ax.axhline(30, color=RED, lw=1, linestyle='--', alpha=0.5)
    ax.text(1991, 31, 'Risk threshold: 30 days', color=RED, fontsize=8, alpha=0.8)

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Extreme Heat Days (Tmax ≥ 30°C)', fontsize=11)
    ax.set_title('Extreme Heat Days — Durham Region\nECCC Historical Data + ML Projection + TRCA 2024 Validation',
                 fontsize=13, fontweight='bold', color=WHITE, pad=15)
    ax.legend(loc='upper left', framealpha=0.2, labelcolor=OFF_WHITE, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.set_xlim(1988, 2053)
    ax.set_ylim(0, 46)

    # Source note
    fig.text(0.01, 0.01, 'Sources: ECCC AHCCD Station 6156732 (Oshawa) | TRCA Climate Projections 2024 (RCP 8.5)',
             fontsize=7.5, color=MID_GRAY, style='italic')

    # Legend patches for decade colors
    patches = [mpatches.Patch(color=c, label=f'{d}s') for d, c in decade_colors.items()]
    ax.legend(handles=patches + ax.get_legend_handles_labels()[0],
              labels=[f'{d}s' for d in decade_colors] + ax.get_legend_handles_labels()[1],
              loc='upper left', framealpha=0.15, labelcolor=OFF_WHITE, fontsize=8.5, ncol=2)

    save_chart(fig, '01_heat_days_historical_projected.png')


# ─────────────────────────────────────────────────────────────────────────────
# 02 — Population Growth & Flood Exposure
# ─────────────────────────────────────────────────────────────────────────────
def chart_population_exposure(pop_risk):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    fig.patch.set_facecolor(DARK_BG)
    ax2 = ax1.twinx()

    hist = pop_risk[~pop_risk['is_projection']]
    proj = pop_risk[pop_risk['is_projection']]

    # Population bars
    ax1.bar(hist['year'], hist['population'] / 1e6, color=TEAL, alpha=0.85, width=3.5, label='Census Population')
    ax1.bar(proj['year'], proj['population'] / 1e6, color=TEAL, alpha=0.4, width=3.5,
            label='Projected Population', hatch='///')

    # Exposed population overlay
    ax2.plot(pop_risk['year'], pop_risk['exposed_population_est'] / 1e3,
             color=AMBER, lw=2.5, marker='o', markersize=5, label='Flood-Exposed Pop. (000s)', zorder=5)
    ax2.fill_between(pop_risk['year'], 0, pop_risk['exposed_population_est'] / 1e3,
                     color=AMBER, alpha=0.10)

    # 1.3M annotation
    ax1.annotate('1.3M by 2051\n(Durham Official Plan)',
                  xy=(2051, 1.3), xytext=(2038, 1.15),
                  arrowprops=dict(arrowstyle='->', color=MINT, lw=1.5),
                  color=MINT, fontsize=9, fontweight='bold')

    ax1.set_xlabel('Year', fontsize=11)
    ax1.set_ylabel('Population (millions)', fontsize=11, color=TEAL_LIGHT)
    ax2.set_ylabel('Flood-Exposed Population (thousands)', fontsize=11, color=AMBER)
    ax2.tick_params(colors=AMBER)
    ax1.set_title('Durham Region Population Growth & Flood Exposure Trajectory\nStatistics Canada Projections + GEI/TRCA Flood Risk Assessment',
                  fontsize=13, fontweight='bold', color=WHITE, pad=15)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               framealpha=0.15, labelcolor=OFF_WHITE, fontsize=9)

    ax1.axvline(2021.5, color=MID_GRAY, lw=1, linestyle=':', alpha=0.6)
    ax1.text(2011, 1.28, '← Census', color=MID_GRAY, fontsize=8.5)
    ax1.text(2023, 1.28, 'Projected →', color=TEAL_LIGHT, fontsize=8.5)
    ax1.grid(axis='y', alpha=0.25)

    fig.text(0.01, 0.01, 'Sources: Statistics Canada 17-10-0057-01 | Durham Region Official Plan 2023 | GEI Consultants 2022',
             fontsize=7.5, color=MID_GRAY, style='italic')
    save_chart(fig, '02_population_growth_exposure.png')


# ─────────────────────────────────────────────────────────────────────────────
# 03 — Infrastructure Vulnerability Scores
# ─────────────────────────────────────────────────────────────────────────────
def chart_infrastructure_vulnerability(infra_vuln):
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor(DARK_BG)

    df = infra_vuln.sort_values('vulnerability_score_norm', ascending=True)

    tier_colors = {'High': RED, 'Medium': AMBER, 'Low': TEAL}
    colors = [tier_colors.get(str(t), TEAL) for t in df['priority_tier']]

    bars = ax.barh(df['infrastructure_category'], df['vulnerability_score_norm'],
                   color=colors, alpha=0.88, height=0.65)

    # Value labels
    for bar, val in zip(bars, df['vulnerability_score_norm']):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f'{val:.0f}', va='center', ha='left', fontsize=10,
                color=WHITE, fontweight='bold')

    # % at risk secondary labels
    for i, (_, row) in enumerate(df.iterrows()):
        ax.text(1, i, f"  {row['pct_at_risk']}% at risk",
                va='center', ha='left', fontsize=8.5, color=MID_GRAY)

    # Legend
    patches = [mpatches.Patch(color=c, label=f'{t} Priority') for t, c in tier_colors.items()]
    ax.legend(handles=patches, loc='lower right', framealpha=0.15,
              labelcolor=OFF_WHITE, fontsize=9)

    ax.set_xlabel('Vulnerability Score (0–100)', fontsize=11)
    ax.set_title('Durham Infrastructure Vulnerability Score\nComposite: Flood Risk % × Asset Value × Monitoring Gap',
                 fontsize=13, fontweight='bold', color=WHITE, pad=15)
    ax.set_xlim(0, 115)
    ax.grid(axis='x', alpha=0.25)
    ax.axvline(66, color=AMBER, lw=1, linestyle='--', alpha=0.5)
    ax.axvline(33, color=TEAL, lw=1, linestyle='--', alpha=0.5)

    fig.text(0.01, 0.01, 'Sources: GEI Consultants 2022 | TRCA Flood Ready Durham | ClimateShield vulnerability model',
             fontsize=7.5, color=MID_GRAY, style='italic')
    save_chart(fig, '03_infrastructure_vulnerability.png')


# ─────────────────────────────────────────────────────────────────────────────
# 04 — Flood Risk by Category (% at risk + replacement value)
# ─────────────────────────────────────────────────────────────────────────────
def chart_flood_risk_category(flood_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 6))
    fig.patch.set_facecolor(DARK_BG)

    df = flood_df.sort_values('pct_at_risk', ascending=True)

    # Left: % at risk
    bar_colors = [RED if p >= 60 else AMBER if p >= 40 else BLUE for p in df['pct_at_risk']]
    ax1.barh(df['infrastructure_category'], df['pct_at_risk'],
             color=bar_colors, alpha=0.88, height=0.65)
    for i, val in enumerate(df['pct_at_risk']):
        ax1.text(val + 0.5, i, f'{val}%', va='center', ha='left', fontsize=10,
                 color=WHITE, fontweight='bold')
    ax1.set_xlabel('% Infrastructure at Flood Risk', fontsize=11)
    ax1.set_title('Flood Risk by Category', fontsize=12, fontweight='bold', color=WHITE)
    ax1.set_xlim(0, 105)
    ax1.grid(axis='x', alpha=0.25)

    # Right: replacement value
    df2 = flood_df[flood_df['estimated_replacement_value_m_cad'] > 0].sort_values(
        'estimated_replacement_value_m_cad', ascending=True)
    ax2.barh(df2['infrastructure_category'],
             df2['estimated_replacement_value_m_cad'] / 1000,
             color=TEAL, alpha=0.85, height=0.65)
    for i, val in enumerate(df2['estimated_replacement_value_m_cad'] / 1000):
        ax2.text(val + 0.05, i, f'${val:.1f}B', va='center', ha='left', fontsize=10,
                 color=WHITE, fontweight='bold')
    ax2.set_xlabel('Estimated Replacement Value (CAD Billions)', fontsize=11)
    ax2.set_title('Asset Value at Risk', fontsize=12, fontweight='bold', color=WHITE)
    ax2.set_xlim(0, 11)
    ax2.grid(axis='x', alpha=0.25)

    fig.suptitle('Durham Region Infrastructure Flood Risk Assessment',
                 fontsize=14, fontweight='bold', color=WHITE, y=1.02)
    fig.text(0.01, -0.03, 'Sources: GEI Consultants 2022 | TRCA Flood Ready Durham (NDMP-funded)',
             fontsize=7.5, color=MID_GRAY, style='italic')
    plt.tight_layout()
    save_chart(fig, '04_flood_risk_by_category.png')


# ─────────────────────────────────────────────────────────────────────────────
# 05 — Insured Losses Trend
# ─────────────────────────────────────────────────────────────────────────────
def chart_insured_losses(impact_df):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    ax.bar(impact_df['year'], impact_df['insured_losses_m_cad'] / 1000,
           color=RED, alpha=0.75, width=0.7, label='Insured Losses (CAD B)')

    # Trend line
    z = np.polyfit(impact_df['year'], impact_df['insured_losses_m_cad'] / 1000, 1)
    p = np.poly1d(z)
    years_ext = np.linspace(impact_df['year'].min(), impact_df['year'].max() + 1, 100)
    ax.plot(years_ext, p(years_ext), color=AMBER, lw=2.5, linestyle='--',
            label=f'Trend (+${z[0]*1000:.0f}M/yr)')

    # Cumulative overlay
    ax2 = ax.twinx()
    ax2.plot(impact_df['year'], impact_df['insured_losses_m_cad'].cumsum() / 1000,
             color=MINT, lw=2, marker='o', markersize=5, label='Cumulative Losses')
    ax2.set_ylabel('Cumulative Losses (CAD B)', color=MINT, fontsize=10)
    ax2.tick_params(colors=MINT)

    total = impact_df['insured_losses_m_cad'].sum() / 1000
    ax.text(2022.5, 3.6, f'Total 2015–2023:\n${total:.1f}B', color=WHITE,
            fontsize=10, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD_BG, edgecolor=RED, alpha=0.8))

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Annual Insured Losses (CAD Billions)', fontsize=11)
    ax.set_title('Ontario Climate-Related Insured Losses 2015–2023\nIBC Severe Weather Reports + Public Safety Canada DFAA',
                 fontsize=13, fontweight='bold', color=WHITE, pad=15)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              framealpha=0.15, labelcolor=OFF_WHITE, fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    fig.text(0.01, 0.01, 'Sources: IBC Severe Weather Reports 2015-2023 | Public Safety Canada DFAA',
             fontsize=7.5, color=MID_GRAY, style='italic')
    save_chart(fig, '05_insured_losses_trend.png')


# ─────────────────────────────────────────────────────────────────────────────
# 06 — Response Time: Status Quo vs ClimateShield
# ─────────────────────────────────────────────────────────────────────────────
def chart_response_time(model_outputs, impact_df):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    scenarios = model_outputs['response_time']['scenarios']

    # Historical
    ax.plot(impact_df['year'], impact_df['avg_response_time_hrs'],
            color=OFF_WHITE, lw=2, marker='o', markersize=5, label='Historical (Actual)')

    # Status quo projection
    ax.plot(scenarios['year'], scenarios['projected_response_time_hrs_status_quo'],
            color=RED, lw=2.5, linestyle='--', marker='s', markersize=5,
            label='Status Quo Projection (no platform)')

    # ClimateShield projection
    ax.plot(scenarios['year'], scenarios['projected_response_time_hrs_with_climateshield'],
            color=MINT, lw=2.5, marker='^', markersize=6,
            label='With ClimateShield (60% reduction KPI)')

    # Target line
    ax.axhline(1.5, color=TEAL, lw=1.5, linestyle=':', alpha=0.8)
    ax.text(2024.1, 1.65, 'KPI Target: <90 min (1.5 hrs)', color=TEAL, fontsize=9)

    # Gap shading
    ax.fill_between(scenarios['year'],
                    scenarios['projected_response_time_hrs_with_climateshield'],
                    scenarios['projected_response_time_hrs_status_quo'],
                    color=TEAL, alpha=0.12, label='Impact of ClimateShield')

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Average Emergency Response Time (hrs)', fontsize=11)
    ax.set_title('Emergency Response Time — Status Quo vs ClimateShield Scenario\nRidge Regression Model | Features: Flood Events, Heat Days, Insured Losses',
                 fontsize=12, fontweight='bold', color=WHITE, pad=15)
    ax.legend(loc='upper left', framealpha=0.15, labelcolor=OFF_WHITE, fontsize=9)
    ax.grid(axis='y', alpha=0.25)
    ax.set_ylim(0, 10)

    fig.text(0.01, 0.01, 'Sources: IBC/Public Safety Canada impact data | ClimateShield Ridge Regression Model',
             fontsize=7.5, color=MID_GRAY, style='italic')
    save_chart(fig, '06_response_time_deterioration.png')


# ─────────────────────────────────────────────────────────────────────────────
# 07 — Hospitalisation Avoided Harm Scenario
# ─────────────────────────────────────────────────────────────────────────────
def chart_hospitalisation_scenario(model_outputs):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    sc = model_outputs['hospitalisation']['scenario']

    ax.bar(sc['year'], sc['predicted_hosp_baseline'], color=RED, alpha=0.65,
           width=0.7, label='Baseline Hospitalisations (predicted)')
    ax.bar(sc['year'], sc['predicted_hosp_with_alerts'], color=TEAL, alpha=0.85,
           width=0.7, label='With ClimateShield Alerts (−30%)')

    for _, row in sc.iterrows():
        ax.text(row['year'], row['predicted_hosp_with_alerts'] + 5,
                f"−{row['avoided_hospitalisations']:.0f}", ha='center',
                fontsize=8.5, color=MINT, fontweight='bold')

    total_avoided = sc['avoided_hospitalisations'].sum()
    ax.text(2022.5, 770, f'Total avoided:\n{total_avoided:.0f} hospitalisations\n(2015–2023 backtest)',
            color=WHITE, fontsize=10, fontweight='bold', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=CARD_BG, edgecolor=MINT, alpha=0.9))

    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Heat-Related Hospitalisations (Ontario)', fontsize=11)
    ax.set_title('Heat-Related Hospitalisations — Baseline vs ClimateShield Early Warning\nGradient Boosting Model | 30% Reduction Assumption (Literature-based)',
                 fontsize=12, fontweight='bold', color=WHITE, pad=15)
    ax.legend(loc='upper left', framealpha=0.15, labelcolor=OFF_WHITE, fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    fig.text(0.01, 0.01, 'Sources: IBC/Public Safety Canada | ClimateShield GB Model | 30% reduction per WHO early warning literature',
             fontsize=7.5, color=MID_GRAY, style='italic')
    save_chart(fig, '07_hospitalisation_scenario.png')


# ─────────────────────────────────────────────────────────────────────────────
# 08 — Funding Growth
# ─────────────────────────────────────────────────────────────────────────────
def chart_funding_growth(funding_df):
    fig, ax = plt.subplots(figsize=(11, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    width = 0.38
    x = np.arange(len(funding_df))
    ax.bar(x - width/2, funding_df['fcm_gmf_disbursed_m_cad'],
           width, color=TEAL, alpha=0.88, label='FCM Green Municipal Fund')
    ax.bar(x + width/2, funding_df['infra_canada_climate_m_cad'],
           width, color=BLUE, alpha=0.88, label='Infrastructure Canada Climate')

    ax2 = ax.twinx()
    ax2.plot(x, funding_df['num_municipalities_funded'],
             color=AMBER, lw=2.5, marker='o', markersize=6, label='# Municipalities funded')
    ax2.set_ylabel('Municipalities Funded', color=AMBER, fontsize=10)
    ax2.tick_params(colors=AMBER)

    ax.set_xticks(x)
    ax.set_xticklabels(funding_df['year'])
    ax.set_xlabel('Year', fontsize=11)
    ax.set_ylabel('Funding Disbursed (CAD Millions)', fontsize=11)
    ax.set_title('Federal Climate Funding to Canadian Municipalities 2018–2025\nFCM Green Municipal Fund + Infrastructure Canada',
                 fontsize=12, fontweight='bold', color=WHITE, pad=15)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
              framealpha=0.15, labelcolor=OFF_WHITE, fontsize=9)
    ax.grid(axis='y', alpha=0.25)

    fig.text(0.01, 0.01, 'Sources: FCM Annual Reports 2018-2025 | Infrastructure Canada Climate Resilience Program',
             fontsize=7.5, color=MID_GRAY, style='italic')
    save_chart(fig, '08_funding_growth.png')


# ─────────────────────────────────────────────────────────────────────────────
# 09 — Climate Risk Composite Dashboard (multi-panel)
# ─────────────────────────────────────────────────────────────────────────────
def chart_composite_dashboard(heat_df, impact_df, pop_risk, infra_vuln):
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor(DARK_BG)
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # Panel 1: Heat days trend
    ax1 = fig.add_subplot(gs[0, 0])
    hist = heat_df[heat_df['data_type'] == 'historical']
    ax1.plot(hist['year'], hist['extreme_heat_days'], color=AMBER, lw=2, marker='o', markersize=3)
    ax1.fill_between(hist['year'], 0, hist['extreme_heat_days'], color=AMBER, alpha=0.15)
    ax1.set_title('Extreme Heat Days (1990–2023)', fontsize=10, fontweight='bold')
    ax1.set_ylabel('Days', fontsize=9)
    ax1.grid(alpha=0.2)

    # Panel 2: Flood events
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.bar(impact_df['year'], impact_df['flood_events_durham'], color=BLUE, alpha=0.8)
    z = np.polyfit(impact_df['year'], impact_df['flood_events_durham'], 1)
    ax2.plot(impact_df['year'], np.poly1d(z)(impact_df['year']),
             color=MINT, lw=2, linestyle='--')
    ax2.set_title('Annual Flood Events — Durham', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Events', fontsize=9)
    ax2.grid(alpha=0.2)

    # Panel 3: Insured losses
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.bar(impact_df['year'], impact_df['insured_losses_m_cad'] / 1000, color=RED, alpha=0.8)
    ax3.set_title('Insured Losses (CAD B)', fontsize=10, fontweight='bold')
    ax3.set_ylabel('CAD Billions', fontsize=9)
    ax3.grid(alpha=0.2)

    # Panel 4: Population growth
    ax4 = fig.add_subplot(gs[1, 0])
    hist_pop = pop_risk[~pop_risk['is_projection']]
    proj_pop = pop_risk[pop_risk['is_projection']]
    ax4.bar(hist_pop['year'], hist_pop['population'] / 1e6, color=TEAL, alpha=0.85, width=3)
    ax4.bar(proj_pop['year'], proj_pop['population'] / 1e6, color=TEAL, alpha=0.4, width=3, hatch='///')
    ax4.set_title('Durham Population (M)', fontsize=10, fontweight='bold')
    ax4.set_ylabel('Millions', fontsize=9)
    ax4.grid(alpha=0.2)

    # Panel 5: Infrastructure vulnerability
    ax5 = fig.add_subplot(gs[1, 1])
    top5 = infra_vuln.nlargest(5, 'vulnerability_score_norm')
    tier_colors = {'High': RED, 'Medium': AMBER, 'Low': TEAL}
    colors = [tier_colors.get(str(t), TEAL) for t in top5['priority_tier']]
    ax5.barh(top5['infrastructure_category'], top5['vulnerability_score_norm'],
             color=colors, alpha=0.85)
    ax5.set_title('Top 5 Vulnerable Assets', fontsize=10, fontweight='bold')
    ax5.set_xlabel('Vulnerability Score', fontsize=9)
    ax5.grid(alpha=0.2)

    # Panel 6: Response time
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(impact_df['year'], impact_df['avg_response_time_hrs'],
             color=RED, lw=2, marker='o', markersize=4)
    ax6.axhline(1.5, color=MINT, lw=1.5, linestyle='--', label='KPI Target')
    ax6.set_title('Avg Response Time (hrs)', fontsize=10, fontweight='bold')
    ax6.set_ylabel('Hours', fontsize=9)
    ax6.legend(fontsize=8, framealpha=0.2)
    ax6.grid(alpha=0.2)

    fig.suptitle('ClimateShield — Durham Region Climate Risk Dashboard\nAll Panels: Official Canadian Open Data Sources',
                 fontsize=14, fontweight='bold', color=WHITE, y=1.01)
    fig.text(0.01, -0.02, 'Sources: ECCC | Statistics Canada | GEI Consultants 2022 | TRCA 2024 | IBC | FCM',
             fontsize=8, color=MID_GRAY, style='italic')
    save_chart(fig, '09_climate_risk_composite_dashboard.png')


# ─────────────────────────────────────────────────────────────────────────────
# 10 — ML Model Accuracy Summary
# ─────────────────────────────────────────────────────────────────────────────
def chart_model_accuracy(results_path):
    import json
    with open(results_path) as f:
        results = json.load(f)

    model_names = list(results.keys())
    r2_vals  = [results[m]['r2']   for m in model_names]
    rmse_vals = [results[m]['rmse'] for m in model_names]

    display_names = {
        'heat_projection':  'Heat Projection\n(Poly Regression)',
        'flood_frequency':  'Flood Frequency\n(Random Forest)',
        'hospitalisation':  'Hospitalisation Risk\n(Gradient Boosting)',
        'response_time':    'Response Time\n(Ridge Regression)',
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(DARK_BG)

    labels = [display_names.get(m, m) for m in model_names]
    colors = [TEAL if r >= 0.85 else AMBER if r >= 0.70 else RED for r in r2_vals]

    # R² chart
    bars = ax1.bar(labels, r2_vals, color=colors, alpha=0.88, width=0.55)
    ax1.axhline(0.85, color=MINT, lw=1.5, linestyle='--', alpha=0.7)
    ax1.text(3.3, 0.86, 'Good fit', color=MINT, fontsize=8.5)
    for bar, val in zip(bars, r2_vals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.3f}', ha='center', fontsize=11, fontweight='bold', color=WHITE)
    ax1.set_ylabel('R² Score', fontsize=11)
    ax1.set_title('Model Fit (R²)\nHigher = better', fontsize=12, fontweight='bold', color=WHITE)
    ax1.set_ylim(0, 1.1)
    ax1.grid(axis='y', alpha=0.25)

    # RMSE chart
    bars2 = ax2.bar(labels, rmse_vals, color=colors, alpha=0.88, width=0.55)
    for bar, val in zip(bars2, rmse_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f'{val:.2f}', ha='center', fontsize=11, fontweight='bold', color=WHITE)
    ax2.set_ylabel('RMSE', fontsize=11)
    ax2.set_title('Root Mean Square Error\nLower = better', fontsize=12, fontweight='bold', color=WHITE)
    ax2.grid(axis='y', alpha=0.25)

    fig.suptitle('ClimateShield ML Model Performance Summary\nTime-Series Cross-Validation',
                 fontsize=14, fontweight='bold', color=WHITE)
    plt.tight_layout()
    save_chart(fig, '10_model_accuracy_summary.png')


# ─────────────────────────────────────────────────────────────────────────────
# MASTER RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def generate_all_charts(datasets, eda_outputs, model_outputs):
    print("\n📈 Generating Charts")
    print("=" * 50)

    chart_heat_projection(eda_outputs['heat_features'], model_outputs)
    chart_population_exposure(eda_outputs['pop_risk'])
    chart_infrastructure_vulnerability(eda_outputs['infra_vuln'])
    chart_flood_risk_category(datasets['flood_risk'])
    chart_insured_losses(datasets['impacts'])
    chart_response_time(model_outputs, datasets['impacts'])
    chart_hospitalisation_scenario(model_outputs)
    chart_funding_growth(datasets['funding'])
    chart_composite_dashboard(datasets['heat'], datasets['impacts'],
                               eda_outputs['pop_risk'], eda_outputs['infra_vuln'])
    results_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'models', 'model_results.json')
    chart_model_accuracy(results_path)

    chart_files = sorted(os.listdir(CHARTS_DIR))
    print(f"\n✅ {len(chart_files)} charts saved to outputs/charts/")
    return chart_files


if __name__ == '__main__':
    datasets     = load_all()
    eda_outputs  = run_eda(datasets)
    model_outputs = run_all_models(eda_outputs)
    generate_all_charts(datasets, eda_outputs, model_outputs)
