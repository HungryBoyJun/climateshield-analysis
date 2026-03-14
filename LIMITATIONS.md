# Model Limitations & Honest Caveats

> Credibility comes from knowing what your models can and cannot do.
> This document is intentionally included as part of the ClimateShield pipeline.

---

## Why This Document Exists

When presenting data-backed claims to judges, investors, or municipal partners,
the fastest way to lose credibility is to overclaim. This file documents known
limitations of our models so that anyone reviewing or building on this work
knows exactly where to apply caution.

---

## Dataset Limitations

### Small Sample Sizes
- **Climate impact data** (insured losses, flood events, hospitalisations): 9 years only (2015–2023).
  This is the full extent of consistently available IBC + Public Safety Canada annual data.
  With n=9, all models trained on this data are indicative rather than statistically conclusive.

- **Population data**: 7 historical census points + 6 projected. Projections are taken directly
  from the Durham Region Official Plan (2023) medium scenario — they are not independently modelled.

### Proxy Data
- Extreme heat days use **Oshawa ECCC Station 6156732** as a proxy for Durham Region.
  Durham spans urban (Oshawa, Ajax) and rural (Scugog, Brock) geographies with meaningfully
  different microclimates. A production system would integrate multiple stations.

- Flood risk percentages from **GEI Consultants 2022** are road-network focused and may not
  fully represent residential or agricultural exposure.

---

## Model Limitations

### Model 1 — Heat Projection (Polynomial Regression)

| Metric | Value | Notes |
|---|---|---|
| Training R² | 0.810 | Reasonable fit on full training set |
| CV R² (mean) | −1.604 | **Negative** — polynomial overfits on small held-out windows |
| TRCA validation | 3/6 checkpoints within 90% CI | Model underprojects post-2040 |

**What this means:** The polynomial trend captures the historical warming signal well
(R²=0.81) but the time-series cross-validation reveals it struggles to generalise
across disjoint time windows — a known issue with polynomial extrapolation on short
climate series. The TRCA RCP 8.5 projections (from a full climate ensemble model)
are the authoritative source; our model is a supporting data science exercise.

**Production recommendation:** Use an ensemble of ECCC CanESM5/CMIP6 projections
directly from the TRCA, rather than extrapolating from a single station.

---

### Model 2 — Flood Frequency (Random Forest)

| Metric | Value | Notes |
|---|---|---|
| Training R² | 0.825 | Good in-sample fit |
| CV R² (mean) | −12.002 | **Highly negative** — clear overfitting |
| Top feature | Insured losses (46.7%) | Possible target leakage |

**What this means:** With only 9 annual observations and 4 features, the Random Forest
memorises the training data. The negative CV scores confirm overfitting, not generalisation.
The feature importance showing insured losses as top predictor likely reflects simultaneous
causation (floods cause losses) rather than prediction power.

**Production recommendation:** Collect monthly or event-level flood data from TRCA
flood gauges, Durham Region emergency services logs, and insurance claims at a finer
temporal granularity. This is precisely the data gap ClimateShield's IoT layer would fill.

---

### Model 3 — Hospitalisation Risk (Gradient Boosting)

| Metric | Value | Notes |
|---|---|---|
| Training R² | 1.000 | Perfect — **severe overfitting** |
| CV R² (mean) | NaN | Insufficient samples for lag-2 splits |

**What this means:** With lag-2 features and n=7 effective samples, the Gradient Boosting
model achieves perfect training R² by memorisation. The NaN CV scores indicate that
TimeSeriesSplit exhausted the available data before generating valid held-out folds.
The 1,267 "avoided hospitalisations" figure is a scenario analysis output, not a
cross-validated prediction — the 30% reduction assumption is drawn from WHO early
warning effectiveness literature, not from this model alone.

**Production recommendation:** Use Ontario-level health unit data at monthly granularity,
linked to ECCC climate station records. Durham Region Health Department data would
be the ideal source.

---

### Model 4 — Response Time Deterioration (Ridge Regression)

| Metric | Value | Notes |
|---|---|---|
| Training R² | 0.967 | Very high |
| CV R² (mean) | −2.073 | Negative — overfitting on small n |

**What this means:** Same small-data problem. The Ridge model captures the trend in the
training set convincingly, but the time-series CV is unstable. The 60% response time
reduction KPI is an *aspirational target*, not a cross-validated prediction — it is
grounded in emergency management literature on early warning system effectiveness.

---

## Summary: What the Models ARE Good For

Despite the CV limitations, this pipeline provides genuine value:

1. **Trend direction** is correct and consistent across all models — heat days, flood events,
   losses, and response times are all worsening. The models agree on direction even where magnitude is uncertain.

2. **Feature engineering outputs** (vulnerability scores, exposure indices, compound risk flags)
   are valid and useful regardless of model fit.

3. **Scenario analysis** (response time, hospitalisation avoided harm) is transparently
   assumption-based and cites its assumptions — which is more rigorous than many
   municipal planning reports.

4. **The pipeline is production-ready** — swap in larger, higher-frequency datasets
   (which ClimateShield's IoT layer would generate) and the models immediately improve.

---

## The Honest Pitch to Judges

> "These models are trained on the best publicly available data for Durham Region.
> The small sample sizes reflect the exact data gap ClimateShield is designed to close.
> Our IoT sensor network would generate daily or hourly readings across 2 districts,
> replacing annual proxies with granular ground-truth — making these models materially
> more accurate within the first year of deployment."

This is both honest and a strong argument for why ClimateShield needs to exist.

---

## References

- WHO. (2004). *Using Climate to Predict Infectious Disease Epidemics.* Geneva: WHO Press.
- Lowe, R. et al. (2021). *Combined effects of hydrometeorological hazards and urbanisation on dengue risk in Brazil.* Nature Communications.
- TRCA. (2024). *Climate Projections for Durham Region.* Toronto and Region Conservation Authority.
- GEI Consultants. (2022). *Durham Region Flood Risk Roads Study.*
- IBC. (2023). *Severe Weather Drove Record Losses in 2023.* Insurance Bureau of Canada.
