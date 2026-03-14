"""
run_pipeline.py
===============
ClimateShield — Master Pipeline Runner

Usage:
    python run_pipeline.py              # full pipeline
    python run_pipeline.py --data-only  # data + EDA only (skip ML + charts)
    python run_pipeline.py --no-charts  # data + EDA + ML (skip chart generation)

Outputs:
    data/raw/           — raw datasets (CSV)
    data/processed/     — engineered features (CSV)
    outputs/models/     — trained models (.pkl) + results (.json)
    outputs/charts/     — publication-quality charts (.png)
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    args = sys.argv[1:]
    data_only  = '--data-only'  in args
    no_charts  = '--no-charts'  in args

    print("=" * 60)
    print("  ClimateShield — Data Science Pipeline")
    print("  Ontario Tech University | Smart Communities Challenge")
    print("=" * 60)
    t0 = time.time()

    from data_loader import load_all
    datasets = load_all()

    from eda_feature_engineering import run_eda
    eda_outputs = run_eda(datasets)

    if data_only:
        print(f"\n✅ Data + EDA complete in {time.time()-t0:.1f}s")
        return

    from ml_models import run_all_models
    model_outputs = run_all_models(eda_outputs)

    if not no_charts:
        from visualizations import generate_all_charts
        generate_all_charts(datasets, eda_outputs, model_outputs)

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Charts:  outputs/charts/")
    print(f"  Models:  outputs/models/")
    print(f"  Data:    data/processed/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
