# Architecture — AI Fairness Platform

## Overview

The platform follows a clean 3-tier architecture:

```
Tier 1: Presentation    → React dashboard + Browser
Tier 2: Application     → FastAPI backend + ML pipelines
Tier 3: Data            → PostgreSQL + Redis + File system
```

## Directory Structure

```
detecting-and-mitigating-bias-in-machine-learning-models/
├── backend/                  # FastAPI application + ML pipelines
│   ├── app/
│   │   ├── api/routes/       # 7 route modules (20+ endpoints)
│   │   ├── core/             # Config, logging, rate limiting
│   │   ├── db/               # SQLAlchemy models (future)
│   │   ├── ml/
│   │   │   ├── metrics/      # 8 fairness metrics engine
│   │   │   ├── bias_detection/ # Auto-detection, proxy, intersectional
│   │   │   ├── mitigation/   # 6 mitigation algorithms
│   │   │   ├── explainability/ # SHAP + LIME + group analysis
│   │   │   └── monitoring/   # Drift detection (KS/Chi²/fairness)
│   │   ├── schemas/          # Pydantic request/response models
│   │   └── services/         # Dataset loader, model trainer, report generator
│   └── tests/                # pytest test suite (90%+ coverage target)
│
├── frontend/                 # React 19 dashboard
│   └── src/
│       └── App.tsx           # Full dashboard (9 pages, charts, dark theme)
│
├── datasets/
│   ├── raw/                  # Downloaded raw dataset files
│   ├── processed/            # Preprocessed CSV files
│   └── scripts/              # Download + validation scripts
│
├── models/
│   ├── checkpoints/          # Trained model pickles (gitignored)
│   └── registry/             # MLflow model registry stubs
│
├── notebooks/                # Jupyter analysis notebooks
├── reports/generated/        # HTML audit reports (gitignored)
├── scripts/                  # Preprocessing, init scripts
├── assets/                   # Banner SVG, screenshots
├── .github/                  # CI/CD workflows, issue templates
└── docs/                     # Extended documentation
```

## ML Pipeline Data Flow

```
Raw Data
  │
  ▼
DatasetBundle
  (X_train, X_test, y_train, y_test,
   sensitive_train, sensitive_test,
   privileged_values, feature_names)
  │
  ├──► ModelTrainer.train()
  │         │
  │         └──► TrainingResult
  │               (model, y_pred, y_prob, metrics, mlflow_run_id)
  │
  ├──► compute_all_metrics()
  │         │
  │         └──► dict[attr → FairnessMetrics]
  │               (SPD, DI, EOD, EqOdds, PP, Cal, TE, Individual)
  │
  ├──► BiasDetector.run()
  │         │
  │         └──► BiasDetectionReport
  │               (alerts, proxy_bias, intersectional, label_distribution)
  │
  ├──► BiasMitigator.run_all()
  │         │
  │         └──► dict[technique → MitigationResult]
  │               (before/after accuracy, before/after fairness metrics)
  │
  ├──► ExplainabilityEngine
  │         ├──► compute_shap_values() → global feature importance
  │         ├──► compute_shap_local()  → per-instance explanation
  │         └──► group_shap_analysis() → disparity across groups
  │
  └──► FairnessDriftMonitor.run()
            └──► DriftReport (data drift, fairness drift, quality issues)
```

## API Routes

| Module | Prefix | Endpoints |
|--------|--------|-----------|
| analysis.py | /api/v1/analysis | run, session/{id}, sessions |
| datasets.py | /api/v1/datasets | /, /{type}/info |
| mitigation.py | /api/v1/mitigation | techniques, run, run-all |
| reports.py | /api/v1/reports | generate, /{id}/download, /{id}/view |
| explainability.py | /api/v1/explain | global, local, group-shap |
| monitoring.py | /api/v1/monitoring | drift |
| advisor.py | /api/v1/advisor | explain-metric, recommendations |
