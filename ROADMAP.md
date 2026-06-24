# AI Fairness Platform — Roadmap

## Phase 1 — Core Engine (Current)
- [x] 8 fairness metrics (SPD, DI, EOD, EqOdds, PP, Calibration, TE, Individual)
- [x] 6 mitigation techniques (Reweighing, DIR, EG, PR, EqOdds, ROC)
- [x] 4 benchmark datasets (Adult, COMPAS, German, Bank)
- [x] FastAPI backend + React dashboard
- [x] Explainability (SHAP + LIME)
- [x] Drift monitoring (KS test, Chi², fairness drift)
- [x] HTML fairness audit reports
- [x] MLflow experiment tracking

## Phase 2 — Research Extensions
- [ ] Counterfactual fairness metrics
- [ ] Causal fairness (do-calculus based)
- [ ] Multi-label fairness (beyond binary classification)
- [ ] Regression fairness (continuous outcomes)
- [ ] NLP fairness (text classification models)
- [ ] Image model fairness (CV applications)

## Phase 3 — Enterprise Features
- [ ] OAuth2/SSO authentication
- [ ] Multi-tenant dataset isolation
- [ ] Audit log and compliance reporting
- [ ] Real-time production monitoring via webhook
- [ ] OpenAI-powered AI advisor (full implementation)
- [ ] PDF report generation (with charts)

## Phase 4 — Ecosystem
- [ ] Hugging Face model hub integration
- [ ] MLflow model registry with fairness gates
- [ ] Kubernetes operator for continuous fairness monitoring
- [ ] VS Code extension for in-editor fairness checks

## Research Agenda
- [ ] Theoretical analysis of mitigation technique interactions
- [ ] Benchmark: do pre/in/post-processing techniques conflict?
- [ ] Impossibility theorem analysis on our benchmark datasets
- [ ] User study: which explanation format (SHAP vs. LIME) is more actionable?
