# Contributing to AI Fairness Platform

Thank you for contributing to responsible AI! This document explains how to
add new fairness metrics, mitigation techniques, datasets, and features.

## Quick Setup

```bash
git clone https://github.com/your-org/detecting-and-mitigating-bias-in-machine-learning-models.git
cd detecting-and-mitigating-bias-in-machine-learning-models
cp .env.example .env
bash scripts/init.sh
```

## Contribution Areas

### Adding a New Fairness Metric
1. Add the computation to `backend/app/ml/metrics/fairness_metrics.py`
2. Add the field to `FairnessMetrics` dataclass
3. Update `_compute_severity_score()` if it affects overall severity
4. Add unit tests in `backend/tests/test_metrics.py`
5. Document the mathematical formula in the metric's docstring

### Adding a New Mitigation Technique
1. Implement in `backend/app/ml/mitigation/mitigator.py`
2. Register in `BiasMitigator.TECHNIQUES` dict
3. Add the handler in `BiasMitigator.mitigate()`
4. Add unit tests in `backend/tests/test_mitigation.py`
5. Reference the original paper in the docstring

### Adding a New Dataset
1. Create a loader class in `backend/app/services/dataset_service.py`
2. Register in `DATASET_LOADERS` dict
3. Document sensitive attributes and privileged values
4. Add download instructions to `datasets/README.md`
5. Add tests in `backend/tests/test_dataset_service.py`

## Code Standards
- Python: Black (line 100) + Ruff + Mypy strict
- TypeScript: ESLint + Prettier strict
- All PRs need: tests passing, coverage maintained, docstring with paper reference

## Research Contributions
When adding algorithms, always include:
- Mathematical formulation in docstring
- Original paper citation (BibTeX format)
- Limitations section
- Link to reference implementation (if exists)
