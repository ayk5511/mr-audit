# mr-audit

**An audit-trail framework for ML pipelines under SR 11-7 and the EU AI Act.**

`mr-audit` is an open-source Python package that produces an immutable,
hash-chained, replayable, drift-aware audit trail for any machine-learning
inference pipeline. It is designed to satisfy the technical record-keeping
requirements implied by:

- **U.S. Federal Reserve SR 11-7 / OCC Bulletin 2011-12** — *Guidance on Model
  Risk Management* (April 4, 2011)
- **EU Regulation 2024/1689 (AI Act)** — Articles 12 and 19
  (Record-keeping, Automatically generated logs)
- **NIST AI 100-1** — *AI Risk Management Framework 1.0* (January 26, 2023)

## What it does

For every prediction your ML pipeline makes, `mr-audit` records:

- **Code state**: git commit hash of the calling code, library versions
- **Data state**: hash of input features, data-source version
- **Model state**: model name, version, full hyperparameters, model-artifact hash
- **Compute state**: Python version, platform, RAM
- **Random state**: random seed used
- **Prediction**: value, horizon, predicted-for date
- **Chain**: SHA-256 hash of this record, plus hash of previous record (Merkle-style)

Tampering with any single record breaks the chain.

## What it does NOT do

- It does **not** make any model "SR 11-7 compliant" or "EU AI Act compliant."
  Compliance is a determination by the firm and its regulators; `mr-audit`
  provides the technical infrastructure that compliance requires.
- It is **not** production-ready software. It is a research artifact released
  under MIT license, intended to demonstrate that audit-grade logging is
  feasible without commercial cloud platforms.

## Quickstart

```python
from mr_audit import AuditContext, auditable

@auditable(model_name="GJR-GARCH", model_version="1.0.0")
def predict_volatility(features, params):
    # your model logic here
    return forecast

with AuditContext(log_path="audit/run_2026_04_28.db"):
    forecast = predict_volatility(features, params)

# Inspect the log:
$ mr-audit show audit/run_2026_04_28.db
$ mr-audit replay audit/run_2026_04_28.db --record-id <id>
$ mr-audit verify audit/run_2026_04_28.db
$ mr-audit export audit/run_2026_04_28.db --bundle audit_bundle.zip
```

## Companion paper

The package is the technical artifact accompanying:

> Khan, A. (2026). "An Audit-Trail Framework for ML Pipelines under SR 11-7
> and the EU AI Act: A Demonstration on Volatility Forecasting." Working
> paper. SSRN.

The paper applies `mr-audit` to instrument a complete ML volatility-forecasting
pipeline (the seven-model panel of [Khan 2026](https://github.com/ayk5511/volatility-forecasting))
and demonstrates four use cases: reproducibility audit, drift detection,
model-risk validation report, and regulator-export bundle.

## License

MIT. See [LICENSE](LICENSE).
