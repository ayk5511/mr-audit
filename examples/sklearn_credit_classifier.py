"""mr-audit on a sklearn LogisticRegression credit-risk classifier.

Demonstrates that mr-audit is not GARCH/vol-specific. The same one-decorator
integration works for any prediction function.

Scenario: a small bank operating a logistic-regression credit scorecard. SR
11-7 §V (Ongoing Monitoring) requires per-prediction logging of inputs,
parameters, prediction probability, and decision. EU AI Act Article 6 +
Annex III.5(b) explicitly classifies creditworthiness assessment as
high-risk, so Article 12 / 19 logging is mandatory in scope.

Synthetic data only (no real applicant PII).
"""
from __future__ import annotations

import numpy as np
from mr_audit import AuditContext, auditable
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(2026)


def main():
    # --- Synthetic credit data ---
    X, y = make_classification(
        n_samples=2000, n_features=10, n_informative=6, n_redundant=2,
        n_classes=2, weights=[0.85, 0.15], random_state=2026,
    )
    feature_names = [f"f{i}" for i in range(X.shape[1])]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.4, random_state=2026
    )

    # Train
    clf = LogisticRegression(C=1.0, max_iter=200, random_state=2026)
    clf.fit(X_train, y_train)
    print(f"Trained LogisticRegression: train acc {clf.score(X_train, y_train):.3f}, "
          f"test acc {clf.score(X_test, y_test):.3f}")

    # Capture full coefficient set as model parameters
    model_params = {
        "C": 1.0,
        "max_iter": 200,
        "n_features": int(X.shape[1]),
        "intercept": float(clf.intercept_[0]),
        "coef": [float(c) for c in clf.coef_[0]],
        "classes": [int(c) for c in clf.classes_],
    }

    @auditable(
        model_name="LogisticRegression-CreditScore",
        model_version="1.0.0",
        feature_extractor=lambda *, features, **_: dict(zip(feature_names, features)),
    )
    def score_applicant(*, features, threshold=0.5, **_):
        """Return (predicted_default_probability, decision).

        decision: 1.0 if probability >= threshold (deny), 0.0 otherwise.
        Logged value is the probability; the decision is recorded in
        model parameters for downstream auditing.
        """
        x = np.asarray(features).reshape(1, -1)
        proba = float(clf.predict_proba(x)[0, 1])
        return proba

    # --- Audit-instrumented inference loop ---
    log_path = "examples/audit_logs/credit_classifier.db"
    n_decisions = 0
    n_denied = 0
    print(f"\nLogging {len(X_test)} credit decisions to {log_path}...")
    with AuditContext(log_path) as ctx:
        for i in range(len(X_test)):
            features = X_test[i].tolist()
            proba = score_applicant(
                features=features,
                threshold=0.5,
                **model_params,
                _data_source="synthetic_make_classification:seed=2026",
                _predicted_for_date="2026-04-28",
                _random_seed=2026,
            )
            n_decisions += 1
            if proba >= 0.5:
                n_denied += 1
        records = ctx.store.read_all()

    print(f"  Audit records written: {len(records)}")
    print(f"  Decisions: {n_decisions} total, {n_denied} denied "
          f"({100*n_denied/n_decisions:.1f}%)")
    print(f"  First record's model.parameters has {len(records[0].model.parameters)} keys "
          f"including the full coefficient vector")

    # Verify chain
    from mr_audit import verify_chain
    valid, errors = verify_chain(records)
    print(f"  Chain valid: {valid}")

    # Show one record's structure
    print("\nExample record [0]:")
    rec = records[0]
    print(f"  model: {rec.model.name} v{rec.model.version}")
    print(f"  predicted: {rec.prediction.value:.4f}")
    print(f"  features captured: {list(rec.data.feature_values_summary.keys())}")
    print(f"  intercept logged: {rec.model.parameters['intercept']:.4f}")
    print(f"  coef[0] logged: {rec.model.parameters['coef'][0]:.4f}")


if __name__ == "__main__":
    main()
