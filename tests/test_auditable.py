"""Tests for the @auditable decorator + AuditContext."""
from __future__ import annotations


from mr_audit import AuditContext, auditable
from mr_audit.core.hash import verify_chain


def test_decorator_no_context_runs_unrecorded():
    """Without an active AuditContext, the decorator is a no-op."""
    @auditable(model_name="Pass-through", model_version="0.1.0")
    def f(x):
        return x * 2.0
    assert f(3.0) == 6.0


def test_decorator_writes_one_record_per_call(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x + 1.0

    db = tmp_path / "test.db"
    with AuditContext(db) as ctx:
        predict(x=1.0)
        predict(x=2.0)
        predict(x=3.0)
        records = ctx.store.read_all()
    assert len(records) == 3
    assert all(r.model.name == "M" for r in records)


def test_predictions_chain_correctly(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x * 2.0

    db = tmp_path / "test.db"
    with AuditContext(db) as ctx:
        predict(x=1.0)
        predict(x=2.0)
        predict(x=3.0)
        records = ctx.store.read_all()
    valid, errors = verify_chain(records)
    assert valid, f"Chain invalid: {errors}"


def test_private_kwargs_excluded_from_hash(tmp_path):
    """kwargs prefixed with _ are mr-audit metadata, not data."""
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x * 2.0

    db1 = tmp_path / "a.db"
    db2 = tmp_path / "b.db"
    with AuditContext(db1) as ctx1:
        predict(x=1.0, _data_source="src1", _random_seed=42)
        rec1 = ctx1.store.read_all()[0]
    with AuditContext(db2) as ctx2:
        predict(x=1.0, _data_source="src2", _random_seed=99)
        rec2 = ctx2.store.read_all()[0]
    # Same x, different private kwargs -> same data hash
    assert rec1.data.input_data_hash == rec2.data.input_data_hash
    # But the data_source and random_seed are still recorded
    assert rec1.data.input_data_source == "src1"
    assert rec2.data.input_data_source == "src2"
    assert rec1.random_seed == 42
    assert rec2.random_seed == 99


def test_horizon_recorded(tmp_path):
    @auditable(model_name="M", model_version="1.0.0", horizon_days=5)
    def predict(*, x):
        return x

    db = tmp_path / "h.db"
    with AuditContext(db) as ctx:
        predict(x=1.0)
        rec = ctx.store.read_all()[0]
    assert rec.prediction.horizon_days == 5


def test_predicted_for_date_passed_through(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x

    db = tmp_path / "d.db"
    with AuditContext(db) as ctx:
        predict(x=1.0, _predicted_for_date="2026-05-01")
        rec = ctx.store.read_all()[0]
    assert rec.prediction.predicted_for_date == "2026-05-01"


def test_feature_extractor_populates_summary(tmp_path):
    @auditable(
        model_name="M",
        model_version="1.0.0",
        feature_extractor=lambda *, x: {"x": x, "x_squared": x * x},
    )
    def predict(*, x):
        return x

    db = tmp_path / "f.db"
    with AuditContext(db) as ctx:
        predict(x=3.0)
        rec = ctx.store.read_all()[0]
    assert rec.data.feature_values_summary == {"x": 3.0, "x_squared": 9.0}
    assert "x" in rec.data.feature_names
    assert "x_squared" in rec.data.feature_names


def test_jsonl_backend_via_extension(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x

    log = tmp_path / "log.jsonl"
    with AuditContext(log) as ctx:
        predict(x=1.0)
        predict(x=2.0)
    # File should be JSONL
    lines = log.read_text().strip().split("\n")
    assert len(lines) == 2


def test_list_prediction_supported(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return [x, x * 2, x * 3]

    db = tmp_path / "list.db"
    with AuditContext(db) as ctx:
        predict(x=1.5)
        rec = ctx.store.read_all()[0]
    assert rec.prediction.value == [1.5, 3.0, 4.5]
