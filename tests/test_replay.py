"""Tests for the replay engine."""
from __future__ import annotations


from mr_audit import AuditContext, auditable
from mr_audit.replay import replay


def test_replay_with_correct_inputs_matches(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x, beta=2.0):
        return x * beta + 0.1

    db = tmp_path / "r.db"
    with AuditContext(db) as ctx:
        predict(x=2.0, beta=3.0)
        rec = ctx.store.read_all()[0]

    res = replay(rec, predict.__wrapped__, kwargs={"x": 2.0, "beta": 3.0})
    assert res.input_hash_match is True
    assert res.prediction_match is True
    assert res.logged_value == 6.1
    assert res.replayed_value == 6.1


def test_replay_with_wrong_inputs_detects_mismatch(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x, beta=2.0):
        return x * beta

    db = tmp_path / "wrong.db"
    with AuditContext(db) as ctx:
        predict(x=2.0, beta=3.0)
        rec = ctx.store.read_all()[0]

    res = replay(rec, predict.__wrapped__, kwargs={"x": 99.0, "beta": 3.0})
    assert res.input_hash_match is False
    assert any("Input hash mismatch" in note for note in res.notes)


def test_replay_detects_changed_function_logic(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x * 2.0

    db = tmp_path / "logic.db"
    with AuditContext(db) as ctx:
        predict(x=5.0)
        rec = ctx.store.read_all()[0]

    # Replay with a different function (simulating code change)
    def different_predict(*, x):
        return x * 3.0  # different multiplier

    res = replay(rec, different_predict, kwargs={"x": 5.0})
    assert res.input_hash_match is True  # inputs match
    assert res.prediction_match is False  # but prediction differs
    assert res.logged_value == 10.0
    assert res.replayed_value == 15.0


def test_replay_preserves_record(tmp_path):
    """The record passed to replay must NOT be mutated."""
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x

    db = tmp_path / "preserve.db"
    with AuditContext(db) as ctx:
        predict(x=1.0)
        rec = ctx.store.read_all()[0]

    original_id = rec.record_id
    original_hash = rec.data.input_data_hash
    replay(rec, predict.__wrapped__, kwargs={"x": 1.0})
    assert rec.record_id == original_id
    assert rec.data.input_data_hash == original_hash


def test_replay_tolerance(tmp_path):
    """Replay tolerates small floating-point differences."""
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return x * 2.0

    db = tmp_path / "tol.db"
    with AuditContext(db) as ctx:
        predict(x=1.0)
        rec = ctx.store.read_all()[0]

    # Same logical function, but introducing a 1e-12 perturbation
    def slightly_different(*, x):
        return (x * 2.0) + 1e-12

    res = replay(rec, slightly_different, kwargs={"x": 1.0}, tol=1e-9)
    assert res.prediction_match is True


def test_replay_with_list_prediction(tmp_path):
    @auditable(model_name="M", model_version="1.0.0")
    def predict(*, x):
        return [x, x * 2, x * 3]

    db = tmp_path / "list.db"
    with AuditContext(db) as ctx:
        predict(x=2.0)
        rec = ctx.store.read_all()[0]

    res = replay(rec, predict.__wrapped__, kwargs={"x": 2.0})
    assert res.prediction_match is True
    assert res.logged_value == [2.0, 4.0, 6.0]
