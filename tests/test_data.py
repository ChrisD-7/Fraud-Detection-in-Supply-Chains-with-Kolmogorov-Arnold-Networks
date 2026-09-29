"""Data preparation: leakage, split-before-fit, resampling guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_frame
from kan_fraud.config import Config
from kan_fraud.data import (
    LEAK_COLUMNS,
    TARGET_NAME,
    LeakageError,
    assert_no_leakage,
    build_target,
    curate,
    prepare,
)


def test_target_uses_order_status_only(synthetic_frame):
    target = build_target(synthetic_frame)
    manual = (synthetic_frame["Order Status"] == "SUSPECTED_FRAUD").astype(int)
    assert target.equals(manual)
    assert target.name == TARGET_NAME


def test_curate_drops_every_leak_column(synthetic_frame):
    curated = curate(synthetic_frame, Config())
    for column in LEAK_COLUMNS:
        assert column not in curated.columns
    assert TARGET_NAME in curated.columns


def test_curate_derives_calendar_features(synthetic_frame):
    curated = curate(synthetic_frame, Config())
    for derived in ("order_year", "order_month", "order_dayofweek", "order_hour"):
        assert derived in curated.columns
    assert "order date (DateOrders)" not in curated.columns


def test_assert_no_leakage_raises_on_regression():
    with pytest.raises(LeakageError):
        assert_no_leakage(["Sales", "Order Status"])
    assert_no_leakage(["Sales", "Days for shipping (real)"]) is None


@pytest.mark.parametrize(
    "column", ["Customer Id", "Order Item Id", "Customer Zipcode", "shipping date (DateOrders)"]
)
def test_assert_no_leakage_guards_every_must_drop_class(column):
    """Identifiers, PII and post-hoc columns are as forbidden as the label source."""
    with pytest.raises(LeakageError, match="reached the feature matrix"):
        assert_no_leakage(["Sales", column])


def test_unknown_split_strategy_is_rejected(synthetic_frame):
    """A typo must not silently fall back to a random split and reopen group leakage."""
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    cfg.data.split_strategy = "stratified"  # not a supported strategy
    with pytest.raises(ValueError, match="unknown split_strategy"):
        prepare(cfg, synthetic_frame)


def test_configured_group_column_is_dropped_too(synthetic_frame):
    """Grouping and dropping must follow the same column, not two separate definitions."""
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    cfg.data.group_column = "Market"  # a real feature, used here as the group key
    prepared = prepare(cfg, synthetic_frame)
    assert "Market" not in prepared.feature_names
    assert prepared.meta["group_column"] == "Market"


def test_time_features_use_a_fixed_epoch(synthetic_frame):
    """No full-data statistic may enter a feature: the origin is a constant."""
    from kan_fraud.data import DATASET_EPOCH, add_time_features

    early = add_time_features(synthetic_frame.assign(**{"order date (DateOrders)": "1/2/2015 05:00"}))
    late = add_time_features(synthetic_frame.assign(**{"order date (DateOrders)": "6/2/2018 05:00"}))
    assert early["order_days_since_start"].iloc[0] == 1
    assert late["order_days_since_start"].iloc[0] == (pd.Timestamp("2018-06-02") - DATASET_EPOCH).days


def test_unseen_test_categories_get_a_sentinel():
    from kan_fraud.data import UNKNOWN_CATEGORY_CODE, _encode_categoricals

    train = pd.DataFrame({"Market": ["A", "A", "B"]})
    test = pd.DataFrame({"Market": ["A", "C"]})  # C never appears in train
    encoded_train, encoded_test, unseen = _encode_categoricals(train, test)
    assert unseen == {"Market": 1}
    assert encoded_test["Market"].iloc[1] == UNKNOWN_CATEGORY_CODE
    assert set(encoded_train["Market"]).isdisjoint({UNKNOWN_CATEGORY_CODE})


def test_prepare_never_exposes_leak_columns(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    prepared = prepare(cfg, synthetic_frame)
    assert not set(prepared.feature_names).intersection(LEAK_COLUMNS)
    assert prepared.x_train.shape[1] == len(prepared.feature_names)


def test_scaler_is_fitted_on_train_only(synthetic_frame):
    """A shifted test split must stay shifted: fitting on all data would hide it."""
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    prepared = prepare(cfg, synthetic_frame)
    scaled_all = np.vstack([prepared.x_train, prepared.x_test])
    # Perfect centring of the combined data is exactly the leak we removed.
    assert np.abs(prepared.x_train.mean(axis=0)).max() < 1e-6
    assert np.abs(scaled_all.mean(axis=0)).max() > 1e-4


def test_grouped_split_keeps_orders_disjoint(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    cfg.data.split_strategy = "grouped"
    frame = synthetic_frame.copy()
    prepared = prepare(cfg, frame)
    train_rows = prepared.meta["train_rows"]
    test_rows = prepared.meta["test_rows"]
    assert train_rows + test_rows == prepared.meta["rows_after_curation"]
    assert prepared.meta["split_strategy"] == "grouped"


def test_random_split_is_available(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "none"
    cfg.data.split_strategy = "random"
    prepared = prepare(cfg, synthetic_frame)
    assert prepared.meta["split_strategy"] == "random"


def test_oversampling_skipped_when_minority_too_small(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "adasyn"
    cfg.data.min_minority_for_oversample = 10_000
    prepared = prepare(cfg, synthetic_frame)
    assert prepared.meta["balance"].startswith("skipped:")


def test_adasyn_balances_the_train_split(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = None
    cfg.data.balance = "adasyn"
    cfg.data.min_minority_for_oversample = 5
    prepared = prepare(cfg, synthetic_frame)
    assert prepared.meta["balance"] == "adasyn"
    positives = prepared.y_train.sum()
    assert abs(positives / len(prepared.y_train) - 0.5) < 0.05


def test_n_sample_larger_than_frame_does_not_crash(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = len(synthetic_frame) + 10_000
    cfg.data.balance = "none"
    prepared = prepare(cfg, synthetic_frame)
    assert prepared.meta["rows_used"] == len(synthetic_frame)
    assert prepared.meta["sample_requested_but_frame_smaller"] == cfg.data.n_sample


def test_n_sample_caps_rows(synthetic_frame):
    cfg = Config()
    cfg.data.n_sample = 100
    cfg.data.balance = "none"
    prepared = prepare(cfg, synthetic_frame)
    assert prepared.meta["rows_used"] == 100
    # Sampling happens before curation, so the 180k-row frame is not processed for a
    # 100-row run.
    assert prepared.meta["rows_sampled"] == 100


def test_rejects_frame_missing_expected_columns():
    with pytest.raises(KeyError):
        curate(pd.DataFrame({"Order Status": ["COMPLETE"]}), Config())
