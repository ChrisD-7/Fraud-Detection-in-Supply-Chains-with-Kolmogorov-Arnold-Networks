"""Data loading, curation and leak-free preparation.

Every column dropped here is dropped for a stated reason. The column lists are the
single source of truth for what the model is allowed to see, and
``assert_no_leakage`` re-checks them at run time so a later edit cannot quietly
reintroduce the original bug (the label was derived from ``Order Status`` while
``Order Status`` stayed in the feature matrix).

Invariant enforced throughout: **nothing may be fitted on the full data**. Sampling,
splitting and then fitting encoders/scaler on the train split only - including the
time-feature origin, which is a fixed epoch rather than ``data.min()`` so that no
full-data statistic enters a feature.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from .config import Config

# --- column provenance -------------------------------------------------------

# The target is a function of these. Leaving any of them in X is target leakage.
LEAK_COLUMNS = ("Order Status", "Delivery Status", "Late_delivery_risk")

# Group key used for splitting: a per-order identifier, never a feature.
GROUP_KEY_COLUMNS = ("Order Id",)

# Row identifiers / quasi-identifiers: near-unique, or duplicated per line item.
IDENTIFIER_COLUMNS = (
    "Customer Id",
    "Order Customer Id",
    "Order Item Id",
    "Product Card Id",
    "Order Item Cardprod Id",
    "Customer Zipcode",
    "Order Zipcode",
)

# Personal data. The public dataset already masks these, but they must never be
# features. Dropping them also keeps the pipeline safe if an unmasked copy is used.
PII_COLUMNS = (
    "Customer Email",
    "Customer Password",
    "Customer Street",
    "Customer Fname",
    "Customer Lname",
)

# Empty, free text or constant: all null, or an image path.
UNUSABLE_COLUMNS = ("Product Description", "Product Image", "Product Status")

# Recorded only after the shipment/order resolved, so unavailable at prediction time.
POST_HOC_COLUMNS = ("shipping date (DateOrders)",)

# Raw timestamps are replaced by the derived calendar features below.
RAW_DATETIME_COLUMNS = ("order date (DateOrders)",)

# Everything that must not be a feature, by reason. assert_no_leakage() checks all of
# them, so a curate() edit cannot silently widen the feature set.
MUST_DROP: dict[str, tuple[str, ...]] = {
    "label-derived": LEAK_COLUMNS,
    "group-key": GROUP_KEY_COLUMNS,
    "identifier": IDENTIFIER_COLUMNS,
    "pii": PII_COLUMNS,
    "post-hoc": POST_HOC_COLUMNS,
    "raw-datetime": RAW_DATETIME_COLUMNS,
    "unusable": UNUSABLE_COLUMNS,
}

TARGET_SOURCE = "Order Status"
TARGET_POSITIVE = "SUSPECTED_FRAUD"
TARGET_NAME = "flagged"

# Fixed epoch for order_days_since_start. A constant, deliberately: deriving it from
# the data (data.min()) would put a full-data statistic into a feature and break the
# split-before-fit invariant.
DATASET_EPOCH = pd.Timestamp("2015-01-01")
TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M"
# sklearn requires an integer (or NaN) sentinel with handle_unknown="use_encoded_value".
UNKNOWN_CATEGORY_CODE = -1


class LeakageError(AssertionError):
    """Raised when a column the label is derived from survives into the features."""


@dataclass
class PreparedData:
    """Leak-free, split-first, train-fitted arrays ready for the model."""

    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def feature_names(self) -> list[str]:
        return list(self.meta["feature_names"])

    @property
    def n_features(self) -> int:
        return self.x_train.shape[1]

    def to_torch(self, device: str = "cpu") -> dict[str, Any]:
        """pykan expects a dict of tensors; classification needs long class indices."""
        import torch

        def t(arr: np.ndarray, dtype) -> Any:
            return torch.tensor(arr).to(dtype).to(device)

        dtype = torch.get_default_dtype()
        return {
            "train_input": t(self.x_train, dtype),
            "test_input": t(self.x_test, dtype),
            "train_label": t(self.y_train, torch.long),
            "test_label": t(self.y_test, torch.long),
        }


def load_dataframe(cfg: Config, path: str | Path | None = None) -> pd.DataFrame:
    """Read the CSV straight out of the tracked zip - no manual unzip step."""
    target = Path(path or cfg.data.zip_path)
    if not target.exists():
        raise FileNotFoundError(
            f"{target} not found. Run from the repository root or set data.zip_path."
        )
    if target.suffix == ".zip":
        return pd.read_csv(target, compression="zip", encoding=cfg.data.encoding)
    return pd.read_csv(target, encoding=cfg.data.encoding)


def add_time_features(
    df: pd.DataFrame, column: str = RAW_DATETIME_COLUMNS[0]
) -> pd.DataFrame:
    """Derive calendar features from the order timestamp; the caller drops the column."""
    stamps = pd.to_datetime(df[column], format=TIMESTAMP_FORMAT, errors="coerce")
    if stamps.isna().any():
        raise ValueError(f"{int(stamps.isna().sum())} unparseable timestamps in {column!r}")
    df["order_year"] = stamps.dt.year
    df["order_month"] = stamps.dt.month
    df["order_dayofweek"] = stamps.dt.dayofweek
    df["order_hour"] = stamps.dt.hour
    df["order_days_since_start"] = (stamps - DATASET_EPOCH).dt.days
    return df


def build_target(df: pd.DataFrame) -> pd.Series:
    """Single definition of fraud: the order is flagged as suspected fraud.

    The original notebook OR-ed this with ``SUSPECTED_FRAUD & Late delivery``, a
    strict subset that matches zero rows in the dataset - dead code pretending to
    be a second condition.
    """
    return (df[TARGET_SOURCE] == TARGET_POSITIVE).astype(int).rename(TARGET_NAME)


def assert_no_leakage(feature_names: list[str]) -> None:
    """Fail the run if any must-drop column reached the feature matrix."""
    present = set(feature_names)
    leaked = sorted(present.intersection(LEAK_COLUMNS))
    if leaked:
        raise LeakageError(
            "label-derived column(s) reached the feature matrix: "
            f"{leaked}. The target is a function of these columns."
        )
    for reason, columns in MUST_DROP.items():
        still_there = sorted(present.intersection(columns))
        if still_there:
            raise LeakageError(f"{reason} column(s) reached the feature matrix: {still_there}")


def curate(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Attach the target, derive calendar features, drop everything unusable."""
    df = df.copy()
    df[TARGET_NAME] = build_target(df)
    df = add_time_features(df)

    to_drop = [
        *LEAK_COLUMNS,
        *IDENTIFIER_COLUMNS,
        *PII_COLUMNS,
        *UNUSABLE_COLUMNS,
        *POST_HOC_COLUMNS,
        *RAW_DATETIME_COLUMNS,
    ]
    missing = sorted(set(to_drop) - set(df.columns))
    if missing:
        raise KeyError(f"expected columns absent from the dataset: {missing}")

    kept = df.drop(columns=to_drop)
    if cfg.data.drop_na_rows:
        kept = kept.dropna()
    return kept


def _encode_categoricals(
    train: pd.DataFrame, test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Ordinal-encode non-numeric columns, fitting the encoders on TRAIN ONLY.

    Unseen test categories get an explicit sentinel (``UNKNOWN_CATEGORY_CODE``) rather
    than being folded onto some real class, and the count is reported, so silent
    feature corruption shows up in the run metadata instead of hiding.
    """
    train, test = train.copy(), test.copy()
    categorical = [
        column for column in train.columns if not pd.api.types.is_numeric_dtype(train[column])
    ]
    unseen_counts: dict[str, int] = {}
    for column in categorical:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=UNKNOWN_CATEGORY_CODE,
            dtype=np.float64,
        ).fit(train[[column]].astype(str))
        train[column] = encoder.transform(train[[column]].astype(str)).ravel()
        codes = encoder.transform(test[[column]].astype(str)).ravel()
        unseen = int((codes == UNKNOWN_CATEGORY_CODE).sum())
        if unseen:
            unseen_counts[column] = unseen
        test[column] = codes
    return train, test, unseen_counts


def _split(
    frame: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split into train/test before any statistic is fitted."""
    strategy = cfg.data.split_strategy
    if strategy not in ("grouped", "random"):
        raise ValueError(
            f"unknown split_strategy {strategy!r}; expected 'grouped' (per-order, no "
            "sibling leakage) or 'random'. Refusing to guess - a silent fallback to a "
            "random split would reopen the group-leakage channel."
        )

    y = frame[TARGET_NAME]
    if strategy == "grouped":
        group_column = cfg.data.group_column
        if group_column not in frame.columns:
            raise KeyError(f"group column {group_column!r} not in the frame")
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=cfg.data.test_size, random_state=cfg.data.seed
        )
        train_idx, test_idx = next(splitter.split(frame, y, frame[group_column]))
        return (
            frame.iloc[train_idx],
            frame.iloc[test_idx],
            {"split_strategy": "grouped", "group_column": group_column},
        )

    train, test = train_test_split(
        frame, test_size=cfg.data.test_size, stratify=y, random_state=cfg.data.seed
    )
    return train, test, {"split_strategy": "random", "group_column": None}


def _maybe_resample(
    x_train: np.ndarray, y_train: np.ndarray, cfg: Config
) -> tuple[np.ndarray, np.ndarray, str]:
    """Balance the TRAIN split only. Falls back to as-is when too small for ADASYN."""
    if cfg.data.balance == "none":
        return x_train, y_train, "none"

    minority = int(y_train.sum())
    if minority < cfg.data.min_minority_for_oversample:
        return (
            x_train,
            y_train,
            f"skipped:{cfg.data.balance} (only {minority} positive train rows)",
        )

    if cfg.data.balance == "adasyn":
        from imblearn.over_sampling import ADASYN

        sampler = ADASYN(random_state=cfg.data.seed)
    elif cfg.data.balance == "smote":
        from imblearn.over_sampling import SMOTE

        sampler = SMOTE(random_state=cfg.data.seed)
    else:
        raise ValueError(f"unknown balance strategy {cfg.data.balance!r}")

    x_res, y_res = sampler.fit_resample(x_train, y_train)
    return x_res, y_res, cfg.data.balance


def _sample(frame: pd.DataFrame, cfg: Config, meta: dict[str, Any]) -> pd.DataFrame:
    """Cap the row count before curation so small runs do not process 180k rows."""
    if cfg.data.n_sample is None:
        return frame
    if cfg.data.n_sample >= len(frame):
        meta["sample_requested_but_frame_smaller"] = cfg.data.n_sample
        return frame
    return frame.sample(n=cfg.data.n_sample, random_state=cfg.data.seed)


def prepare(cfg: Config, df: pd.DataFrame | None = None) -> PreparedData:
    """Full preparation: sample -> curate -> split -> fit on train -> resample train."""
    raw = load_dataframe(cfg) if df is None else df
    meta: dict[str, Any] = {
        "rows_loaded": int(len(raw)),
        "base_rate": float(build_target(raw).mean()),
        "zip_path": cfg.data.zip_path,
    }

    sampled = _sample(raw, cfg, meta)
    meta["rows_sampled"] = int(len(sampled))
    frame = curate(sampled, cfg)
    meta["rows_after_curation"] = int(len(frame))
    meta["rows_used"] = int(len(frame))

    train, test, split_meta = _split(frame, cfg)
    meta.update(split_meta)
    # The group key survives curation only so the split can be grouped: it is dropped
    # here, after the split, and the drop follows the configured column so the two
    # cannot desynchronise.
    group_columns = [c for c in {cfg.data.group_column, *GROUP_KEY_COLUMNS} if c in train.columns]
    if group_columns:
        train = train.drop(columns=group_columns)
        test = test.drop(columns=group_columns)

    y_train = train.pop(TARGET_NAME).to_numpy()
    y_test = test.pop(TARGET_NAME).to_numpy()
    train, test, unseen = _encode_categoricals(train, test)

    train_values = train.to_numpy(dtype=np.float64)
    scaler = StandardScaler().fit(train_values)
    x_train = scaler.transform(train_values).astype(np.float32)
    x_test = scaler.transform(test.to_numpy(dtype=np.float64)).astype(np.float32)

    x_train, y_train, balance_used = _maybe_resample(x_train, y_train, cfg)

    feature_names = list(train.columns)
    assert_no_leakage(feature_names)

    meta.update(
        {
            "n_features": int(x_train.shape[1]),
            "feature_names": feature_names,
            "train_rows": int(x_train.shape[0]),
            "test_rows": int(x_test.shape[0]),
            "train_positive": int(y_train.sum()),
            "test_positive": int(y_test.sum()),
            "test_base_rate": float(y_test.mean()),
            "balance": balance_used,
            "unseen_test_categories": unseen,
            "dropped_columns": sorted(set(raw.columns) - set(feature_names) - {TARGET_NAME}),
            "high_cardinality_features": {
                column: count
                for column in train.columns
                if (count := int(train[column].nunique())) > 50
            },
        }
    )
    return PreparedData(x_train, x_test, y_train, y_test, meta)
