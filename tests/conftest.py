"""Shared fixtures: a synthetic frame with the real dataset's column names."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from kan_fraud.data import (
    GROUP_KEY_COLUMNS,
    IDENTIFIER_COLUMNS,
    LEAK_COLUMNS,
    PII_COLUMNS,
    POST_HOC_COLUMNS,
    RAW_DATETIME_COLUMNS,
    TARGET_NAME,
    UNUSABLE_COLUMNS,
)

REQUIRED_COLUMNS = (
    *LEAK_COLUMNS,
    *GROUP_KEY_COLUMNS,
    *IDENTIFIER_COLUMNS,
    *PII_COLUMNS,
    *UNUSABLE_COLUMNS,
    *POST_HOC_COLUMNS,
    *RAW_DATETIME_COLUMNS,
)


def make_frame(n_rows: int = 400, fraud_rate: float = 0.05, seed: int = 0) -> pd.DataFrame:
    """A small stand-in with the dataset's schema and a learnable-ish signal."""
    rng = np.random.default_rng(seed)
    n_orders = max(n_rows // 4, 2)
    order_ids = rng.integers(1000, 1000 + n_orders, size=n_rows)

    fraud = rng.random(n_rows) < fraud_rate
    frame = pd.DataFrame(
        {
            "Order Id": order_ids,
            "Order Status": np.where(fraud, "SUSPECTED_FRAUD", "COMPLETE"),
            "Delivery Status": np.where(fraud, "Late delivery", "Advance shipping"),
            "Late_delivery_risk": rng.integers(0, 2, size=n_rows),
            "Customer Id": rng.integers(1, 200, size=n_rows),
            "Order Customer Id": rng.integers(1, 200, size=n_rows),
            "Order Item Id": np.arange(n_rows),
            "Product Card Id": rng.integers(1, 30, size=n_rows),
            "Order Item Cardprod Id": rng.integers(1, 30, size=n_rows),
            "Customer Zipcode": rng.integers(100, 999, size=n_rows),
            "Order Zipcode": rng.integers(100, 999, size=n_rows),
            "Customer Email": "XXXXXXXXX",
            "Customer Password": "XXXXXXXXX",
            "Customer Street": "XXXXXXXXX",
            "Customer Fname": "XXXXXXXXX",
            "Customer Lname": "XXXXXXXXX",
            "Product Description": np.nan,
            "Product Image": "http://example.invalid/img.png",
            "Product Status": "1",
            "shipping date (DateOrders)": "1/1/2018 00:00",
            "order date (DateOrders)": [
                f"{rng.integers(1, 13)}/{rng.integers(1, 28)}/2018 {rng.integers(0, 24)}:00"
                for _ in range(n_rows)
            ],
            # legitimately usable features
            "Type": rng.choice(["DEBIT", "TRANSFER", "PAYMENT", "CASH"], size=n_rows),
            "Market": rng.choice(["Pacific Asia", "USCA", "Europe"], size=n_rows),
            "Category Name": rng.choice(["Sporting Goods", "Fishing", "Cleats"], size=n_rows),
            "Shipping Mode": rng.choice(["Standard Class", "First Class"], size=n_rows),
            "Days for shipping (real)": rng.integers(0, 7, size=n_rows),
            "Days for shipment (scheduled)": rng.integers(0, 7, size=n_rows),
            "Sales": rng.normal(200, 40, size=n_rows),
            "Benefit per order": rng.normal(20, 25, size=n_rows),
            "Order Item Discount": rng.normal(15, 8, size=n_rows),
            "Order Item Quantity": rng.integers(1, 5, size=n_rows),
            "Product Price": rng.normal(150, 50, size=n_rows),
            "Latitude": rng.normal(30, 5, size=n_rows),
            "Longitude": rng.normal(-90, 10, size=n_rows),
        }
    )
    # Signal so a model can learn something: fraud orders skew to late shipping.
    frame.loc[fraud, "Days for shipping (real)"] += 3
    return frame


@pytest.fixture
def synthetic_frame() -> pd.DataFrame:
    return make_frame()


@pytest.fixture
def smoke_config(tmp_path):
    from kan_fraud.config import Config

    cfg = Config()
    cfg.data.n_sample = None
    cfg.model.hidden = 2
    cfg.model.grid = 3
    cfg.model.steps = 1
    cfg.model.device = "cpu"  # deterministic in CI; pykan CUDA grid updates can fail
    cfg.symbolic.enabled = False
    cfg.output.results_dir = str(tmp_path / "results")
    cfg.output.run_name = "test"
    return cfg.validate()


__all__ = ["REQUIRED_COLUMNS", "make_frame", "TARGET_NAME"]
