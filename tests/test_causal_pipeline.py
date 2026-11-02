"""End-to-End Pipeline Verification Suite for CausalOptima-Engine."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.data.generator import generate_causal_data
from src.causal.dag_model import build_and_identify_causal_model
from src.models.uplift import TLearnerUplift, FEATURE_COLS, classify_quadrant
from src.evaluation.metrics import compute_qini_curve, compute_campaign_profit
from src.api.main import app, load_or_train_model


@pytest.fixture(scope="session")
def sample_dataset():
    return generate_causal_data(n_samples=2500, seed=123).to_pandas()


def test_data_generation_integrity(sample_dataset):
    df = sample_dataset
    assert len(df) == 2500
    expected_cols = [
        "user_id", "tenure_months", "baseline_spend_usd", "recency_days",
        "frequency_orders", "discount_sensitivity", "treatment", "outcome",
        "true_tau", "quadrant"
    ]
    for col in expected_cols:
        assert col in df.columns

    # Verify all 4 quadrants are represented
    quadrants = df["quadrant"].unique().tolist()
    assert "Persuadable" in quadrants
    assert "Sleeping Dog" in quadrants
    assert "Sure Thing" in quadrants
    assert "Lost Cause" in quadrants


def test_dowhy_identification(sample_dataset):
    model, estimand = build_and_identify_causal_model(sample_dataset.head(500))
    assert model is not None
    assert estimand is not None
    assert "backdoor" in str(estimand).lower()


def test_tlearner_estimation_direction(sample_dataset):
    df = sample_dataset
    X = df[FEATURE_COLS]
    t = df["treatment"].values
    y = df["outcome"].values

    model = TLearnerUplift(n_estimators=30, max_depth=3).fit(X, t, y)
    cate_pred = model.predict_cate(X)
    baseline_pred = model.predict_baseline(X)

    assert len(cate_pred) == len(df)
    # Check that estimated mean uplift for true Persuadables is higher than for true Sleeping Dogs
    is_persuadable = df["quadrant"] == "Persuadable"
    is_sleeping_dog = df["quadrant"] == "Sleeping Dog"

    mean_tau_persuadable = np.mean(cate_pred[is_persuadable])
    mean_tau_sleeping_dog = np.mean(cate_pred[is_sleeping_dog])

    assert mean_tau_persuadable > mean_tau_sleeping_dog
    assert mean_tau_sleeping_dog < 0.0

    # Test quadrant classifier
    assigned_quads = classify_quadrant(cate_pred, baseline_pred)
    assert len(assigned_quads) == len(df)


def test_qini_curve_and_profit_metrics():
    y_mock = np.array([15, 25, 0, 5, 20, 35, 1, 4])
    t_mock = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    cate_mock = np.array([12.0, 18.0, -3.0, 0.5, 9.0, 22.0, -1.5, 2.0])

    pct, qini, auuc = compute_qini_curve(y_mock, t_mock, cate_mock, n_bins=4)
    assert len(pct) == len(qini)
    assert auuc > 0.0

    profit_dict = compute_campaign_profit(cate_mock, margin_rate=0.40, cost_per_treatment=2.50)
    assert profit_dict["targeted_users_count"] == 4  # Users with cate > 6.25: [12.0, 18.0, 9.0, 22.0]
    assert profit_dict["net_profit_usd"] > 0.0


def test_fastapi_decision_service():
    load_or_train_model()
    client = TestClient(app)

    # Health check
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["status"] == "healthy"

    # Persuadable candidate (high discount sensitivity, low recency)
    persuadable_payload = {
        "user_id": "usr_test_01",
        "tenure_months": 12,
        "baseline_spend_usd": 75.0,
        "recency_days": 15,
        "frequency_orders": 3,
        "discount_sensitivity": 0.95,
    }
    res = client.post("/api/v1/recommend-treatment", json=persuadable_payload)
    assert res.status_code == 200
    body = res.json()
    assert body["user_id"] == "usr_test_01"
    assert body["action"] in ["TREAT", "HOLD"]
    assert "causal_quadrant" in body

    # Sleeping dog candidate (low sensitivity, high frequency)
    sleeping_dog_payload = {
        "user_id": "usr_test_02",
        "tenure_months": 36,
        "baseline_spend_usd": 120.0,
        "recency_days": 85,
        "frequency_orders": 8,
        "discount_sensitivity": 0.05,
    }
    res = client.post("/api/v1/recommend-treatment", json=sleeping_dog_payload)
    assert res.status_code == 200
    body = res.json()
    assert body["action"] in ["SUPPRESS", "HOLD"]
