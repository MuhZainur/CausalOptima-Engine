"""Uplift Modeling and Heterogeneous Treatment Effect (CATE) Estimators."""

from typing import Literal
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from econml.dml import LinearDML


FEATURE_COLS = [
    "tenure_months",
    "baseline_spend_usd",
    "recency_days",
    "frequency_orders",
    "discount_sensitivity",
]


class TLearnerUplift:
    """Two-Model Estimator: Separate LightGBM models for control and treated cohorts."""

    def __init__(self, n_estimators: int = 100, max_depth: int = 4, seed: int = 42):
        self.model_control = LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, verbose=-1
        )
        self.model_treated = LGBMRegressor(
            n_estimators=n_estimators, max_depth=max_depth, random_state=seed, verbose=-1
        )

    def fit(self, X: pd.DataFrame, treatment: np.ndarray, outcome: np.ndarray) -> "TLearnerUplift":
        is_treated = treatment == 1
        self.model_treated.fit(X[is_treated], outcome[is_treated])
        self.model_control.fit(X[~is_treated], outcome[~is_treated])
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        """Estimate Individual Treatment Effect Tau_i = E[Y|X, T=1] - E[Y|X, T=0]."""
        y1_pred = self.model_treated.predict(X)
        y0_pred = self.model_control.predict(X)
        return y1_pred - y0_pred

    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        return self.model_control.predict(X)


class DMLUplift:
    """Double Machine Learning (Chernozhukov et al.) with Neyman Orthogonal Residuals."""

    def __init__(self, seed: int = 42):
        model_y = LGBMRegressor(n_estimators=100, max_depth=4, random_state=seed, verbose=-1)
        model_t = LGBMRegressor(n_estimators=100, max_depth=4, random_state=seed, verbose=-1)
        self.dml = LinearDML(
            model_y=model_y,
            model_t=model_t,
            discrete_treatment=True,
            random_state=seed,
            cv=3,
        )

    def fit(
        self,
        Y: np.ndarray,
        T: np.ndarray,
        X: pd.DataFrame,
        W: pd.DataFrame | None = None,
    ) -> "DMLUplift":
        """Fit DML on outcome Y, treatment T, effect modifiers X, and confounders W."""
        self.dml.fit(Y, T, X=X, W=W)
        return self

    def predict_cate(self, X: pd.DataFrame) -> np.ndarray:
        return self.dml.effect(X)


def classify_quadrant(
    cate: np.ndarray,
    baseline_pred: np.ndarray,
    tau_threshold_pos: float = 8.0,
    tau_threshold_neg: float = -5.0,
    baseline_threshold: float = 30.0,
) -> np.ndarray:
    """Classify users into 4 Causal Quadrants based on estimated uplift and untreated baseline."""
    n = len(cate)
    quadrants = np.empty(n, dtype=object)

    is_persuadable = cate >= tau_threshold_pos
    is_sleeping_dog = cate <= tau_threshold_neg
    is_sure_thing = ~is_persuadable & ~is_sleeping_dog & (baseline_pred >= baseline_threshold)
    is_lost_cause = ~is_persuadable & ~is_sleeping_dog & ~is_sure_thing

    quadrants[is_persuadable] = "Persuadable"
    quadrants[is_sleeping_dog] = "Sleeping Dog"
    quadrants[is_sure_thing] = "Sure Thing"
    quadrants[is_lost_cause] = "Lost Cause"
    return quadrants


if __name__ == "__main__":
    from src.data.generator import generate_causal_data

    raw = generate_causal_data(n_samples=2000).to_pandas()
    X = raw[FEATURE_COLS]
    t = raw["treatment"].values
    y = raw["outcome"].values

    # Test T-Learner
    tlearner = TLearnerUplift().fit(X, t, y)
    cate_pred = tlearner.predict_cate(X)
    assert len(cate_pred) == 2000
    print("TLearner mean CATE:", np.mean(cate_pred))
