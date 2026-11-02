"""FastAPI Decision Server: Real-Time Personalized Policy Recommendation."""

from contextlib import asynccontextmanager
from typing import List, Literal
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd

from src.models.uplift import TLearnerUplift, FEATURE_COLS, classify_quadrant
from src.data.generator import generate_causal_data


# Global in-memory trained estimator
estimator: TLearnerUplift | None = None
MARGIN_RATE = 0.40
TREATMENT_COST = 2.50
BREAKEVEN_TAU = TREATMENT_COST / MARGIN_RATE  # $6.25


def load_or_train_model():
    """Initialize and warm up T-Learner estimator."""
    global estimator
    df = generate_causal_data(n_samples=5000).to_pandas()
    X = df[FEATURE_COLS]
    t = df["treatment"].values
    y = df["outcome"].values
    estimator = TLearnerUplift(n_estimators=50).fit(X, t, y)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_or_train_model()
    yield


app = FastAPI(
    title="CausalOptima Policy Recommendation Engine",
    version="0.1.0",
    description="Real-Time Algorithmic Treatment Policy Engine using Causal Uplift Modeling",
    lifespan=lifespan,
)


class UserFeatures(BaseModel):
    user_id: str = Field(..., json_schema_extra={"example": "usr_10492"})
    tenure_months: int = Field(..., ge=0, json_schema_extra={"example": 18})
    baseline_spend_usd: float = Field(..., ge=0.0, json_schema_extra={"example": 85.50})
    recency_days: int = Field(..., ge=0, json_schema_extra={"example": 24})
    frequency_orders: int = Field(..., ge=1, json_schema_extra={"example": 5})
    discount_sensitivity: float = Field(..., ge=0.0, le=1.0, json_schema_extra={"example": 0.82})


class RecommendationResponse(BaseModel):
    user_id: str
    action: Literal["TREAT", "SUPPRESS", "HOLD"]
    estimated_uplift_usd: float
    causal_quadrant: str
    expected_net_margin_usd: float
    reason: str


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": estimator is not None,
        "breakeven_tau": BREAKEVEN_TAU,
    }


@app.post("/api/v1/recommend-treatment", response_model=RecommendationResponse)
def recommend_treatment(user: UserFeatures):
    if estimator is None:
        raise HTTPException(status_code=503, detail="Model is not initialized.")

    row_df = pd.DataFrame([user.model_dump()])[FEATURE_COLS]
    cate = float(estimator.predict_cate(row_df)[0])
    baseline = float(estimator.predict_baseline(row_df)[0])
    quadrant = str(classify_quadrant(np.array([cate]), np.array([baseline]))[0])

    net_margin = round(cate * MARGIN_RATE - TREATMENT_COST, 2)

    if quadrant == "Persuadable" and cate > BREAKEVEN_TAU:
        action = "TREAT"
        reason = f"Positive incrementality ({cate:.2f}$) exceeds breakeven threshold ({BREAKEVEN_TAU:.2f}$)."
    elif quadrant == "Sleeping Dog":
        action = "SUPPRESS"
        reason = f"Negative incrementality ({cate:.2f}$); marketing triggers churn/disengagement."
    else:
        action = "HOLD"
        reason = f"Zero/insufficient lift ({cate:.2f}$); user is a {quadrant}."

    return RecommendationResponse(
        user_id=user.user_id,
        action=action,
        estimated_uplift_usd=round(cate, 2),
        causal_quadrant=quadrant,
        expected_net_margin_usd=net_margin if action == "TREAT" else 0.0,
        reason=reason,
    )
