"""Synthetic Customer Cohort Generator with Embedded Ground-Truth Causal Mechanisms."""

import numpy as np
import polars as pl


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_causal_data(n_samples: int = 50000, seed: int = 42) -> pl.DataFrame:
    """Generate observational customer data with known treatment effects and 4 causal quadrants."""
    rng = np.random.default_rng(seed)

    # 1. Confounders W (affect both treatment assignment and baseline spend)
    tenure_months = rng.integers(1, 60, size=n_samples)
    baseline_spend = rng.gamma(shape=2.5, scale=25.0, size=n_samples)  # Mean ~$62.5

    # 2. Heterogeneity Features X (modulate response to treatment)
    recency_days = rng.integers(1, 180, size=n_samples)
    frequency_orders = rng.poisson(lam=3.0, size=n_samples) + 1
    discount_sensitivity = rng.uniform(0.0, 1.0, size=n_samples)

    # 3. Four Causal Quadrants Assignment based on latent customer persona
    # - Persuadables: high sensitivity, moderate recency -> positive lift
    # - Sleeping Dogs: low sensitivity, high tenure/frequency -> negative lift (churn when bugged)
    # - Sure Things: high frequency, low recency -> buy anyway (zero lift, high baseline)
    # - Lost Causes: high recency (dormant), low frequency -> won't buy (zero lift, zero baseline)
    latent_score = 0.5 * discount_sensitivity - 0.003 * recency_days + 0.02 * frequency_orders

    quadrant = np.empty(n_samples, dtype=object)
    tau_true = np.zeros(n_samples, dtype=np.float64)

    is_persuadable = (discount_sensitivity > 0.55) & (recency_days < 90)
    is_sleeping_dog = (discount_sensitivity < 0.20) & (frequency_orders > 4)
    is_sure_thing = (recency_days < 30) & (frequency_orders > 5) & ~is_persuadable & ~is_sleeping_dog
    is_lost_cause = ~(is_persuadable | is_sleeping_dog | is_sure_thing)

    quadrant[is_persuadable] = "Persuadable"
    tau_true[is_persuadable] = 25.0 + 10.0 * discount_sensitivity[is_persuadable]

    quadrant[is_sleeping_dog] = "Sleeping Dog"
    tau_true[is_sleeping_dog] = -15.0 - 5.0 * (1.0 - discount_sensitivity[is_sleeping_dog])

    quadrant[is_sure_thing] = "Sure Thing"
    tau_true[is_sure_thing] = 0.0

    quadrant[is_lost_cause] = "Lost Cause"
    tau_true[is_lost_cause] = 0.0

    # 4. Observational Treatment Assignment (Confounded Propensity)
    # Marketers preferentially send promotions to higher-spending / longer-tenure users
    logit_p = -0.5 + 0.015 * tenure_months + 0.008 * baseline_spend - 0.01 * recency_days
    propensity = sigmoid(logit_p)
    treatment = rng.binomial(1, propensity, size=n_samples)

    # 5. Realized Outcome Y (30-day post-treatment spend)
    # Untreated baseline Y(0)
    y0 = 8.0 + 0.7 * baseline_spend + 4.0 * np.log1p(frequency_orders) - 0.05 * recency_days
    y0 = np.clip(y0, 0.0, None)
    # Treated outcome Y(1) = Y(0) + tau + noise
    noise = rng.normal(0.0, 4.0, size=n_samples)
    outcome = np.clip(y0 + treatment * tau_true + noise, 0.0, None)

    df = pl.DataFrame({
        "user_id": [f"usr_{i:06d}" for i in range(n_samples)],
        "tenure_months": tenure_months,
        "baseline_spend_usd": baseline_spend,
        "recency_days": recency_days,
        "frequency_orders": frequency_orders,
        "discount_sensitivity": discount_sensitivity,
        "treatment": treatment,
        "outcome": outcome,
        "propensity": propensity,
        "true_tau": tau_true,
        "quadrant": quadrant,
    })

    return df


if __name__ == "__main__":
    df = generate_causal_data(n_samples=5000)
    assert df.height == 5000
    assert "Persuadable" in df["quadrant"].unique().to_list()
    assert "Sleeping Dog" in df["quadrant"].unique().to_list()
    print(f"Generated {df.height} samples successfully.")
    print(df["quadrant"].value_counts())
