"""DoWhy Structural Causal DAG Modeling, Identification, and Refutation."""

from typing import Any, Dict
import dowhy
from dowhy import CausalModel
import pandas as pd


CAUSAL_GRAPH_DOT = """
digraph {
    tenure_months -> treatment;
    baseline_spend_usd -> treatment;
    tenure_months -> outcome;
    baseline_spend_usd -> outcome;
    recency_days -> treatment;
    recency_days -> outcome;
    frequency_orders -> outcome;
    discount_sensitivity -> outcome;
    treatment -> outcome;
}
"""


def build_and_identify_causal_model(df: pd.DataFrame) -> tuple[CausalModel, Any]:
    """Construct DoWhy CausalModel and identify causal estimand via Backdoor Criterion."""
    model = CausalModel(
        data=df,
        treatment="treatment",
        outcome="outcome",
        graph=CAUSAL_GRAPH_DOT.replace("\n", " "),
    )
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    return model, estimand


def run_refutation_suite(
    model: CausalModel,
    estimate: Any,
    num_simulations: int = 5,
) -> Dict[str, Any]:
    """Execute Placebo Treatment and Random Common Cause refutations to test identification robustness."""
    # 1. Placebo Treatment Refuter (New effect must be ~0)
    placebo_ref = model.refute_estimate(
        estimand=estimate.estimand,
        estimate=estimate,
        method_name="placebo_treatment_refuter",
        num_simulations=num_simulations,
    )

    # 2. Random Common Cause Refuter (Effect should not drastically change)
    rcc_ref = model.refute_estimate(
        estimand=estimate.estimand,
        estimate=estimate,
        method_name="random_common_cause",
        num_simulations=num_simulations,
    )

    return {
        "placebo_p_value": placebo_ref.refutation_result.get("p_value", 1.0)
        if isinstance(placebo_ref.refutation_result, dict)
        else 1.0,
        "placebo_new_effect": placebo_ref.new_effect,
        "rcc_new_effect": rcc_ref.new_effect,
        "rcc_p_value": rcc_ref.refutation_result.get("p_value", 1.0)
        if isinstance(rcc_ref.refutation_result, dict)
        else 1.0,
    }


if __name__ == "__main__":
    from src.data.generator import generate_causal_data

    sample_df = generate_causal_data(n_samples=1000).to_pandas()
    c_model, c_estimand = build_and_identify_causal_model(sample_df)
    c_estimate = c_model.estimate_effect(
        c_estimand,
        method_name="backdoor.linear_regression",
    )
    print("Identified effect:", c_estimate.value)
    assert c_estimate.value is not None
