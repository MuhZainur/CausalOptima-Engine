"""Evaluate uplift policies on synthetic customers whose true treatment effects are known.

Run from the repository root: python scripts/measure_policy.py
Writes reports/policy_benchmark.json. Models fit on 70% of customers; policies are scored
on the other 30% using the true effects, which only synthetic data can provide.
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lightgbm import LGBMRegressor
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from src.data.generator import generate_causal_data
from src.evaluation.metrics import compute_qini_curve
from src.models.uplift import DMLUplift, FEATURE_COLS, TLearnerUplift

SEEDS = [42, 7, 123]
N_SAMPLES = 50000
MARGIN_RATE = 0.40
COST_PER_TREATMENT = 2.50
BREAKEVEN_TAU = COST_PER_TREATMENT / MARGIN_RATE


def profit(mask, true_tau):
    """Net profit of treating `mask`, scored with the true effects."""
    return round(float(true_tau[mask].sum() * MARGIN_RATE - mask.sum() * COST_PER_TREATMENT), 2)


def run_seed(seed):
    df = generate_causal_data(n_samples=N_SAMPLES, seed=seed).to_pandas()
    train, test = train_test_split(df, test_size=0.30, random_state=seed)
    x_train, x_test = train[FEATURE_COLS], test[FEATURE_COLS]
    t_train, y_train = train["treatment"].values, train["outcome"].values
    true_tau = test["true_tau"].values

    t_cate = TLearnerUplift(seed=seed).fit(x_train, t_train, y_train).predict_cate(x_test)
    dml_cate = DMLUplift(seed=seed).fit(y_train, t_train, X=x_train).predict_cate(x_test)
    spend = LGBMRegressor(n_estimators=100, max_depth=4, random_state=seed, verbose=-1)
    spend_pred = spend.fit(x_train, y_train).predict(x_test)

    t_mask = t_cate > BREAKEVEN_TAU
    top_spend = np.zeros(len(test), dtype=bool)
    top_spend[np.argsort(-spend_pred)[: int(t_mask.sum())]] = True

    naive = df.loc[df.treatment == 1, "outcome"].mean() - df.loc[df.treatment == 0, "outcome"].mean()
    design = np.column_stack([df["treatment"].values, df[FEATURE_COLS].values])
    backdoor = LinearRegression().fit(design, df["outcome"].values).coef_[0]

    def estimator_quality(cate):
        mask = cate > BREAKEVEN_TAU
        return {
            "targeted_share": round(float(mask.mean()), 3),
            "mean_true_effect_of_targeted": round(float(true_tau[mask].mean()), 2),
            "correlation_with_true_effect": round(float(np.corrcoef(cate, true_tau)[0, 1]), 3),
            "rmse": round(float(np.sqrt(np.mean((cate - true_tau) ** 2))), 2),
            "qini_auuc": round(compute_qini_curve(test["outcome"].values, test["treatment"].values, cate)[2], 0),
        }

    groups = df.groupby("quadrant")["true_tau"].agg(["size", "mean"])

    return {
        "seed": seed,
        "groups": {
            name: {"share": round(row["size"] / len(df), 3), "mean_true_effect": round(row["mean"], 2)}
            for name, row in groups.iterrows()
        },
        "estimators": {"t_learner": estimator_quality(t_cate), "double_ml": estimator_quality(dml_cate)},
        "average_effect": {
            "true": round(float(df["true_tau"].mean()), 2),
            "naive_difference": round(float(naive), 2),
            "backdoor_adjusted_ols": round(float(backdoor), 2),
        },
        "holdout_customers": len(test),
        "targeted_by_t_learner": int(t_mask.sum()),
        "holdout_profit": {
            "oracle_true_effects": profit(true_tau > BREAKEVEN_TAU, true_tau),
            "t_learner": profit(t_mask, true_tau),
            "double_ml": profit(dml_cate > BREAKEVEN_TAU, true_tau),
            "top_predicted_spend_same_budget": profit(top_spend, true_tau),
            "treat_everyone": profit(np.ones(len(test), dtype=bool), true_tau),
        },
    }


def main():
    report = {
        "scope": "Synthetic data with known treatment effects; not a field experiment.",
        "margin_rate": MARGIN_RATE,
        "cost_per_treatment": COST_PER_TREATMENT,
        "customers_per_seed": N_SAMPLES,
        "runs": [run_seed(seed) for seed in SEEDS],
    }
    out = Path(__file__).resolve().parents[1] / "reports" / "policy_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
