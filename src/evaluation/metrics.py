"""Causal Uplift Metrics: Qini Curve, AUUC, and Incremental Campaign ROI."""

from typing import Dict, Tuple
import numpy as np


def compute_qini_curve(
    y_true: np.ndarray,
    treatment: np.ndarray,
    cate_pred: np.ndarray,
    n_bins: int = 100,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Compute Qini curve coordinates and Qini AUUC score."""
    # Sort descending by predicted uplift
    order = np.argsort(-cate_pred)
    y_sorted = y_true[order]
    t_sorted = treatment[order]

    n = len(y_true)
    bin_sizes = np.linspace(n // n_bins, n, n_bins, dtype=int)

    qini_vals = [0.0]
    percentiles = [0.0]

    for k in bin_sizes:
        sub_y = y_sorted[:k]
        sub_t = t_sorted[:k]

        n_t = np.sum(sub_t == 1)
        n_c = np.sum(sub_t == 0)

        if n_t == 0 or n_c == 0:
            qini_vals.append(0.0)
        else:
            r_t = np.sum(sub_y[sub_t == 1])
            r_c = np.sum(sub_y[sub_t == 0])
            q = r_t - r_c * (n_t / n_c)
            qini_vals.append(q)

        percentiles.append(k / n)

    # Compute AUUC using trapezoidal rule (NumPy 2.0 compatible)
    trap_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    auuc = float(trap_fn(qini_vals, percentiles))
    return np.array(percentiles), np.array(qini_vals), auuc


def compute_campaign_profit(
    cate_pred: np.ndarray,
    margin_rate: float = 0.40,
    cost_per_treatment: float = 2.50,
) -> Dict[str, float]:
    """Calculate net incremental profit under optimal threshold decision rule."""
    # Breakeven condition: tau_i * margin_rate > cost_per_treatment
    breakeven_tau = cost_per_treatment / margin_rate
    target_mask = cate_pred > breakeven_tau

    incremental_revenue = np.sum(cate_pred[target_mask])
    gross_profit = incremental_revenue * margin_rate
    treatment_costs = np.sum(target_mask) * cost_per_treatment
    net_profit = gross_profit - treatment_costs

    return {
        "breakeven_tau": round(breakeven_tau, 2),
        "targeted_users_count": int(np.sum(target_mask)),
        "targeted_fraction": round(float(np.mean(target_mask)), 4),
        "incremental_revenue_usd": round(float(incremental_revenue), 2),
        "net_profit_usd": round(float(net_profit), 2),
        "roi_multiple": round(float(net_profit / max(treatment_costs, 1e-5)), 2),
    }


if __name__ == "__main__":
    y_dummy = np.array([10, 20, 0, 5, 15, 30, 2, 8])
    t_dummy = np.array([1, 1, 0, 0, 1, 1, 0, 0])
    cate_dummy = np.array([8.0, 12.0, -2.0, 1.0, 5.0, 15.0, -1.0, 3.0])

    pct, qini, auuc = compute_qini_curve(y_dummy, t_dummy, cate_dummy, n_bins=4)
    assert len(qini) == 5
    profit_res = compute_campaign_profit(cate_dummy, margin_rate=0.40, cost_per_treatment=2.0)
    assert profit_res["targeted_users_count"] > 0
    print("Qini AUUC:", auuc)
    print("Profit:", profit_res)
