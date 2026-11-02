# Decision Science & Campaign ROI Audit
## CausalOptima-Engine: Business Impact Framework

---

### 1. ROI Optimization Model: Maximizing Net Incremental Profit
The optimal targeting threshold $k^*$ is determined by equating marginal incremental revenue to marginal cost:

$$\max_{\pi} \sum_{i=1}^{N} \pi(X_i) \left[ \hat{\tau}(X_i) \cdot \text{Gross Margin} - \text{Cost per Treatment} \right]$$

Subject to the decision rule:
$$\pi(X_i) = \begin{cases} 1 & \text{if } \hat{\tau}(X_i) > \frac{\text{Cost per Treatment}}{\text{Gross Margin}} \\ 0 & \text{otherwise} \end{cases}$$

#### Example Campaign Economics:
- **Treatment Cost:** $\$2.50$ (SMS cost + minimum promo subsidy).
- **Gross Margin:** $40\%$.
- **Breakeven Uplift:** $\tau^* = \frac{2.50}{0.40} = \$6.25$ incremental spend.
- **Audit Decision:** Any customer with estimated $\hat{\tau}_i(X) \le \$6.25$ is suppressed from the campaign, preventing capital destruction.

---

### 2. Statistical Refutation Acceptance Criteria

| Refuter Test | Methodology | Acceptance Standard | Action on Failure |
| :--- | :--- | :--- | :--- |
| **Placebo Treatment** | Replace treatment with random noise $T \sim \text{Bernoulli}(0.5)$. | Estimated effect $|\hat{\tau}| < 0.05$ with $p > 0.10$. | Reject model. Unobserved confounding detected. |
| **Random Common Cause** | Add synthetic random covariate $Z \sim \mathcal{N}(0, 1)$ as confounder. | Estimate variation $|\Delta \hat{\tau}| < 5\%$. | Check for model instability or overfitting. |
| **Data Subset Removal** | Re-estimate effect on 80% bootstrap subsets. | Estimate variation $|\Delta \hat{\tau}| < 8\%$. | Increase sample size or regularize base learners. |
