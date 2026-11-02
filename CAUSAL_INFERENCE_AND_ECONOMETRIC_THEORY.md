# Theoretical Foundations: Causal Inference, Uplift Modeling & Econometrics
## CausalOptima-Engine Technical Deep Dive

---

### 1. The Fundamental Problem of Causal Inference (Holland, 1986)
For any individual customer $i$, we observe only one realized factual outcome:
$$Y_i = T_i \cdot Y_i(1) + (1 - T_i) \cdot Y_i(0)$$
We cannot simultaneously observe customer $i$ receiving the discount and not receiving the discount at time $t$. The counterfactual outcome is permanently missing.

Predictive machine learning circumvents this by modeling the conditional expectation:
$$\mathbb{E}[Y | T=1, X] - \mathbb{E}[Y | T=0, X]$$
However, in observational data, this quantity suffers from **selection bias**:
$$\mathbb{E}[Y | T=1, X] - \mathbb{E}[Y | T=0, X] = \underbrace{\tau(X)}_{\text{True Causal Uplift}} + \underbrace{\mathbb{E}[Y(0) | T=1, X] - \mathbb{E}[Y(0) | T=0, X]}_{\text{Confounding Selection Bias}}$$

If marketing teams preferentially send coupons to active spenders, $\mathbb{E}[Y(0) | T=1] > \mathbb{E}[Y(0) | T=0]$, artificially inflating perceived campaign success.

---

### 2. Meta-Learner Taxonomy: S, T, X, and R-Learners

```mermaid
graph TD
    subgraph SLearner ["1. S-Learner (Single Model)"]
        S_Desc["Model: Y = f(X, T)<br>Tau_i = f(X, 1) - f(X, 0)<br>Risk: Decision trees often ignore T if X has much higher dimensionality, shrinking Tau -> 0."]
    end

    subgraph TLearner ["2. T-Learner (Two Models)"]
        T_Desc["Model 0: Y0 = f0(X) on Control<br>Model 1: Y1 = f1(X) on Treated<br>Tau_i = f1(X) - f0(X)<br>Risk: Differences in regularizer or sample size create artificial uplift noise."]
    end

    subgraph XLearner ["3. X-Learner (Imbalanced Design)"]
        X_Desc["Step 1: Fit f0 and f1.<br>Step 2: Impute counterfactuals D1 = Y1 - f0(X1), D0 = f1(X0) - Y0.<br>Step 3: Fit tau1(X) on D1, tau0(X) on D0.<br>Step 4: Tau_i = e(X)*tau0(X) + (1 - e(X))*tau1(X).<br>Best for: Highly imbalanced treatment cohorts."]
    end

    subgraph RLearner ["4. R-Learner / DML (Residualized)"]
        R_Desc["Robinson (1988) / Nie & Wager (2021):<br>Formulates loss directly on causal parameter using Neyman orthogonal residuals.<br>Guarantees asymptotic efficiency and robustness against complex non-linear confounding."]
    end
```

---

### 3. Business Incrementality & The Qini Curve
The standard Receiver Operating Characteristic (ROC-AUC) curve is useless for uplift because ground truth labels ($Y_i(1) - Y_i(0)$) cannot be observed per individual.

Instead, the **Qini Curve** evaluates policy performance across ranked percentiles $n$:

$$Q(n) = R_T(n) - R_C(n) \cdot \frac{N_T(n)}{N_C(n)}$$
Where:
- $R_T(n)$: Cumulative outcome of treated users in the top $n$ percentile.
- $R_C(n)$: Cumulative outcome of control users in the top $n$ percentile.
- $N_T(n), N_C(n)$: Counts of treated and control users in the top $n$ percentile.

A model with high **Qini Score / AUUC** successfully prioritizes users whose incremental revenue is steepest, allowing the marketing team to target the top 20% of users while capturing 90% of the possible incremental lift.
