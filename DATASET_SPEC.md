# Dataset Specification & Causal Data Generator
## CausalOptima-Engine: Uplift & CATE Benchmark

---

### 1. Data Schema & Feature Structure
The customer dataset consists of 100,000 observational rows containing:

```yaml
features:
  user_id: string (Primary Key)
  tenure_months: integer (User lifetime)
  recency_days: integer (Days since last interaction)
  frequency_orders: integer (Total historical purchases)
  baseline_spend_usd: float (Pre-treatment 90-day gross spend)
  mobile_app_installed: boolean (Platform engagement)
  discount_sensitivity_index: float (Observed price elasticity)

causal_variables:
  W_confounders: [tenure_months, baseline_spend_usd] # Distorts both treatment assignment and purchase
  X_heterogeneity: [recency_days, frequency_orders, discount_sensitivity_index] # Modulates uplift effect
  T_treatment: boolean # 1 = 15% discount coupon sent, 0 = no outreach
  Y_outcome: float # 30-day post-treatment spend (USD)
  ground_truth_tau: float # True synthetic unobservable uplift for validation
  true_quadrant: string # 'Persuadable', 'Sure Thing', 'Lost Cause', 'Sleeping Dog'
```

---

### 2. Embedded Causal Mechanism Formula
The data generator simulates complex non-linear confounding and treatment heterogeneity:

$$\text{Propensity}(T=1 | W) = \sigma\left( -0.5 + 0.02 \cdot \text{tenure} + 0.005 \cdot \text{baseline\_spend} \right)$$

$$\tau(X) = 25.0 \cdot \mathbf{1}_{X \in \text{Persuadables}} - 15.0 \cdot \mathbf{1}_{X \in \text{SleepingDogs}} + 0.0 \cdot \mathbf{1}_{X \in \{\text{SureThings}, \text{LostCauses}\}}$$

$$Y = 10.0 + 1.2 \cdot \text{baseline\_spend} + \tau(X) \cdot T + \epsilon \quad \text{where } \epsilon \sim \mathcal{N}(0, 5)$$

---

### 3. Verification Protocol
Because ground-truth treatment effects are unobservable in real-world observational data, this semi-synthetic formulation allows exact benchmarking:
- Mean Squared Error of CATE: $\text{MSE}(\hat{\tau}, \tau_{\text{true}})$.
- Quadrant classification accuracy: F1-score of detecting true Persuadables vs Sleeping Dogs.
