# Product Requirements Document (PRD)
## CausalOptima-Engine: Algorithmic Causal AI & Uplift Decision Platform

---

### 1. Executive Summary & Problem Statement
In conventional corporate marketing, retention, and pricing operations, data science teams deploy standard predictive classification models (e.g., Logistic Regression, XGBoost churn predictors) to answer: *"Which customers are most likely to churn?"* or *"Which customers have the highest propensity to purchase?"*

Targeting promotions based on raw prediction probabilities leads to catastrophic ROI destruction:
1. **Subsidizing "Sure Things":** High-propensity customers who would have purchased regardless of a coupon receive expensive discounts, cannibalizing profit margins.
2. **Wasting Budget on "Lost Causes":** High-churn-risk users who are already determined to leave receive retention incentives that fail to alter their behavior.
3. **Triggering "Sleeping Dogs" (Do-Not-Disturbs):** Users who are irritated by marketing outreach and churn *because* they received an email or push notification.

**CausalOptima-Engine** transitions enterprise decision-making from passive correlation to **active causal intervention**. Utilizing Microsoft **EconML**, **DoWhy**, Double Machine Learning (DML), and Causal Forests, the platform isolates **Heterogeneous Treatment Effects (HTE)** at the individual customer level (Conditional Average Treatment Effect - CATE). It segments the customer base into four distinct causal quadrants and outputs optimal personalized treatment policies (e.g., "Give 15% discount only to Persuadables; suppress outreach to Sleeping Dogs and Sure Things").

---

### 2. User Personas & Target Audiences
1. **Chief Marketing Officer (CMO) / Growth Director:** Aims to maximize campaign incremental gross margin per marketing dollar spent, avoiding wasteful coupon blanket blasts.
2. **Lead Econometrician / Senior Business Data Scientist:** Demands robust identification strategies (backdoor adjustments, instrumental variables, refutation tests) to prove causal validity over confounding bias.
3. **Product Manager (Incentives & Monetization):** Requires a production-ready REST API returning real-time personalized policy assignments for active user sessions.

---

### 3. Core Functional Requirements

#### FR-1: Structural Causal Graph Modeling & Identification (DoWhy)
- Define and visualize Directed Acyclic Graphs (DAGs) capturing relationships between:
  - **Confounders ($W$):** User age, tenure, baseline spending, device type.
  - **Treatment ($T$):** Marketing incentive (0% baseline, 10% promo, 25% promo).
  - **Outcome ($Y$):** 30-day incremental revenue / conversion.
  - **Effect Modifiers ($X$):** Recency, frequency, platform engagement.
- Execute automated backdoor criterion identification and Pearl's do-calculus transformations.

#### FR-2: Advanced Uplift Modeling & Double Machine Learning (EconML)
- Implement multi-meta-learner comparison suite:
  - **S-Learner (Single Model):** Standard baseline pooling treatment with features.
  - **T-Learner (Two Models):** Independent estimators for control vs treated cohorts.
  - **X-Learner:** Optimized two-stage estimator designed for extreme sample size imbalances ($T \ll C$).
  - **Double Machine Learning (DML / R-Learner):** Orthogonalized residual-on-residual regression eliminating regularized first-stage bias.
  - **Causal Forest (Generalized Random Forests):** Non-parametric tree partitioner maximizing heterogeneity of treatment effects.

#### FR-3: Rigorous Causal Refutation & Sensitivity Auditing
- Stress-test estimated causal effects against unobserved confounding via automated refutation tests:
  - **Random Common Cause:** Inject synthetic confounders to verify effect stability.
  - **Placebo Treatment:** Replace true treatment with random permutation (estimated effect must drop to 0).
  - **Data Subset Removal:** Verify estimate stability across 80% bootstrap subsets.

#### FR-4: Business Incrementality & Qini Metric Optimization
- Compute cumulative gain curves and **Qini Curves**:
  $$Q(n) = Y^T(n) - Y^C(n) \frac{N^T(n)}{N^C(n)}$$
- Quantify **AUUC (Area Under the Uplift Curve)** and **Qini Score** against random targeting and naive propensity targeting.
- Generate optimal budget allocation cutoff curves that maximize net incremental profit:
  $$\text{Net Profit} = \sum_{i \in \text{Targeted}} \left( \hat{\tau}(X_i) \cdot \text{Margin} - \text{Cost}(T) \right)$$

#### FR-5: Production Policy Serving API & Interactive Visualizer
- Serve optimal treatment decisions via FastAPI (`POST /api/v1/recommend-treatment`).
- Provide Streamlit dashboard for business stakeholders:
  - Visual Causal DAG viewer.
  - Interactive Qini vs Churn ROC curve comparisons.
  - Causal Quadrant Breakdown (Sure Things, Persuadables, Lost Causes, Sleeping Dogs).
  - Campaign Profit Simulator slider (Cost per contact vs Incremental margin).

---

### 4. Non-Functional Requirements & Performance SLAs

| Metric | Target SLA | Critical Failure Threshold |
| :--- | :--- | :--- |
| **Policy Recommendation Latency (P99)** | $\le 20\text{ ms}$ per user | $> 50\text{ ms}$ |
| **Qini Curve Advantage vs Naive Model** | $\ge +25.0\%\text{ AUUC}$ | $\le 0\%$ |
| **Placebo Refuter p-value** | $\ge 0.90$ (Fail to reject null of zero effect) | $< 0.05$ (Broken identification) |
| **CATE Estimation Batch Speed** | $\ge 100,000\text{ users/min}$ | $< 10,000\text{ users/min}$ |
