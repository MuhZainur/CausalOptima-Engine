# Implementation Plan & Milestone Roadmap
## CausalOptima-Engine: Causal AI & Uplift Decision Platform

---

### Phase 1: Environment Setup & Synthetic Semi-Synthetic Experiment Data (Days 1–2)
- [ ] Initialize Python 3.11 virtual environment with `uv`.
- [ ] Implement `src/data/generator.py`:
  - Synthesize realistic e-commerce customer behavioral dataset ($N=100,000$ rows).
  - Embed known ground-truth causal mechanisms: confounders $W$, effect modifiers $X$, treatment assignment $T$, and distinct causal quadrants (Persuadables, Sure Things, Lost Causes, Sleeping Dogs).
  - Add observational selection bias to test de-biasing capability.

---

### Phase 2: Causal Graph Formulation & DoWhy Identification (Days 3–4)
- [ ] Implement `src/causal/dag_model.py`:
  - Construct network structural DAG using DoWhy.
  - Formulate backdoor criterion identification.
  - Export graphical visualization (`reports/causal_dag.png`).
- [ ] Unit tests: verify DAG acyclicity and identification validity.

---

### Phase 3: Meta-Learner Suite & Double Machine Learning (Days 5–7)
- [ ] Implement `src/models/meta_learners.py`:
  - S-Learner and T-Learner baselines using LightGBM.
  - X-Learner with propensity weighting for imbalanced treatment ratios.
- [ ] Implement `src/models/dml_engine.py`:
  - Double Machine Learning (`LinearDML` and `CausalForestDML`) using Microsoft `EconML`.
  - Extract Individual Treatment Effects $\hat{\tau}_i(X)$ and confidence intervals.
  - Benchmark MSE against ground-truth synthetic $\tau_i$.

---

### Phase 4: Causal Refutation & Sensitivity Auditing (Days 8–9)
- [ ] Implement `src/validation/refuter.py`:
  - Random Common Cause test (inject synthetic confounder; verify $\Delta \hat{\tau} \approx 0$).
  - Placebo Treatment refuter (permute treatment; verify $\hat{\tau} \to 0$ with $p > 0.10$).
  - Data Subset Removal refuter (80% bootstrap stability test).
  - Compile audit summary in `reports/CAUSAL_VALIDATION_REPORT.md`.

---

### Phase 5: Business Incrementality & Qini Curve Optimization (Days 10–11)
- [ ] Implement `src/evaluation/uplift_metrics.py`:
  - Compute Qini Curve, Cumulative Gain, and AUUC.
  - Compare Causal Uplift vs Standard Churn Propensity model.
- [ ] Implement `src/policy/profit_optimizer.py`:
  - Compute optimal threshold $k^*$ maximizing net incremental campaign profit.
  - Classify customer base into the 4 causal quadrants.

---

### Phase 6: FastAPI Serving & Streamlit Decision Suite (Days 12–13)
- [ ] Implement `src/api/main.py`:
  - `POST /api/v1/recommend-treatment`: Ingest user feature vector, return recommended action (`TREAT`, `SUPPRESS`, `HOLD`) and expected incremental ROI.
- [ ] Implement `src/dashboard/app.py` in Streamlit:
  - Interactive ROI calculator slider (adjust promo cost and profit margin).
  - Qini Curve and Cumulative Uplift visualizer.
  - Causal Quadrant distribution breakdown.
