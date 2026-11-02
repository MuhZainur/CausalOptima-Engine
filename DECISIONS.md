# Architectural Decision Records (ADR)
## CausalOptima-Engine

---

### ADR-001: Microsoft EconML for Uplift & CATE Estimation
- **Date:** 2026-11-02
- **Context:** Implementing custom Double Machine Learning and Causal Forests from scratch is prone to subtle statistical bias and lack of asymptotic guarantees.
- **Decision:** Adopt Microsoft `EconML`, the gold-standard open-source library for modern econometric and causal ML algorithms.
- **Consequences:** Access to battle-tested implementations of LinearDML, CausalForestDML, and cross-fitting estimators.

### ADR-002: DoWhy for Causal Identification and Refutation
- **Date:** 2026-11-02
- **Context:** Many ML practitioners fit models without mathematically testing whether the causal effect is identifiable or fragile to unmeasured confounders.
- **Decision:** Use `DoWhy` to explicitly define DAGs and run automated refuter algorithms (Placebo treatment and Random Common Cause).
- **Consequences:** Transparent, auditable scientific defensibility for enterprise executive leadership.

### ADR-003: Qini Curve and AUUC over Standard ROC-AUC
- **Date:** 2026-11-02
- **Context:** Standard classification metrics (ROC-AUC, Precision/Recall) reward models for identifying users who purchase, not users whose purchase was *caused* by the promotion.
- **Decision:** Make Qini score and Net Incremental Profit the primary optimization criteria across the entire pipeline.
- **Consequences:** Direct alignment between model optimization and marketing campaign ROI.
