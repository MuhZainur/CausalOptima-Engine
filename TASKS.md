# Task Backlog & Execution Status
## CausalOptima-Engine

---

| Task ID | Task Description | Status | Assignee | Deliverable |
| :--- | :--- | :---: | :---: | :--- |
| **TSK-CSL-01** | Create project scaffolding & pyproject.toml | ✅ Completed | System | Directory structure & config |
| **TSK-CSL-02** | Complete PRD, Architecture, and Theory specs | ✅ Completed | System | Markdown blueprints |
| **TSK-CSL-03** | Implement synthetic causal dataset generator | ⏳ Pending | `@causal-ml` | `src/data/generator.py` |
| **TSK-CSL-04** | Formulate DoWhy causal DAG & identification test | ⏳ Pending | `@econometrician` | `src/causal/dag_model.py` |
| **TSK-CSL-05** | Implement Meta-Learners (S, T, X) with LightGBM | ⏳ Pending | `@causal-ml` | `src/models/meta_learners.py` |
| **TSK-CSL-06** | Implement Double Machine Learning & Causal Forests | ⏳ Pending | `@causal-ml` | `src/models/dml_engine.py` |
| **TSK-CSL-07** | Implement DoWhy Placebo and Confounder Refuters | ⏳ Pending | `@econometrician` | `src/validation/refuter.py` |
| **TSK-CSL-08** | Implement Qini curve & cumulative gain calculator | ⏳ Pending | `@decision-science` | `src/evaluation/uplift_metrics.py` |
| **TSK-CSL-09** | Build optimal policy assignment & profit optimizer | ⏳ Pending | `@decision-science` | `src/policy/profit_optimizer.py` |
| **TSK-CSL-10** | Implement FastAPI serving & Streamlit decision UI | ⏳ Pending | `@decision-science` | `src/api/main.py` & `src/dashboard/app.py` |
