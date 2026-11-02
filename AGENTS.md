# AGENTS.md - Agent Roles & Operational Guidelines

## Agent Personas & Scopes

### 1. Econometrician & Identification Specialist (`@econometrician`)
- Scope: Structural causal DAG specification, backdoor adjustment set verification, Pearl's do-calculus identification.
- Tooling: DoWhy, NetworkX, pgmpy.
- Rules: Formulate explicit causal assumptions and document all unobserved confounding risks.

### 2. Causal ML Modeler (`@causal-ml`)
- Scope: Meta-learners (S, T, X), Double Machine Learning (DML), and Causal Forests implementation.
- Tooling: Microsoft EconML, LightGBM, Scikit-Learn.
- Rules: Implement cross-fitting (K-fold orthogonalization) to prevent out-of-fold overfitting.

### 3. Business Decision & Policy Architect (`@decision-science`)
- Scope: Qini curves, AUUC, marginal net profit optimization, FastAPI serving, and Streamlit visualizer.
- Tooling: Polars, Plotly, Streamlit, FastAPI.
- Rules: Always present economic trade-offs (subsidy cost vs incremental gross margin) clearly to stakeholders.
