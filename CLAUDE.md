# CLAUDE.md - Developer Guidelines for CausalOptima-Engine

## Project Overview
CausalOptima-Engine is an enterprise-grade Causal AI and Uplift Modeling decision platform utilizing Microsoft EconML, DoWhy, Double Machine Learning (DML), and Causal Forests to compute individual treatment effects and optimize campaign ROI.

## Development Commands
- Install dependencies: `uv pip install -e ".[dev]"`
- Generate synthetic data: `python -m src.data.generator`
- Run test suite: `pytest tests/ -v`
- Run linting: `ruff check src/ tests/`
- Start decision API: `uvicorn src.api.main:app --reload --port 8000`
- Launch Streamlit dashboard: `streamlit run src/dashboard/app.py`

## Architecture & Code Invariants
- Never confuse correlation with causation: feature importance in standard XGBoost must never be used to justify policy decisions.
- Every causal model production candidate must successfully pass all three DoWhy refutation tests (Placebo, Random Common Cause, Subset Removal) before deployment.
- Uplift evaluations must report Qini AUUC and Cumulative Net Profit alongside traditional statistical metrics.
- Policy recommendation endpoints must respond in $\le 20\text{ ms}$ P99.
