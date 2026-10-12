# EstatePulse: Real Estate Price Prediction & Dynamic Property Matchmaker

An integrated Data Mining and Artificial Intelligence system combining:
1. **Data Mining (Regression Modeling)**: Multi-model linear and non-linear regression suite for property valuation.
2. **Artificial Intelligence (CSP Engine)**: Constraint Satisfaction Problem solver using forward checking, domain pruning, backtracking, and soft constraint relaxation for property matching.
3. **Interactive Web Application**: Responsive dark glassmorphism dashboard built with FastAPI and modern Vanilla web technologies.

---

## Architecture Overview

```
c:\data mining\
│
├── app.py                      # FastAPI REST API & Web Application Server
├── run_demo.py                 # Comprehensive Terminal Verification Script
│
├── data/
│   ├── dataset_generator.py    # Synthetic Real Estate Data Mining Generator
│   └── real_estate_data.csv    # 650 Curated Property Listings
│
├── models/
│   ├── regression_models.py    # ML Suite: SLR, MLR, Polynomial, Ridge, Random Forest
│   ├── model_trainer.py        # Model Training & Metric Evaluation Pipeline
│   └── trained_suite.joblib    # Serialized Model Artifacts
│
├── ai_csp/
│   ├── csp_problem.py          # Formal CSP Definition (Variables, Domains, Constraints)
│   └── csp_solver.py           # Dynamic Matchmaker (Forward Checking, Trace, Soft Relaxation)
│
└── static/
    ├── index.html              # Interactive Dashboard
    ├── style.css               # Glassmorphic Dark Design System
    └── app.js                  # Dynamic Application Logic & Chart Integrations
```

---

## 1. Data Mining: Price Estimation Models

The regression suite includes 5 comparative models trained on 650 realistic listings:

- **Simple Linear Regression (SLR)**: OLS formulation modeling price exclusively against living area:
  $$\text{Price} = \beta_0 + \beta_1 \cdot \text{sqft}$$
- **Multiple Linear Regression (MLR)**: Evaluates square footage, bedrooms, bathrooms, age, garage, commute distance, and one-hot encoded locations and amenities.
- **Polynomial Regression (Degree 2)**: Captures non-linear feature interactions and compounding returns.
- **Ridge Regression (L2 Regularization)**: Prevents overfitting across collinear geographic and amenity indicators.
- **Random Forest Regressor (Non-Linear Ensemble)**: 100 decision trees discovering non-linear price frontiers.

### Evaluation Metrics Summary
- **$R^2$ Score**: Up to $0.980$ (Polynomial) / $0.926$ (Random Forest)
- **RMSE & MAE**: Strict dollar error benchmarking
- **80% Confidence Intervals**: Estimated per prediction based on residual variance

---

## 2. Artificial Intelligence: CSP Property Matchmaker

Formalized as a Constraint Satisfaction Problem $\langle X, D, C \rangle$:
- **Variable $X$**: Property assignment $P$
- **Domain $D(P)$**: Universe of catalog properties ($|D| = 650$)
- **Hard Constraints $C_{hard}$**:
  - $C_{\text{budget}}: \text{price}(P) \le \text{max\_budget}$
  - $C_{\text{location}}: \text{location}(P) \in \text{preferred\_locations}$
  - $C_{\text{bedrooms}}: \text{bedrooms}(P) \ge \text{min\_bedrooms}$
  - $C_{\text{bathrooms}}: \text{bathrooms}(P) \ge \text{min\_bathrooms}$
  - $C_{\text{amenities}}: \text{essential\_amenities} \subseteq \text{amenities}(P)$
  - $C_{\text{commute}}: \text{commute}(P) \le \text{max\_commute}$

### Forward Checking & Domain Pruning
Step-by-step constraint propagation eagerly filters inconsistent domain values:
$$\text{Universe (650)} \xrightarrow{C_{\text{budget}}} D_1 \xrightarrow{C_{\text{loc}}} D_2 \xrightarrow{C_{\text{rooms}}} D_3 \xrightarrow{C_{\text{am}}} D_{final}$$

### Soft Constraint Relaxation (Over-Constrained Queries)
When strict satisfaction yields zero candidates ($|D| = 0$), the engine engages tiered relaxation:
- **Tier 1**: $+10\%$ budget margin or $1$ missing essential amenity.
- **Tier 2**: $+20\%$ budget margin or $1$ bedroom allowance.
- **Satisfaction Scoring**: Scores each candidate ($0-100\%$) incorporating budget efficiency, desired amenity coverage, and proximity bonuses.

---

## 3. Running the Project

### Start the Web Application
```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

### Run Terminal Demonstration
```bash
python run_demo.py
```
