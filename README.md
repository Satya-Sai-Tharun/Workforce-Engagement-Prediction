# Predicting Workforce Engagement Levels

## 📊 Overview

This project builds a high-performance machine learning predictive model to classify employee engagement levels based on historical attendance, employee demographics, workload, health data, and commuting metrics. The solution is designed for Human Resources (HR) teams to proactively identify at-risk personnel and implement data-driven intervention strategies.

## 🎯 Objective & Target

The primary objective is to accurately predict engagement categories derived from employee absence duration:

- `0` → **Highly Engaged** (0 hours absence)
- `1` → **Moderately Engaged** (1–8 hours absence)
- `2` → **At-Risk** (> 8 hours absence)

**Target Metrics:** Achieve **Accuracy > 85%** and a **Weighted F1-Score > 85%** on the validation data.

## 🛠 Project Pipeline

The analytical pipeline follows an end-to-end data lifecycle:

1. **Data Cleaning:** Fixing anomalous data types, handling missing variables, and replacing anomalous categories.
2. **Outlier Treatment:** Capping outlier absence durations using 3×IQR winsorization.
3. **Feature Engineering:** Creating 12 domain-informed features such as:
   - `health_risk_score`, `commute_burden`
   - `workload_per_performance`, `lifestyle_blend`
   - `age_workload_interaction`
4. **Data Balancing:** Applying `SMOTETomek` (Synthetic Minority Over-sampling Technique + Tomek Links) to handle severe class imbalances without generating noisy borders.
5. **Ensemble Model Training:** Training and cross-validating an array of powerful classifiers.
6. **Evaluation & Actionable Exports:** Producing a robust analytical dataset for Business Intelligence dashboards (Tableau/Power BI).

## 🧠 Machine Learning Models

The following classification algorithms are trained and compared:

- Decision Tree
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- XGBoost
- LightGBM
- Soft Voting Ensemble
- Stacking Classifier

**Output Evaluation:** A model comparison leaderboard evaluates training scenarios to identify the ultimate top-performing framework.

## 📈 Human Resources Actionability

Beyond model metrics, the framework is designed to deliver immediate stakeholder value:

- **HR Dashboarding:** In-notebook visualizations of high-risk segments (e.g., Risk-factor heatmaps of Age Band × Tenure Band).
- **Forecasting:** Inference capability on new data to output the predicted class probabilities per employee.
- **Actionable Insights:** Disseminating the highest-ranking feature importances into a priority action plan.
- **BI Export:** Generation of a fully engineered `.csv` ready for seamless Tableau/Power BI integration with mapped confidence scores and human-readable classes.

## 📂 Dataset Dictionary

Key features included within the model (before transformation):

| Feature | Description |
|---|---|
| `absence_reason_code` | Coded reason for absence |
| `commute_cost`/`commute_distance` | Monetary & distance commute metrics |
| `years_at_company` | Employee tenure in years |
| `employee_age` | Employee age |
| `daily_workload` | Average daily workload score |
| `performance_target` | Performance KPI target |
| `disciplinary_action` | Flag indicating past disciplinary action |
| `alcohol_consumption` / `tobacco_use` | Personal lifestyle indicators |
| `body_weight_kg`/`body_height_cm`/`bmi_score` | Health metrics |

## 🚀 Getting Started

### Prerequisites

Before running the notebook, ensure your Python environment has the following requisite packages installed. You can install all dependencies via `pip`:

```bash
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn xgboost lightgbm
```

*(Note: On Mac/macOS you may also need to install `libomp` via Homebrew for XGBoost compatibility: `brew install libomp`)*

### Execution

Simply run the Jupyter Notebook (`Workforce_Engagement_Prediction.ipynb`) in a cohesive Python environment sequentially. The process finishes by outputting a fully prepared dataset with predictions and confidence metrics that can act as the source system for your broader HR dashboard stack.
