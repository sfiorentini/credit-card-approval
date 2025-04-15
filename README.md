# Credit Card Approval Risk Prediction Mode

## Project Overview
This project predicts credit card approval decisions by merging application and credit history data, analyzing risk patterns, and building a conservative model that prioritizes risk minimization given extreme class imbalance (only 1.9% had credit history).

---

## Key Questions Addressed

### 1. Dataset Merging Strategy

### Convert days to years for interpretability
application_df['AGE_YEARS'] = round(abs(application_df['DAYS_BIRTH']) / 365, 1)
application_df['YEARS_EMPLOYED'] = round(abs(application_df['DAYS_EMPLOYED']) / 365, 1)

### Map worst payment status from credit records
def get_credit_status(credit_df):
    status_map = {'C': -1, '0': 0, '2': 2, '3': 3, '4': 4, '5': 5}  # Risk scale
    credit_df['STATUS_NUM'] = credit_df['STATUS'].map(status_map)
    return credit_df.groupby('ID')['STATUS_NUM'].max()  # Worst status per client

merged_data = pd.merge(
    application_df,
    credit_df.groupby('ID')['STATUS_NUM'].max().reset_index(),
    on='ID',
    how='left'
)
merged_data['WORST_STATUS'] = merged_data['STATUS_NUM'].fillna(-2)  # -2 = No history
Key Steps:

Unified records via ID with left join to preserve all applicants.

Engineered WORST_STATUS (-2=No history, -1=Paid, 0-5=Delinquency severity).

Smart imputation for 30% missing occupations using education/income/family status.

### 2. Key Insights
Finding	Business Implication
Top income quintile has 2x higher delinquency	High income ≠ lower risk
"Laborers" show highest delinquency rates	Occupation-based risk tiers needed
92.45% lacked credit history	Hybrid approach (model + business rules) critical
Risk by Occupation
Delinquency rates by occupation (existing clients)

### 3. Approval Recommendations & Risk Control
Model Performance (Optimal Threshold: 0.599):

              precision    recall  f1-score   support
        0       0.99      0.99      0.99      9810  # Correct rejections
        1       0.63      0.71      0.67       190  # Controlled approvals
Risk Mitigation Framework:

Existing Clients:

Auto-reject if WORST_STATUS >= 2 (60+ days delinquency).

Extra scrutiny for "Laborers" and "Drivers".

New Applicants:

Approve only if RISK_SCORE <= 75th percentile.

Tiered credit limits (e.g., $500 for medium-risk scores).

System Design:

def approve_credit(client_data):
    proba = model.predict_proba(client_data)[0, 1]
    return "REJECT" if proba >= 0.599 else "APPROVE"  # Strict threshold
Why It Works:

63% approval precision: Balances risk while allowing manageable false approvals.

71% recall: Captures most creditworthy applicants without excessive exposure.

Technical Implementation
Data Challenges:

Extreme class imbalance (92.45% no credit history).

Non-linear risk-income relationship.

Model Selection:

Random Forest with class weighting (1:10 penalty for minority class).

Engineered features: INCOME_PER_MEMBER, EMPLOYMENT_STABILITY.

Stratified 5-fold CV (AUC: 0.787 ± 0.012).

Conservative Approach Justification:
Given the high cost of bad debt, we prioritized:

High rejection precision (99%).

Moderate approval recall (71%) to avoid losing good clients.

Explicit risk thresholds over accuracy metrics.

How to Reproduce
Install dependencies:

bash
Copy
pip install pandas scikit-learn matplotlib
Run analysis:

bash
Copy
python credit_risk_analysis.py
Production Recommendations:

Manual review for borderline cases (0.55 < risk < 0.65).

Quarterly threshold recalibration with new data.
