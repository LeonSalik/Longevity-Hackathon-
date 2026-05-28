# Team Name — MASH/CKM Early Risk Detection

## Problem
Patients with MASH and cardiometabolic-kidney risk are often detected late.

## Our Solution
We propose an ML-based pre-screening pipeline that uses structured clinical/lab data and clinical notes to flag high-risk patients.

## Key Design Decisions
1. Use clinically interpretable biomarkers: AST, ALT, platelets, age, BMI, diabetes, CKD markers.
2. Combine rule-based scores such as FIB-4/APRI with ML predictions.
3. Use XGBoost / Random Forest as baseline due to tabular clinical data.
4. Add autoencoder/anomaly-detection experiment for unsupervised risk discovery.
5. Use SHAP for explainability.

## Repository Structure
Explain folders here.
