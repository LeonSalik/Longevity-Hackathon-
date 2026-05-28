# ML Approach

## 1. Problem Formulation
We frame the task as a patient-level risk stratification problem.

Input: labs, demographics, comorbidities, derived liver scores.
Output: probability of MASH/CKM high-risk status.

## 2. Baseline
Clinical rule-based scores:
- FIB-4
- APRI
- AST/ALT ratio

## 3. Supervised Model
Primary model:
- XGBoost classifier

Why:
- strong for tabular clinical data
- handles nonlinear interactions
- works well with missing/noisy features
- compatible with SHAP explanations

## 4. Unsupervised / Exploratory Model
Autoencoder or clustering to identify unusual patient profiles and hidden risk groups.

## 5. Validation
- train/test split
- AUROC
- precision
- recall
- F1
- confusion matrix
- calibration if time allows

## 6. Explainability
Use SHAP and feature importance to show why a patient was flagged.

## 7. Clinical Integration
The model does not diagnose.
It flags patients for review, further testing, or specialist referral.
