# Data Plan

## 1. Real-world Deployment Data
- EHR demographics
- ICD codes
- Labs: AST, ALT, platelets, glucose, HbA1c, creatinine, eGFR
- Vitals: BMI, blood pressure
- Medications
- Clinical notes

## 2. Prototype Data
For the hackathon, we use a synthetic or sample dataset to demonstrate feasibility.

## 3. Preprocessing
- Remove identifiers
- Normalize lab units
- Handle missing values
- Encode diagnosis categories
- Create derived features

## 4. Feature Engineering
- FIB-4
- APRI
- AST/ALT ratio
- Diabetes indicator
- CKD indicator
- Metabolic syndrome proxy

## 5. Labels
Possible labels:
- confirmed MASH/NAFLD diagnosis
- elevated fibrosis risk
- clinical expert-labeled high risk
- proxy label from diagnosis/lab combinations

## 6. Governance
Data minimization, auditability, no autonomous diagnosis.
